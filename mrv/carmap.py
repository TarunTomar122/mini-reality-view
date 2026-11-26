"""Car map (council/PLAN-carmap.md): detector boxes painted into the third-person chase view, and the side branch.

Map picture (C_MAP = 10 channels at 256x512, then area-pooled to 32x64 for the branch stem):
  0-3  coverage x score for cars / heavy vehicles / people & bikes / obstacles
  4    inverse depth (1 / metres) of the visible box
  5    instance boundary
  6-7  heading of the box relative to our car (sin, cos)
  8    box length / 10 m (MetaDrive picks the car model from length)
  9    availability: 1 where a map exists for this frame (an all-zero map with 0 here = "unavailable")
Projection = the judge's third-person camera (7.5 m behind the ego origin, 2.4 m up, pitch -8 deg, 60 deg HFOV,
512x288, top 32 rows cropped); boxes painted far -> near so nearer boxes hide farther ones; our own car (always the
same pixels) masks everything behind it.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFilter
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull

C_MAP = 10
GROUPS = {"car": 0, "truck": 1, "bus": 1, "trailer": 1, "construction_vehicle": 1,
          "pedestrian": 2, "bicycle": 2, "motorcycle": 2, "barrier": 3, "traffic_cone": 3}
W, H, CROP = 512, 288, 32
_f = W / 2 / math.tan(math.radians(30)); _cp, _sp = math.cos(math.radians(8)), math.sin(math.radians(8))


def project(P):
    """ego points (x fwd, y left, z up) -> (u, v) in the cropped 512x256 image, depth"""
    Yf, R_, Zu = P[:, 0] + 7.5, -P[:, 1], P[:, 2] - 2.4
    d = Yf * _cp - Zu * _sp; up = Yf * _sp + Zu * _cp
    dd = np.maximum(d, 1e-3)
    return np.stack([W / 2 + _f * R_ / dd, H / 2 - _f * up / dd - CROP], 1), d


def _own_mask():
    pts = []
    for x in (-1.05, 1.05):
        for y in (-2.5, 2.5):
            for z in (0.0, 1.6):
                pts.append([y, -x, z])  # ego x fwd, y left
    uv, _ = project(np.array(pts, float)); hull = uv[ConvexHull(uv).vertices]
    m = Image.new("L", (512, 256), 0); ImageDraw.Draw(m).polygon([tuple(q) for q in hull], fill=255)
    return np.asarray(m.filter(ImageFilter.MaxFilter(9))) > 0


OWN = _own_mask()


def box_corners_ego(box, l2e_r, l2e_t, conv="mmdet3d"):
    """box (lidar frame: x, y, z_bottom, w, l, h, yaw, ...) -> 8 corners in the ego frame, heading in ego.
    conv "mmdet3d" = BEVFormer / mmdet3d 0.17 output: its yaw is mirrored and reversed relative to the rotation below,
    so yaw -> -yaw - pi (measured on 1,272 detector-GT car pairs: footprint IoU 0.620 -> 0.667, heading within 15 deg of
    GT 19% -> 96%; Fable's finding, drift check 26 Nov). conv "carmap" = already in this function's convention (gt_frame)."""
    x, y, zb, w, l, h, yaw = box[:7]
    if conv == "mmdet3d":
        yaw = -yaw - math.pi
    xs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * w / 2; ys = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * l / 2
    zs = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * h
    c, s = math.cos(yaw), math.sin(yaw)
    P = np.stack([c * xs - s * ys + x, s * xs + c * ys + y, zs + zb], 1)
    R = Quaternion(l2e_r); P = np.array([R.rotate(q) for q in P]) + np.asarray(l2e_t)
    head = R.rotate(np.array([-math.sin(yaw), math.cos(yaw), 0.0]))  # mmdet3d lidar boxes: length along y at yaw 0
    return P, math.atan2(head[1], head[0]), max(w, l)


FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def _raster_tri(zbuf, idb, uv, d, tri, k):
    """z-buffered fill of one projected triangle; depth interpolated linearly in screen space"""
    (x0, y0), (x1, y1), (x2, y2) = uv[list(tri)]
    xa, xb = max(0, int(np.floor(min(x0, x1, x2)))), min(511, int(np.ceil(max(x0, x1, x2))))
    ya, yb = max(0, int(np.floor(min(y0, y1, y2)))), min(255, int(np.ceil(max(y0, y1, y2))))
    if xa > xb or ya > yb:
        return
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(den) < 1e-9:
        return
    X, Y = np.meshgrid(np.arange(xa, xb + 1) + 0.5, np.arange(ya, yb + 1) + 0.5)
    w0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / den
    w1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / den
    w2 = 1 - w0 - w1
    inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
    z = w0 * d[tri[0]] + w1 * d[tri[1]] + w2 * d[tri[2]]
    zb = zbuf[ya:yb + 1, xa:xb + 1]; ib = idb[ya:yb + 1, xa:xb + 1]
    win = inside & (z < zb)
    zb[win] = z[win]; ib[win] = k


def rasterize(frame, thr=0.3, drop=None):
    """frame: {"boxes", "scores", "labels", "lidar2ego_r", "lidar2ego_t"} or None (unavailable) -> (C_MAP, 256, 512).
    Depth-tested faces (PLAN-carmap v3 "far-to-near, depth-tested faces"): every face of every box is rasterised as two
    triangles into a per-pixel z-buffer, so the nearest surface owns each pixel; channels then come from that box."""
    out = np.zeros((C_MAP, 256, 512), np.float32)
    if frame is None:
        return out
    out[9] = 1.0
    zbuf = np.full((256, 512), np.inf, np.float32); idb = np.full((256, 512), -1, np.int32); items = []
    for k, (b, s, lab) in enumerate(zip(frame["boxes"], frame["scores"], frame["labels"])):
        if s < thr or lab not in GROUPS or (drop is not None and drop(k)):
            continue
        P, yaw, length = box_corners_ego(b, frame["lidar2ego_r"], frame["lidar2ego_t"], frame.get("yaw_conv", "mmdet3d"))
        if np.linalg.norm(P[:, :2].mean(0)) > 60:
            continue
        uv, d = project(P)
        if (d <= 0.5).any():
            continue
        n = len(items); items.append((GROUPS[lab], float(s), yaw, length))
        for f in FACES:
            _raster_tri(zbuf, idb, uv, d, (f[0], f[1], f[2]), n); _raster_tri(zbuf, idb, uv, d, (f[0], f[2], f[3]), n)
    if items:
        G = np.array([it[0] for it in items]); S = np.array([it[1] for it in items], np.float32)
        Y = np.array([it[2] for it in items], np.float32); Ln = np.array([it[3] for it in items], np.float32)
        m = idb >= 0; ii = idb[m]
        for g in range(4):
            out[g][m] = np.where(G[ii] == g, S[ii], 0)
        out[4][m] = 1.0 / np.maximum(zbuf[m], 1.0)
        pad = np.pad(idb, 1, constant_values=-1)
        edge = m & ((pad[:-2, 1:-1] != idb) | (pad[2:, 1:-1] != idb) | (pad[1:-1, :-2] != idb) | (pad[1:-1, 2:] != idb))
        out[5][edge] = 1.0
        out[6][m] = np.sin(Y[ii]); out[7][m] = np.cos(Y[ii]); out[8][m] = Ln[ii] / 10.0
    out[:9, OWN] = 0
    return out


def pool(mp):
    """(C, 256, 512) -> (C, 32, 64) area pooling"""
    return torch.nn.functional.avg_pool2d(torch.as_tensor(mp)[None], 8)[0]


class CarMapBranch(nn.Module):
    """Map (C_MAP x 32 x 64, or + grid channels) -> zero-initialised residuals for the four UNet encoder scales
    (320@32x64, 640@16x32, 1280@8x16, 1280@4x8), passed as down_intrablock_additional_residuals (T2I-Adapter style)."""
    def __init__(s, c_in=C_MAP, width=64):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        s.down = nn.ModuleList([nn.Identity()] + [nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU()) for _ in range(3)])
        s.proj = nn.ModuleList([nn.Conv2d(width, ch, 1) for ch in (320, 640, 1280, 1280)])
        for p_ in s.proj:
            nn.init.zeros_(p_.weight); nn.init.zeros_(p_.bias)

    def forward(s, m):
        h = s.stem(m); outs = []
        for d, p_ in zip(s.down, s.proj):
            h = d(h); outs.append(p_(h))
        return outs


def gt_frame(nusc, token, classes=None):
    """nuScenes annotations of a sample as a rasterize()-ready frame (oracle input; ego frame, identity lidar2ego)"""
    smp = nusc.get("sample", token)
    ep = nusc.get("ego_pose", nusc.get("sample_data", smp["data"]["LIDAR_TOP"])["ego_pose_token"])
    Rw, tw = Quaternion(ep["rotation"]), np.array(ep["translation"])
    names = {"vehicle.car": "car", "vehicle.truck": "truck", "vehicle.bus": "bus", "vehicle.trailer": "trailer",
             "vehicle.construction": "construction_vehicle", "human.pedestrian": "pedestrian", "vehicle.bicycle": "bicycle",
             "vehicle.motorcycle": "motorcycle", "movable_object.barrier": "barrier", "movable_object.trafficcone": "traffic_cone"}
    boxes, scores, labels = [], [], []
    for at in smp["anns"]:
        an = nusc.get("sample_annotation", at)
        lab = next((v for k, v in names.items() if an["category_name"].startswith(k)), None)
        if lab is None or an["num_lidar_pts"] < 1:
            continue
        c = Rw.inverse.rotate(np.array(an["translation"]) - tw)
        th = (Rw.inverse * Quaternion(an["rotation"])).yaw_pitch_roll[0]
        w, l, h = an["size"]
        boxes.append([c[0], c[1], c[2] - h / 2, w, l, h, th - math.pi / 2]); scores.append(1.0); labels.append(lab)
    return {"boxes": boxes, "scores": scores, "labels": labels, "lidar2ego_r": [1, 0, 0, 0], "lidar2ego_t": [0, 0, 0],
            "yaw_conv": "carmap"}


_GRID = None


def grid_channels(bev):
    """X2: BEV grid (256, 32, 48) on the ego crop x -8..41 m (columns), y +16.4..-16.4 m (rows) -> samples where each
    32x64 latent cell's ray meets z = 0 and z = 1 m: (2*256 + 2, 32, 64) = two samplings + their valid planes."""
    global _GRID
    if _GRID is None:
        vv, uu = np.meshgrid((np.arange(32) + 0.5) * 8 + CROP, (np.arange(64) + 0.5) * 8, indexing="ij")
        x_r = (uu - W / 2) / _f; y_u = -(vv - H / 2) / _f
        d = np.stack([_cp + 0 * x_r, -x_r, -_sp + 0 * x_r], -1) + y_u[..., None] * np.array([_sp, 0, _cp])
        o = np.array([-7.5, 0, 2.4]); grids, valid = [], []
        for zh in (0.0, 1.0):
            t = (zh - o[2]) / np.where(d[..., 2] < -1e-6, d[..., 2], np.nan)
            P = o + d * t[..., None]
            gx = (P[..., 0] + 8.0) / 49.0 * 2 - 1; gy = (16.4 - P[..., 1]) / 32.8 * 2 - 1
            ok = np.isfinite(t) & (t > 0) & (np.abs(gx) <= 1) & (np.abs(gy) <= 1)
            grids.append(torch.tensor(np.stack([np.nan_to_num(gx, nan=2.0), np.nan_to_num(gy, nan=2.0)], -1), dtype=torch.float32))
            valid.append(torch.tensor(ok, dtype=torch.float32))
        _GRID = (grids, valid)
    grids, valid = _GRID
    b = torch.as_tensor(bev).float()[None]
    samp = [torch.nn.functional.grid_sample(b, g[None], align_corners=False)[0] * v for g, v in zip(grids, valid)]
    return torch.cat([*samp, valid[0][None], valid[1][None]])


N_MAP = 9  # spatial planes 0-8 (no availability plane)


class MapConvIn(nn.Module):
    """Car map R (PLAN-carmap v4): frozen conv_in + bias-free zero-initialised 3x3 conv on the nine map planes (fp32),
    added at every denoising step; set .m (B, 9, 32, 64) before each UNet call (both CFG halves get the same map)."""
    def __init__(s, base):
        super().__init__(); s.base = base; s.map = nn.Conv2d(N_MAP, base.out_channels, 3, padding=1, bias=False)
        nn.init.zeros_(s.map.weight); s.m = None
    def forward(s, x):
        y = s.base(x)
        return y if s.m is None else y + s.map(s.m.float()).to(y.dtype)
