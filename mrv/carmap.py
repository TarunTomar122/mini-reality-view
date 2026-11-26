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


def box_corners_ego(box, l2e_r, l2e_t):
    """BEVFormer box (lidar frame: x, y, z_bottom, w, l, h, yaw, ...) -> 8 corners in the ego frame, yaw in ego"""
    x, y, zb, w, l, h, yaw = box[:7]
    xs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * w / 2; ys = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * l / 2
    zs = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * h
    c, s = math.cos(yaw), math.sin(yaw)
    P = np.stack([c * xs - s * ys + x, s * xs + c * ys + y, zs + zb], 1)
    R = Quaternion(l2e_r); P = np.array([R.rotate(q) for q in P]) + np.asarray(l2e_t)
    head = R.rotate(np.array([-math.sin(yaw), math.cos(yaw), 0.0]))  # mmdet3d lidar boxes: length along y at yaw 0
    return P, math.atan2(head[1], head[0]), max(w, l)


def rasterize(frame, thr=0.3, drop=None):
    """frame: {"boxes", "scores", "labels", "lidar2ego_r", "lidar2ego_t"} or None (unavailable) -> (C_MAP, 256, 512)"""
    out = np.zeros((C_MAP, 256, 512), np.float32)
    if frame is None:
        return out
    out[9] = 1.0
    items = []
    for k, (b, s, lab) in enumerate(zip(frame["boxes"], frame["scores"], frame["labels"])):
        if s < thr or lab not in GROUPS or (drop is not None and drop(k)):
            continue
        P, yaw, length = box_corners_ego(b, frame["lidar2ego_r"], frame["lidar2ego_t"])
        if np.linalg.norm(P[:, :2].mean(0)) > 60:
            continue
        uv, d = project(P)
        if (d <= 0.5).any():
            continue
        try:
            hull = uv[ConvexHull(uv).vertices]
        except Exception:
            continue
        items.append((float(d.mean()), hull, GROUPS[lab], float(s), yaw, length))
    items.sort(key=lambda t: -t[0])  # far first; nearer boxes overwrite
    for dist, hull, g, s, yaw, length in items:
        im = Image.new("L", (512, 256), 0); ImageDraw.Draw(im).polygon([tuple(q) for q in hull], fill=255, outline=128)
        a = np.asarray(im); m = a > 0; edge = a == 128
        out[0:4, m] = 0; out[g, m] = s; out[4, m] = 1.0 / max(dist, 1.0)
        out[5, m] = 0; out[5, edge] = 1.0
        out[6, m] = math.sin(yaw); out[7, m] = math.cos(yaw); out[8, m] = length / 10.0
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
