"""Placement diagnostic (owner review item 4): do camera-only 3D boxes place cars better than our renderer?

  python scripts/bev_box_judge.py --boxes tmp/bev_boxes_decision.json --like runs/judge/tpvE3_10000 \
      --cache runs/judge/keycache_m2f --out runs/judge/bevbox_diag.json

For every judge frame (scene, k, token) of a renders folder, three sets of car masks are compared with the cached
car masks of the exact game key (same judge, own car excluded, Hungarian IoU >= 0.3, pooled F1):
  gt   : nuScenes vehicle.car boxes (>= 1 lidar point) projected into the third-person camera (checks the projection),
  bev  : BEVFormer-base car boxes (score >= --thr) from the same cameras, projected the same way,
  ours : the renderer's own drawings (from the judge_pair score file of the same renders, for reference).
Boxes are projected as convex hulls of their 8 corners, painted far -> near so nearer boxes hide farther ones, kept if
the visible part is >= 8 px tall. Third-person camera: 7.5 m behind the ego origin, 2.4 m up, pitch -8 deg, 60 deg
HFOV, 512x288 with the top 32 rows cropped (sim_render.py --view tpv). Writes per-frame counts and pooled F1s.
"""
import argparse, glob, json, math, os
import numpy as np
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

p = argparse.ArgumentParser()
p.add_argument("--boxes", required=True)
p.add_argument("--like", required=True)
p.add_argument("--cache", default="runs/judge/keycache_m2f")
p.add_argument("--ours", default="", help="judge_pair JSON of the same renders (default: <like>_m2f.json)")
p.add_argument("--thr", type=float, default=0.3)
p.add_argument("--out", required=True)
p.add_argument("--overlays", type=int, default=0, help="save this many overlay pictures (key + GT hulls yellow + BEVFormer hulls cyan)")
a = p.parse_args()
from nuscenes.nuscenes import NuScenes
nusc = NuScenes("v1.0-trainval", "data/nuscenes_full", verbose=False)
W, H, CROP = 512, 288, 32
f = W / 2 / math.tan(math.radians(30)); cp, sp = math.cos(math.radians(8)), math.sin(math.radians(8))
OWN = None


def to_img(P):  # ego-frame points (x fwd, y left, z up) -> pixels in the cropped 512x256 image, depth
    Yf, R_, Zu = P[:, 0] + 7.5, -P[:, 1], P[:, 2] - 2.4
    depth = Yf * cp - Zu * sp; up = Yf * sp + Zu * cp
    return np.stack([W / 2 + f * R_ / np.maximum(depth, 1e-3), H / 2 - f * up / np.maximum(depth, 1e-3) - CROP], 1), depth


def corners(c, size, yaw):  # centre (bottom-centre z handled by caller), size (w along x, l along y, h), yaw about z
    w, l, h = size
    xs, ys, zs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * w / 2, np.array([1, -1, -1, 1, 1, -1, -1, 1]) * l / 2, np.array([0, 0, 0, 0, 1, 1, 1, 1]) * h
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    xy = (R @ np.stack([xs, ys])).T + c[:2]
    return np.concatenate([xy, (zs + c[2])[:, None]], 1)


def paint(corner_sets):
    hulls = []
    for C in corner_sets:
        uv, d = to_img(C)
        if (d <= 0.5).any() or np.linalg.norm(C[:, :2].mean(0)) > 60:
            continue
        try:
            hulls.append((float(d.mean()), uv[ConvexHull(uv).vertices]))
        except Exception:
            pass
    hulls.sort(key=lambda h: -h[0])
    canvas = np.full((256, 512), -1, np.int32); full = []
    for i, (_, hu) in enumerate(hulls):
        im = Image.new("L", (512, 256), 0); ImageDraw.Draw(im).polygon([tuple(x) for x in hu], fill=1)
        m = np.asarray(im).astype(bool); full.append(m); canvas[m] = i
    out = []
    for i, m in enumerate(full):
        vis = canvas == i
        rows = np.where(vis.any(1))[0]
        if len(rows) and rows.max() - rows.min() + 1 >= 8 and vis.sum() >= 0.3 * max(1, m.sum()):
            out.append(vis)
    return out


def match(P, G):
    if not P or not G:
        return 0
    A = np.stack(P).reshape(len(P), -1).astype(np.float32); B = np.stack(G).reshape(len(G), -1).astype(np.float32)
    inter = A @ B.T; iou = inter / np.maximum(A.sum(1)[:, None] + B.sum(1)[None] - inter, 1)
    r, c = linear_sum_assignment(-iou); return int((iou[r, c] >= 0.3).sum())


BX = json.load(open(a.boxes))
idx = [it for it in json.load(open(f"{a.like}/index.json"))["items"] if it["seed"] == 0]
tot = {k: [0, 0, 0] for k in ("gt", "bev")}; frames = []
for it in idx:
    cf = f"{a.cache}/{it['scene']}_{it['k']:03d}_{it['token']}.npz"
    if not os.path.exists(cf) or it["token"] not in BX:
        continue
    z = np.load(cf)
    keycars = [m.astype(bool) for m in np.unpackbits(z["cars"], axis=-1)[..., :512]] if len(z["cars"]) else []
    smp = nusc.get("sample", it["token"])
    ep = nusc.get("ego_pose", nusc.get("sample_data", smp["data"]["LIDAR_TOP"])["ego_pose_token"])
    Rw, tw = Quaternion(ep["rotation"]), np.array(ep["translation"])
    gts = []
    for at in smp["anns"]:
        an = nusc.get("sample_annotation", at)
        if not an["category_name"].startswith("vehicle.car") or an["num_lidar_pts"] < 1:
            continue
        c = Rw.inverse.rotate(np.array(an["translation"]) - tw)
        yaw = (Rw.inverse * Quaternion(an["rotation"])).yaw_pitch_roll[0]
        wlh = an["size"]  # nuScenes: width, length, height; length along the heading
        C = corners(np.array([c[0], c[1], c[2] - wlh[2] / 2]), (wlh[1], wlh[0], wlh[2]), yaw)
        gts.append(C)
    b = BX[it["token"]]; Rl, tl = Quaternion(b["lidar2ego_r"]), np.array(b["lidar2ego_t"])
    bevs = []
    for bb, s, lab in zip(b["boxes"], b["scores"], b["labels"]):
        if lab != "car" or s < a.thr:
            continue
        x, y, zb, w_, l_, h_, yaw = bb[:7]
        C = corners(np.array([x, y, zb]), (w_, l_, h_), yaw)  # lidar frame
        C = np.array([Rl.rotate(q) for q in C]) + tl
        bevs.append(C)
    rec = {"scene": it["scene"], "k": it["k"], "key_cars": len(keycars)}
    if a.overlays and len(frames) % 7 == 0 and len(glob.glob(f"{os.path.dirname(a.out)}/bevbox_ov_*.jpg")) < a.overlays:
        kp = f"data/sim/keys_tpv/{it['scene']}/{it['k']:03d}_{it['token']}.png"
        im = np.asarray(Image.open(kp).convert("RGB").resize((512, 288)))[32:].copy()
        for sets, col in ((gts, (255, 230, 0)), (bevs, (0, 220, 255))):
            for m in paint(sets):
                e = m & ~np.pad(m, 1)[2:, 1:-1] | m & ~np.pad(m, 1)[:-2, 1:-1] | m & ~np.pad(m, 1)[1:-1, 2:] | m & ~np.pad(m, 1)[1:-1, :-2]
                im[e] = col
        Image.fromarray(im).save(f"{os.path.dirname(a.out)}/bevbox_ov_{it['scene']}_{it['k']:03d}.jpg", quality=90)
    for name, sets in (("gt", gts), ("bev", bevs)):
        m = paint(sets); tp = match(m, keycars)
        tot[name][0] += tp; tot[name][1] += len(m) - tp; tot[name][2] += len(keycars) - tp
        rec[name] = [tp, len(m) - tp, len(keycars) - tp]
    frames.append(rec)
f1 = lambda t: 2 * t[0] / max(1, 2 * t[0] + t[1] + t[2])
ours = json.load(open(a.ours or f"{a.like}_m2f.json"))["summary"]
summ = {"frames": len(frames), "gt_f1": f1(tot["gt"]), "bev_f1": f1(tot["bev"]), "ours_f1": ours["f1"], "tot": tot, "thr": a.thr}
json.dump({"summary": summ, "frames": frames}, open(a.out, "w"), indent=1)
print("BEVBOX", json.dumps(summ))
