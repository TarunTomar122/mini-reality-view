"""Astra item 1: verify the stored token rays against the guide on one pinned tick (CPU; no model).

For every token patch centre of the pinned tick: take its STORED Pluecker ray (chase frame, data/wan_tok_v1), intersect it with
the road plane z = 0 of the tick ego frame (3-30 m in front of its camera, origin from the guide's cam2tick), project the point
into the chase image with the guide's chase intrinsics, and compare the colour of the source photo at the patch centre (the
guide's own 294x518 preprocessing) with the GUIDE pixel there (guide mask set, not our own car). Correct geometry -> small
colour error (road points are where the guide put them); control = the same with each camera's colours paired with the NEXT
camera's rays (wrong geometry, same statistics). Also checks the stored rays equal a recompute from the guide's K / cam2tick.
PASS: max ray recompute error < 1e-4 AND median colour error (correct) < 0.6 x median colour error (control), >= 50 points.
Writes <out>.json and <out>.png (guide with the projected patch centres, coloured by camera).
  python scripts/wan_token_check.py --scene scene-0003 --tick 60 --out runs/wan_checks/token_geom_scene-0003_t60
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image, ImageDraw
from mrv import guide as GD
from mrv import wan_cond as WC

p = argparse.ArgumentParser()
p.add_argument("--scene", required=True); p.add_argument("--tick", type=int, required=True)
p.add_argument("--guides", default="data/guides_v1"); p.add_argument("--tokens", default="data/wan_tok_v1")
p.add_argument("--manifest", default="data/temporal_manifest.json"); p.add_argument("--dataroot", default="data/nuscenes_full")
p.add_argument("--out", required=True)
a = p.parse_args()
G = torch.load(f"{a.guides}/{a.scene}.pt", map_location="cpu", weights_only=False, mmap=True)
T = torch.load(f"{a.tokens}/{a.scene}.pt", map_location="cpu", weights_only=False, mmap=True)
i = [int(t) for t in G["ticks"].tolist()].index(a.tick)
assert int(T["ticks"][i]) == a.tick and not bool(G["hard"][i]), "pinned tick missing or hard"
CH = GD.chase_cam_to_ego(float(T["guide_version"].get("chase_dz") or 0.0))
rays = T["rays"][i].numpy().astype(np.float64)  # [6, P, 6]
rec = WC.token_rays(G["K"][i].numpy(), G["cam2tick"][i].numpy(), CH)
ray_err = float(np.abs(rec - rays).max())
H, W = WC.DEPTH_HW
row = json.load(open(a.manifest))[a.scene][i]; assert row["tick"] == a.tick
views = GD.load_views(a.dataroot, row["files"], (W, H))  # [6, H, W, 3]
grgb = G["rgb"][i].permute(1, 2, 0).numpy().astype(np.float64); gmask = G["mask"][i, 0].numpy().astype(bool) & ~GD.OWN_NP
R, t = CH[:3, :3], CH[:3, 3]
E2C = np.linalg.inv(CH)
hh, ww = WC.TOK_GRID
pu = ((np.arange(ww) + 0.5) * WC.PS).astype(int); pv = ((np.arange(hh) + 0.5) * WC.PS).astype(int)
PU, PV = np.meshgrid(pu, pv); PU, PV = PU.ravel(), PV.ravel()


def project(c_ray, c_col):
    d, m = rays[c_ray, :, :3], rays[c_ray, :, 3:]
    p0 = np.cross(d, m)
    o = E2C[:3, :3] @ G["cam2tick"][i, c_ray, :3, 3].numpy() + E2C[:3, 3]  # camera origin, chase frame
    mu = ((o - p0) * d).sum(1)
    lam = -(p0 @ R[2] + t[2]) / (d @ R[2])
    P = p0 + lam[:, None] * d
    dist = lam - mu
    ok = (dist > 3) & (dist < 30) & (P[:, 2] > 0.5)
    u = GD.K_CHASE[0, 0] * P[:, 0] / np.maximum(P[:, 2], 1e-6) + GD.K_CHASE[0, 2]
    v = GD.K_CHASE[1, 1] * P[:, 1] / np.maximum(P[:, 2], 1e-6) + GD.K_CHASE[1, 2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok &= (ui >= 0) & (ui < GD.GW) & (vi >= 0) & (vi < GD.GH)
    ok[ok] &= gmask[vi[ok], ui[ok]]
    src = views[c_col][PV, PU].astype(np.float64)
    err = np.abs(src[ok] - grgb[vi[ok], ui[ok]]).mean(1)
    return err, ui[ok], vi[ok]


good, ctrl, pts = [], [], []
for c in range(6):
    e, u_, v_ = project(c, c); good.append(e); pts.append((c, u_, v_))
    ctrl.append(project((c + 1) % 6, c)[0])
good, ctrl = np.concatenate(good), np.concatenate(ctrl)
res = {"scene": a.scene, "tick": a.tick, "ray_recompute_max_abs": ray_err, "points": int(len(good)), "control_points": int(len(ctrl)),
       "median_err": float(np.median(good)) if len(good) else None, "median_err_control": float(np.median(ctrl)) if len(ctrl) else None}
res["pass"] = bool(ray_err < 1e-4 and len(good) >= 50 and len(ctrl) >= 50 and res["median_err"] < 0.6 * res["median_err_control"])
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out + ".json", "w"), indent=1)
im = Image.fromarray(G["rgb"][i].permute(1, 2, 0).numpy()); dr = ImageDraw.Draw(im)
COL = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
for c, u_, v_ in pts:
    for x, y in zip(u_, v_):
        dr.ellipse([x - 1, y - 1, x + 1, y + 1], fill=COL[c])
im.save(a.out + ".png")
print("TOKEN_GEOM", json.dumps(res))
raise SystemExit(0 if res["pass"] else 1)
