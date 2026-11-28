"""PLAN-video depth scale (SIGNATURE ROUND RESULT: "per-camera scale fitted on train drives and frozen"): per camera, the
median ratio LiDAR depth / estimated depth over LiDAR returns projected into that camera, on TRAIN keyframes only. LiDAR is
supervision only (never an input at run time). The result is frozen in a JSON that mrv/guide.GuideRenderer and
scripts/build_guides.py read (they refuse a file fitted for another depth model / resolution).

  python scripts/fit_depth_scale.py --drives train48 --out data/guides_v1/depth_scale.json

Per keyframe sample: the 6 sample camera images (resized to the depth resolution, same path as the guide), the sample's
LIDAR_TOP sweep moved into each camera through both ego poses (lidar time -> world -> camera time), nearest return per depth
pixel. Ratios with LiDAR depth in [zmin, zmax] m are kept (subsampled per image). Relative fallback model: the ratio is taken
after the per-tick camera-height anchoring, so the scale corrects what the anchoring leaves.
"""
import argparse, json, os, platform, socket, sys, time
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

from mrv import guide as G

p = argparse.ArgumentParser()
p.add_argument("--drives", default="train48", help="drive spec (mrv.guide.drive_list); must be train drives")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--dataroot", default="data/nuscenes_full"); p.add_argument("--version", default="v1.0-trainval")
p.add_argument("--samples", type=int, default=12, help="keyframes per drive, evenly spaced")
p.add_argument("--depth_model", default=G.PRIMARY); p.add_argument("--no_fallback", action="store_true")
p.add_argument("--res", default="294x518", help="depth input HxW (multiples of 14); must match build_guides.py")
p.add_argument("--zmin", type=float, default=2.0); p.add_argument("--zmax", type=float, default=60.0)
p.add_argument("--per_image", type=int, default=4000, help="max ratios kept per image")
p.add_argument("--out", default="data/guides_v1/depth_scale.json")
p.add_argument("--time_log", default="")
a = p.parse_args()
t_start = time.time()
res = tuple(int(x) for x in a.res.split("x"))
L = json.load(open(a.lists))
drives = G.drive_list(a.drives, a.lists)
bad = [d for d in drives if d not in set(L["train"])]
if bad:
    raise SystemExit(f"scale must be fitted on TRAIN drives only; not train: {bad}")
nusc = NuScenes(a.version, a.dataroot, verbose=False)
R = G.GuideRenderer(scale_file=None, res=res, model_id=a.depth_model, allow_fallback=not a.no_fallback)
rng = np.random.default_rng(0)
ratios = {c: [] for c in G.CAMS}; per_drive = {}; absrel_raw = {c: [] for c in G.CAMS}
for di, sc in enumerate(drives):
    idx = G.SceneIndex(nusc, sc)
    pick = np.unique(np.linspace(0, len(idx.samples) - 1, min(a.samples, len(idx.samples))).round().astype(int))
    dr = {c: [] for c in G.CAMS}
    for k in pick:
        smp = nusc.get("sample", idx.samples[k])
        files = [nusc.get("sample_data", smp["data"][c])["filename"] for c in G.CAMS]
        K, _, c2e, c2w = idx.cam_geom(files, np.eye(4))
        views = G.load_views(a.dataroot, files, (res[1], res[0]))
        x = torch.from_numpy(views).to(R.device).permute(0, 3, 1, 2).contiguous()
        z = R.metric_depth(R.depth.raw(x), K, c2e, scaled=False)
        lsd = nusc.get("sample_data", smp["data"]["LIDAR_TOP"])
        D = G.lidar_depth(G.lidar_points_world(nusc, a.dataroot, lsd), c2w, K, res, device=R.device)
        for v, c in enumerate(G.CAMS):
            m = torch.isfinite(D[v]) & (D[v] >= a.zmin) & (D[v] <= a.zmax) & torch.isfinite(z[v]) & (z[v] > 0.1)
            if not bool(m.any()):
                continue
            r = (D[v][m] / z[v][m]).cpu().numpy()
            if len(r) > a.per_image:
                r = r[rng.choice(len(r), a.per_image, replace=False)]
            ratios[c].append(r); dr[c].append(r)
    per_drive[sc] = {c: float(np.median(np.concatenate(dr[c]))) if dr[c] else None for c in G.CAMS}
    print(f"[{di + 1}/{len(drives)}] {sc}", {c[4:]: (round(v, 3) if v else None) for c, v in per_drive[sc].items()}, flush=True)

scales, stats = {}, {}
for c in G.CAMS:
    r = np.concatenate(ratios[c]) if ratios[c] else np.array([])
    if not len(r):
        raise SystemExit(f"no LiDAR ratios for {c}")
    s = float(np.median(r)); scales[c] = s
    pd = [v[c] for v in per_drive.values() if v[c]]
    stats[c] = {"n": int(len(r)), "median": s, "p10": float(np.percentile(r, 10)), "p25": float(np.percentile(r, 25)),
                "p75": float(np.percentile(r, 75)), "p90": float(np.percentile(r, 90)),
                # abs-rel of depth vs LiDAR: |z - D| / D = |1/ratio... | expressed through the ratio r = D / z
                "abs_rel_unscaled_median": float(np.median(np.abs(1.0 / r - 1.0))),
                "abs_rel_scaled_median": float(np.median(np.abs(s / r - 1.0))),
                "per_drive_median_spread": [float(np.min(pd)), float(np.max(pd))] if pd else None}
out = {"model_id": R.depth.info["model_id"], "kind": R.depth.kind, "depth_info": R.depth.info, "res": list(res),
       "scales": scales, "stats": stats, "drives": drives, "samples_per_drive": a.samples, "zrange": [a.zmin, a.zmax],
       "per_drive": per_drive, "rule": "median of LiDAR depth / estimated depth over train keyframe returns, per camera; frozen",
       "code_sha": G.code_sha(os.path.abspath(__file__), os.path.abspath(G.__file__)), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
tmp = a.out + f".{os.getpid()}.tmp"; json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, a.out)
print("FIT_DEPTH_SCALE", json.dumps({"model": out["model_id"], "kind": out["kind"], "scales": {c: round(v, 4) for c, v in scales.items()},
                                     "abs_rel_scaled": {c: round(stats[c]["abs_rel_scaled_median"], 3) for c in G.CAMS}}), flush=True)
if a.time_log:
    os.makedirs(os.path.dirname(a.time_log) or ".", exist_ok=True)
    with open(a.time_log, "a") as f:
        f.write(json.dumps({"stage": "fit_depth_scale", "seconds": round(time.time() - t_start, 1), "host": socket.gethostname(),
                            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}) + "\n")
