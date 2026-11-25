"""Geometry judge, part 1: exact layout "truth" in the low-drone view for the judge set (council/PLAN-sim.md).

For each judge frame (same scenes/keyframes as sd_render.py) and each ground-plane label shift (0, 0.5, 1, 2 m along
the car's forward axis, for the validation check), write:
  - car hulls: nuScenes vehicle.car annotations within 30 m with >= 1 lidar point, 3D box corners projected into the
    drone camera (512x288, top 32 rows cropped -> 512x256), convex hull, painted far -> near so nearer cars occlude;
    kept if the visible part is >= 8 px tall and >= 50% of the hull;
  - road: nuScenes map drivable_area rasterised on the ground plane (z = 0 in the ego frame) and projected;
    road boundary points are stored in ego-frame ground metres for the kerb-error metric.
Output: data/judge_gt/<scene>/<k>_<token>_sh<shift>.npz {cars (n,256,512) bool, road (256,512) bool, edge_xy (m,2)}
"""
import argparse, glob, os
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import val as VAL
from mrv.nusc import CHASES, chase_camera, ego_to_world, apply

p = argparse.ArgumentParser()
p.add_argument("--dataroot", default="data/nuscenes_full")
p.add_argument("--tokens", default="data/enc_full")
p.add_argument("--first", type=int, default=30)
p.add_argument("--n_scenes", type=int, default=30)
p.add_argument("--frames", type=int, default=4)
p.add_argument("--shifts", default="0,0.5,1,2")
p.add_argument("--out", default="data/judge_gt")
a = p.parse_args()
nusc = NuScenes("v1.0-trainval", a.dataroot, verbose=False)
maps = {}
W, H, CROP = 512, 288, 32
R, K = chase_camera(W, H, "low")
POS = CHASES["low"][0]


def project(pts_ego):
    c = (pts_ego - POS) @ R.T
    z = c[:, 2]
    uv = (c @ K.T)[:, :2] / np.maximum(z[:, None], 1e-6)
    return uv - np.array([0, CROP]), z


def ground_of_pixels(uv):
    """pixel (u, v in the cropped 512x256 image) -> ego-frame ground point (z = 0)."""
    pix = np.stack([uv[:, 0], uv[:, 1] + CROP, np.ones(len(uv))], 1)
    d = (np.linalg.inv(K) @ pix.T).T @ R  # camera ray -> ego frame
    t = -POS[2] / d[:, 2]
    return POS[:2] + d[:, :2] * t[:, None]


names = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{a.tokens}/*.pt") if os.path.basename(f)[:-3] in set(VAL))
judge = names[a.first:a.first + a.n_scenes]
by_name = {s["name"]: s for s in nusc.scene}
import torch
for sc in judge:
    scene = by_name[sc]
    loc = nusc.get("log", scene["log_token"])["location"]
    if loc not in maps:
        maps[loc] = NuScenesMap(dataroot=a.dataroot, map_name=loc)
    nmap = maps[loc]
    toks = torch.load(f"{a.tokens}/{sc}.pt")["sample_tokens"]
    n = len(toks)
    ks = np.linspace(2, n - 3, a.frames).round().astype(int).tolist()
    os.makedirs(f"{a.out}/{sc}", exist_ok=True)
    for k in ks:
        smp = nusc.get("sample", toks[k])
        lsd = nusc.get("sample_data", smp["data"]["LIDAR_TOP"])
        e2w = ego_to_world(nusc, lsd)
        w2e = np.linalg.inv(e2w)
        # road: rasterise drivable area around the ego on a 0.2 m ground grid, then project the grid
        xs, ys = np.meshgrid(np.arange(-12, 45, 0.2), np.arange(-25, 25, 0.2))
        g_ego = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1)
        g_w = apply(e2w, g_ego)
        layer = nmap.get_records_in_radius(g_w[:, 0].mean(), g_w[:, 1].mean(), 60, ["drivable_area"])["drivable_area"]
        from shapely.geometry import Point, MultiPolygon
        polys = []
        for tok in layer:
            for pt in nmap.get("drivable_area", tok)["polygon_tokens"]:
                polys.append(nmap.extract_polygon(pt))
        mp = MultiPolygon([q for q in polys if q.is_valid]) if polys else None
        # point-in-polygon with matplotlib (shapely 1.8's vectorized module is built against numpy 1 and breaks)
        from matplotlib.path import Path as MPath
        inside = np.zeros(len(g_w), bool)
        for q in (mp.geoms if mp is not None else []):
            pin = MPath(np.asarray(q.exterior.coords)[:, :2]).contains_points(g_w[:, :2])
            for hole in q.interiors:
                pin &= ~MPath(np.asarray(hole.coords)[:, :2]).contains_points(g_w[:, :2])
            inside |= pin
        on = inside.reshape(xs.shape)
        edge = on ^ np.roll(on, 1, 0) | on ^ np.roll(on, 1, 1)
        edge_xy = np.stack([xs[edge], ys[edge]], 1)
        uv, z = project(g_ego[inside])
        road = np.zeros((256, 512), bool)
        ok = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < 512) & (uv[:, 1] >= 0) & (uv[:, 1] < 256)
        road[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = True
        from scipy.ndimage import binary_closing
        road = binary_closing(road, iterations=2)
        for shift in map(float, a.shifts.split(",")):
            hulls = []
            for at in smp["anns"]:
                ann = nusc.get("sample_annotation", at)
                if not ann["category_name"].startswith("vehicle.car") or ann["num_lidar_pts"] < 1:
                    continue
                box = nusc.get_box(at)
                corners = apply(w2e, box.corners().T)  # (8,3) ego frame
                if np.linalg.norm(corners[:, :2].mean(0)) > 30:
                    continue
                corners[:, 0] += shift  # validation: shift labels forward on the ground
                uvc, zc = project(corners)
                if (zc <= 0.5).any():
                    continue
                try:
                    hull = uvc[ConvexHull(uvc).vertices]
                except Exception:
                    continue
                hulls.append((float(zc.mean()), hull))
            hulls.sort(key=lambda h: -h[0])  # far first
            canvas = np.full((256, 512), -1, np.int32)
            full = []
            for i, (_, hull) in enumerate(hulls):
                im = Image.new("L", (512, 256), 0); ImageDraw.Draw(im).polygon([tuple(x) for x in hull], fill=1)
                m = np.asarray(im).astype(bool)
                full.append(m)
                canvas[m] = i
            cars = []
            for i, m in enumerate(full):
                vis = canvas == i
                rows = np.where(vis.any(1))[0]
                if m.sum() == 0 or vis.sum() < 0.5 * m.sum() or len(rows) == 0 or rows.max() - rows.min() + 1 < 8:
                    continue
                cars.append(vis)
            cars = np.stack(cars) if cars else np.zeros((0, 256, 512), bool)
            np.savez_compressed(f"{a.out}/{sc}/{k:03d}_{toks[k]}_sh{shift:g}.npz", cars=cars, road=road, edge_xy=edge_xy.astype(np.float32))
    print(sc, ks, flush=True)
print("GT_DONE")
