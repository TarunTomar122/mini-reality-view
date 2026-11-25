"""Stage 2: build drone-view (chase camera) targets from camera-coloured lidar.

For each keyframe, lidar from neighbouring keyframes (±K, ~0.5 s apart) is moved into the world frame and coloured
by its own keyframe's cameras. Points on moving objects are kept only from the current keyframe, so cars don't smear.
Everything is then drawn from a virtual camera 13 m behind and 8.5 m above the car.
Output: <out>/<scene>/<idx>_<sample_token>.png, RGBA, alpha = pixel covered by a point.
"""
import argparse, os, time
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from mrv.nusc import sensor_to_world, ego_to_world, apply, colour_points, dynamic_mask, project_chase, splat

p = argparse.ArgumentParser()
p.add_argument("--dataroot", default="data/nuscenes")
p.add_argument("--version", default="v1.0-mini")
p.add_argument("--out", default="data/targets_lidar")
p.add_argument("--W", type=int, default=512)
p.add_argument("--H", type=int, default=288)
p.add_argument("--K", type=int, default=100, help="neighbouring keyframes for static points (100 = whole scene)")
p.add_argument("--radius", type=float, default=0.12, help="point size in metres")
p.add_argument("--scenes", default="", help="comma list of scene names, default all")
p.add_argument("--chase", default="high", choices=["high", "low"])
p.add_argument("--shard", default="0/1", help="i/n: process every n-th scene starting at i")
a = p.parse_args()

nusc = NuScenes(a.version, a.dataroot, verbose=False)
scenes = nusc.scene
if a.scenes:
    want = set(a.scenes.split(","))
    scenes = [s for s in scenes if s["name"] in want]
si, sn = map(int, a.shard.split("/"))
scenes = scenes[si::sn]


def keyframe_points(sample):
    """World-frame points of one keyframe, coloured by its cameras, plus a 'moving object' mask."""
    lsd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(f"{a.dataroot}/{lsd['filename']}").points[:3].T
    pc = pc[np.linalg.norm(pc[:, :2], axis=1) > 2.5]  # drop hits on our own car
    pw = apply(sensor_to_world(nusc, lsd), pc)
    col, seen = colour_points(nusc, sample, pw, a.dataroot)
    pw, col = pw[seen], col[seen]
    return pw, col, dynamic_mask(nusc, sample, pw)


for scene in scenes:
    t0 = time.time()
    toks, tok = [], scene["first_sample_token"]
    while tok:
        toks.append(tok)
        tok = nusc.get("sample", tok)["next"]
    samples = [nusc.get("sample", t) for t in toks]
    od = f"{a.out}/{scene['name']}"
    if os.path.isdir(od) and len(os.listdir(od)) == len(samples):
        continue
    need = [nusc.get("sample_data", s["data"][c])["filename"] for s in samples for c in s["data"] if not c.startswith("RADAR")]
    if not all(os.path.exists(f"{a.dataroot}/{f}") for f in need):
        print(f"{scene['name']}: files not downloaded yet, skipping", flush=True)
        continue
    kp = [keyframe_points(s) for s in samples]
    cov = []
    os.makedirs(od, exist_ok=True)
    for i, s in enumerate(samples):
        lsd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        T = np.linalg.inv(ego_to_world(nusc, lsd))
        P, C = [], []
        for j in range(max(0, i - a.K), min(len(samples), i + a.K + 1)):
            pw, col, dyn = kp[j]
            keep = np.ones(len(pw), bool) if j == i else ~dyn
            P.append(pw[keep]); C.append(col[keep])
        pe = apply(T, np.concatenate(P))
        u, v, z, fx = project_chase(pe, a.W, a.H, a.chase)
        img = splat(u, v, z, np.concatenate(C), a.W, a.H, fx, radius_m=a.radius)
        cov.append((img[..., 3] > 0).mean())
        Image.fromarray(img, "RGBA").save(f"{od}/{i:03d}_{s['token']}.png")
    print(f"{scene['name']}: {len(samples)} targets in {time.time() - t0:.0f}s, coverage {np.mean(cov):.3f}", flush=True)
print("TARGETS_DONE")
