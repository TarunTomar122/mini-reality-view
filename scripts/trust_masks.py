"""Step 1: trust masks for splat answer keys.

A splat render is only trustworthy where a real camera actually saw that surface. For every drone pixel:
  1. lift it to 3D with the splat's rendered depth (saved by ds_render_chase.py, with the drone camera and ego pose);
  2. move the point into the ego frame of that keyframe and project it into the 6 real cameras (nuScenes calibration);
  3. a camera "sees" it if it lands inside the image, in front, not hidden behind a nearer surface (judged from that
     keyframe's lidar projected into the same camera), and not too far away;
  4. weight by how similar the real camera's viewing direction is to the drone's (splats are view-dependent).
trust = splat opacity x best camera's weight. Written as the key's alpha (RGB unchanged), so encode.py /
encode_latents.py weight the loss by it.
"""
import argparse, glob, os
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from mrv.nusc import CAMS, sensor_to_world, ego_to_world, apply

p = argparse.ArgumentParser()
p.add_argument("--splat_dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--dataroot", default="data/nuscenes")
p.add_argument("--version", default="v1.0-mini")
p.add_argument("--max_dist", type=float, default=60.0)
p.add_argument("--shard", default="0/1")
a = p.parse_args()
nusc = NuScenes(a.version, a.dataroot, verbose=False)
scenes = sorted(d for d in os.listdir(a.splat_dir) if os.path.isdir(f"{a.splat_dir}/{d}"))
si, sn = map(int, a.shard.split("/"))
LD = 4  # lidar depth maps at 1/4 of 1600x900


def lidar_depth_maps(sample):
    """Per camera: min lidar depth on a 400x225 grid, spread 5x5 so thin gaps don't count as 'see-through'."""
    lsd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pw = apply(sensor_to_world(nusc, lsd), LidarPointCloud.from_file(f"{a.dataroot}/{lsd['filename']}").points[:3].T)
    maps = {}
    for cam in CAMS:
        sd = nusc.get("sample_data", sample["data"][cam])
        K = np.array(nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])["camera_intrinsic"])
        c = apply(np.linalg.inv(sensor_to_world(nusc, sd)), pw)
        c = c[c[:, 2] > 0.5]
        uv = (c @ K.T)[:, :2] / c[:, 2:3] / LD
        x, y = uv[:, 0].astype(int), uv[:, 1].astype(int)
        ok = (x >= 0) & (x < 1600 // LD) & (y >= 0) & (y < 900 // LD)
        m = np.full((900 // LD, 1600 // LD), np.inf)
        np.minimum.at(m, (y[ok], x[ok]), c[ok, 2])
        maps[cam] = (minimum_filter(m, size=5), K, sd)
    return maps


for sc in scenes[si::sn]:
    od = f"{a.out}/{sc}"
    os.makedirs(od, exist_ok=True)
    stats = []
    for f in sorted(glob.glob(f"{a.splat_dir}/{sc}/*_depth.npz")):
        key = f.replace("_depth.npz", ".png")
        tok = os.path.basename(key)[4:-4]
        z = np.load(f)
        D, c2w, K, e2w = z["depth"].astype(np.float32), z["c2w"], z["K"], z["ego_to_world"]
        rgba = np.asarray(Image.open(key)).copy()
        H, W = D.shape
        v, u = np.mgrid[0:H, 0:W]
        cam_pts = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * D, (v + 0.5 - K[1, 2]) / K[1, 1] * D, D], -1).reshape(-1, 3)
        p_ds = cam_pts @ c2w[:3, :3].T + c2w[:3, 3]
        p_ego = apply(np.linalg.inv(e2w), p_ds)
        drone_ego = apply(np.linalg.inv(e2w), c2w[None, :3, 3])[0]
        sample = nusc.get("sample", tok)
        maps = lidar_depth_maps(sample)
        lsd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego2w = ego_to_world(nusc, lsd)
        p_w = apply(ego2w, p_ego)
        drone_w = apply(ego2w, drone_ego[None])[0]
        best = np.zeros(len(p_w))
        for cam, (dm, Kc, sd) in maps.items():
            c = apply(np.linalg.inv(sensor_to_world(nusc, sd)), p_w)
            zc = c[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                uv = (c @ Kc.T)[:, :2] / zc[:, None]
            inside = (zc > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < 1600) & (uv[:, 1] >= 0) & (uv[:, 1] < 900) & (zc < a.max_dist)
            x = np.clip((uv[:, 0] / LD).astype(int), 0, 1600 // LD - 1)
            y = np.clip((uv[:, 1] / LD).astype(int), 0, 900 // LD - 1)
            lz = dm[y, x]
            visible = inside & ((zc <= lz * 1.05 + 1.0) | ~np.isfinite(lz))
            cam_pos = sensor_to_world(nusc, sd)[:3, 3]
            d_real = p_w - cam_pos
            d_drone = p_w - drone_w
            cos = (d_real * d_drone).sum(1) / (np.linalg.norm(d_real, axis=1) * np.linalg.norm(d_drone, axis=1) + 1e-6)
            w = visible * np.clip(cos, 0, 1) * np.exp(-zc.clip(0) / a.max_dist)
            best = np.maximum(best, np.nan_to_num(w))
        trust = (rgba[..., 3] / 255.0) * np.sqrt(best.reshape(H, W))
        trust[~np.isfinite(D) | (D <= 0)] = 0
        rgba[..., 3] = (trust * 255).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(f"{od}/{os.path.basename(key)}")
        stats.append(trust.mean())
    print(sc, len(stats), "mean trust", round(float(np.mean(stats)) if stats else 0, 3), flush=True)
print("TRUST_DONE")
