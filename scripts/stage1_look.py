"""Stage 1: load one nuScenes moment and draw what it contains (6 photos + lidar top-down with camera cones)."""
import argparse, os
import numpy as np
np.Inf = np.inf  # old matplotlib pinned by nuscenes-devkit still uses np.Inf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from mrv.nusc import CAMS, sensor_to_world, ego_to_world, apply

p = argparse.ArgumentParser()
p.add_argument("--dataroot", default="data/nuscenes")
p.add_argument("--version", default="v1.0-mini")
p.add_argument("--scene", default="scene-0061")
p.add_argument("--index", type=int, default=10)
p.add_argument("--out", default="runs/stage1")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

nusc = NuScenes(a.version, a.dataroot, verbose=False)
scene = next(s for s in nusc.scene if s["name"] == a.scene)
tok = scene["first_sample_token"]
for _ in range(a.index):
    tok = nusc.get("sample", tok)["next"]
sample = nusc.get("sample", tok)

fig, axs = plt.subplots(2, 3, figsize=(15, 5.6))
for ax, cam in zip(axs.flat, CAMS):
    sd = nusc.get("sample_data", sample["data"][cam])
    img = Image.open(f"{a.dataroot}/{sd['filename']}")
    ax.imshow(img); ax.set_title(cam, fontsize=9); ax.axis("off")
    print(cam, img.size)
plt.tight_layout(); plt.savefig(f"{a.out}/cameras.jpg", dpi=90); plt.close()

lsd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
pc = LidarPointCloud.from_file(f"{a.dataroot}/{lsd['filename']}")
pts_w = apply(sensor_to_world(nusc, lsd), pc.points[:3].T)
pts_e = apply(np.linalg.inv(ego_to_world(nusc, lsd)), pts_w)
print("lidar points", pts_e.shape)

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_facecolor("#0b1110")
ax.scatter(-pts_e[:, 1], pts_e[:, 0], s=0.3, c=np.clip(pts_e[:, 2], -2, 4), cmap="turbo")
for cam in CAMS:
    sd = nusc.get("sample_data", sample["data"][cam])
    T = np.linalg.inv(ego_to_world(nusc, lsd)) @ sensor_to_world(nusc, sd)
    o = T[:3, 3]; fwd = T[:3, 2]
    yaw = np.arctan2(fwd[1], fwd[0])
    for d in (-0.6, 0.6):
        e = o + 25 * np.array([np.cos(yaw + d), np.sin(yaw + d), 0])
        ax.plot([-o[1], -e[1]], [o[0], e[0]], color="#e9a21c", lw=0.8)
    ax.text(-(o[1] + 27 * np.sin(yaw)), o[0] + 27 * np.cos(yaw), cam[4:], color="w", fontsize=7, ha="center")
ax.add_patch(plt.Rectangle((-0.9, -2.2), 1.8, 4.4, color="w"))
ax.set_xlim(-40, 40); ax.set_ylim(-40, 40); ax.set_aspect("equal"); ax.set_title(f"{a.scene} sample {a.index}: lidar from above (forward = up)")
plt.savefig(f"{a.out}/bev.jpg", dpi=90, bbox_inches="tight"); plt.close()
print("scenes", len(nusc.scene), "samples", len(nusc.sample))
