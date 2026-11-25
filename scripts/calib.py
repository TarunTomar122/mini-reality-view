"""Experiment A helper: per-scene camera calibration -> per-patch viewing rays in the ego frame.

For each scene: the 6 cameras' intrinsics K (scaled to the 448x252 DINOv2 input) and cam->ego transforms
(calibrated_sensor is fixed per log). For every pooled 16x9 patch we store the ray as Plücker coordinates
(d, o x d) with d the unit direction through the patch centre and o the camera origin, both in the ego frame.
Output: data/calib_rays.pt = {scene_name: tensor (6, 144, 6) float32}.
"""
import argparse
import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from mrv.nusc import CAMS

p = argparse.ArgumentParser()
p.add_argument("--dataroot", default="data/nuscenes_full")
p.add_argument("--version", default="v1.0-trainval")
p.add_argument("--out", default="data/calib_rays.pt")
p.add_argument("--extra", default="data/nuscenes:v1.0-mini", help="also add these root:version pairs")
a = p.parse_args()

# patch centres of the pooled 16 x 9 grid on the 448 x 252 image (each pooled patch = 28 x 28 px)
u = (np.arange(16) + 0.5) * 28
v = (np.arange(9) + 0.5) * 28
uu, vv = np.meshgrid(u, v)  # (9,16) row-major like the tokens
pix = np.stack([uu.ravel(), vv.ravel(), np.ones(144)], 1)  # (144,3)
out = {}
for root, ver in [(a.dataroot, a.version)] + [tuple(x.split(":")) for x in a.extra.split(",") if x]:
    nusc = NuScenes(ver, root, verbose=False)
    for sc in nusc.scene:
        s = nusc.get("sample", sc["first_sample_token"])
        rays = []
        for cam in CAMS:
            cs = nusc.get("calibrated_sensor", nusc.get("sample_data", s["data"][cam])["calibrated_sensor_token"])
            K = np.array(cs["camera_intrinsic"], dtype=np.float64)
            K[0] *= 448 / 1600; K[1] *= 252 / 900
            R = Quaternion(cs["rotation"]).rotation_matrix
            o = np.array(cs["translation"])
            d = (np.linalg.inv(K) @ pix.T).T @ R.T
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            m = np.cross(np.broadcast_to(o, d.shape), d)
            rays.append(np.concatenate([d, m], 1))
        out[sc["name"]] = torch.tensor(np.stack(rays), dtype=torch.float32)
    print(ver, len(nusc.scene), flush=True)
torch.save(out, a.out)
print("CALIB_DONE", len(out))
