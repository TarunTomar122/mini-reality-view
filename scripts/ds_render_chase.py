"""Stage 6: render a trained drivestudio splat scene from our chase ("drone") camera at every keyframe.
Runs inside the drivestudio env, from the drivestudio folder (PYTHONPATH=.).
Writes <out>/<scene-name>/<k:03d>_<sample_token>.png (RGBA, alpha = splat opacity) to match make_targets.py.
The chase camera matches mrv/nusc.py: 13 m behind and 8.5 m above the ego origin, looking at a point 12 m ahead on the ground, 60 deg horizontal FOV.
"""
import argparse, os
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from datasets.driving_dataset import DrivingDataset
from datasets.base.pixel_source import get_rays
from utils.misc import import_str

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--scene_idx", type=int, required=True)
p.add_argument("--nusc_root", default="data/nuscenes/raw")
p.add_argument("--version", default="v1.0-mini")
p.add_argument("--out", required=True)
p.add_argument("--skip_done", action="store_true")
p.add_argument("--W", type=int, default=512)
p.add_argument("--H", type=int, default=288)
p.add_argument("--stride", type=int, default=5, help="10 Hz frames per keyframe (interpolate_N + 1)")
p.add_argument("--chase", default="high", choices=["high", "low"])
a = p.parse_args()
dev = torch.device("cuda")

cfg = OmegaConf.load(os.path.join(os.path.dirname(a.ckpt), "config.yaml"))
dataset = DrivingDataset(data_cfg=cfg.data)
trainer = import_str(cfg.trainer.type)(
    **cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model,
    num_train_images=len(dataset.train_image_set), num_full_images=len(dataset.full_image_set),
    test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2, 3), device=dev)
trainer.resume_from_checkpoint(ckpt_path=a.ckpt, load_only_model=True)
trainer.set_eval()

nusc = NuScenes(a.version, a.nusc_root, verbose=False)
scene = nusc.scene[a.scene_idx]
toks, t = [], scene["first_sample_token"]
while t:
    toks.append(t); t = nusc.get("sample", t)["next"]
s0 = nusc.get("sample", toks[0])
cs = nusc.get("calibrated_sensor", nusc.get("sample_data", s0["data"]["CAM_FRONT"])["calibrated_sensor_token"])
cam_to_ego = np.eye(4)
cam_to_ego[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
cam_to_ego[:3, 3] = cs["translation"]
ego_from_cam = torch.tensor(np.linalg.inv(cam_to_ego), dtype=torch.float32)

# chase camera in the ego frame (x fwd, y left, z up), OpenCV axes: x right, y down, z forward
C, L = {"high": (np.array([-13.0, 0, 8.5]), np.array([12.0, 0, 0])), "low": (np.array([-9.0, 0, 4.5]), np.array([15.0, 0, 0.5]))}[a.chase]
f = (L - C) / np.linalg.norm(L - C)
r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
u = np.cross(r, f)
chase_ego = np.eye(4)
chase_ego[:3, 0], chase_ego[:3, 1], chase_ego[:3, 2], chase_ego[:3, 3] = r, -u, f, C
chase_ego = torch.tensor(chase_ego, dtype=torch.float32)
fx = (a.W / 2) / np.tan(np.radians(30))
K = torch.tensor([[fx, 0, a.W / 2], [0, fx, a.H / 2], [0, 0, 1]], dtype=torch.float32)

front = dataset.pixel_source.camera_data[0]  # camera index 0 = CAM_FRONT
od = f"{a.out}/{scene['name']}"
os.makedirs(od, exist_ok=True)
y, x = torch.meshgrid(torch.arange(a.H), torch.arange(a.W), indexing="ij")
x, y = x.flatten().float(), y.flatten().float()
for k, tok in enumerate(toks):
    fr = min(k * a.stride, dataset.num_img_timesteps - 1)
    img_inf, cam_inf = front.get_image(fr)
    c2w = cam_inf["camera_to_world"].float().cpu()
    ego_to_world = c2w @ ego_from_cam
    c2w_new = ego_to_world @ chase_ego
    o, d, n = get_rays(x, y, c2w_new, K)
    ii = {
        "origins": o.reshape(a.H, a.W, 3), "viewdirs": d.reshape(a.H, a.W, 3), "direction_norm": n.reshape(a.H, a.W, 1),
        "pixel_coords": torch.stack([y / a.H, x / a.W], -1).reshape(a.H, a.W, 2),
        "normed_time": torch.full((a.H, a.W), float(img_inf["normed_time"].flatten()[0])),
        "img_idx": torch.full((a.H, a.W), int(img_inf["img_idx"].flatten()[0]), dtype=torch.long),
        "frame_idx": torch.full((a.H, a.W), fr, dtype=torch.long),
    }
    ci = {"cam_id": torch.zeros((a.H, a.W), dtype=torch.long), "cam_name": cam_inf["cam_name"], "camera_to_world": c2w_new,
          "height": torch.tensor(a.H), "width": torch.tensor(a.W), "intrinsics": K}
    ii = {k_: v.to(dev) for k_, v in ii.items()}
    ci = {k_: (v.to(dev) if torch.is_tensor(v) else v) for k_, v in ci.items()}
    with torch.no_grad():
        out = trainer(ii, ci, novel_view=True)
    rgb = (out["rgb"].clamp(0, 1) * 255).byte().cpu().numpy()
    alpha = (out["opacity"].clamp(0, 1) * 255).byte().cpu().numpy().reshape(a.H, a.W, 1)
    Image.fromarray(np.concatenate([rgb, alpha], -1), "RGBA").save(f"{od}/{k:03d}_{tok}.png")
    if "depth" in out:  # metres along the ray, for trust masks
        np.savez_compressed(f"{od}/{k:03d}_{tok}_depth.npz", depth=out["depth"].float().cpu().numpy().reshape(a.H, a.W).astype(np.float16),
                            c2w=c2w_new.numpy(), K=K.numpy(), ego_to_world=ego_to_world.numpy(), frame=fr)
print(scene["name"], len(toks), "RENDER_DONE")
