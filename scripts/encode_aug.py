"""Phone-photo robustness: augmented copies of every frame's camera tokens, with matching viewing rays.

For each keyframe and each of the 6 cameras, independently:
  - zoom/crop: keep a random window of 1/z of the picture, z in [1, 1.3] (narrower field of view, like a phone),
    placed at a random offset (shifts where the camera seems to point: a small yaw/pitch jitter);
  - colour: brightness, contrast, saturation x U(0.75, 1.25), hue shift +-0.03, gamma U(0.8, 1.25);
  - camera quality: JPEG quality 35-95, Gaussian blur sigma 0-1.2 (half the time).
The crop is resized to 448x252 (16:9) for DINOv2. Rays (Plücker, ego frame) are recomputed for the 16x9 pooled
patches *inside the crop*, so geometry stays consistent with what the model sees.
Writes data/<out>_<copy>/<scene>.pt with tokens (n,6,144,384) fp16, rays (n,6,144,6), sample_tokens.
"""
import argparse, io, os, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from transformers import AutoModel
from mrv.nusc import CAMS

p = argparse.ArgumentParser()
p.add_argument("--dataroot", default="data/nuscenes_full")
p.add_argument("--version", default="v1.0-trainval")
p.add_argument("--like", default="data/enc_full", help="take scene list + sample order from here")
p.add_argument("--out", default="data/enc_aug")
p.add_argument("--copy", type=int, default=0)
p.add_argument("--shard", default="0/1")
a = p.parse_args()
dev = "cuda"
rng = np.random.default_rng(1000 + a.copy * 7919 + int(a.shard.split("/")[0]))
nusc = NuScenes(a.version, a.dataroot, verbose=False)
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).half().eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
od = f"{a.out}_{a.copy}"
os.makedirs(od, exist_ok=True)
u0 = (np.arange(16) + 0.5) / 16
v0 = (np.arange(9) + 0.5) / 9
UU, VV = np.meshgrid(u0, v0)


from mrv.aug import augment as _augment, rays_for


def augment(img):
    return _augment(img, rng)


scenes = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{a.like}/*.pt"))
si, sn = map(int, a.shard.split("/"))
for sc in scenes[si::sn]:
    dst = f"{od}/{sc}.pt"
    if os.path.exists(dst):
        continue
    stoks = torch.load(f"{a.like}/{sc}.pt")["sample_tokens"]
    toks, rays = [], []
    for i in range(0, len(stoks), 8):
        imgs, rr = [], []
        for st in stoks[i:i + 8]:
            s = nusc.get("sample", st)
            for cam in CAMS:
                sd = nusc.get("sample_data", s["data"][cam])
                im, box = augment(Image.open(f"{a.dataroot}/{sd['filename']}").convert("RGB"))
                imgs.append(im)
                rr.append(rays_for(nusc.get("calibrated_sensor", sd["calibrated_sensor_token"]), box))
        with torch.no_grad():
            x = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2).float() / 255
            h = dino(pixel_values=((x - MEAN) / STD).half()).last_hidden_state[:, 1:]
            h = F.avg_pool2d(h.view(-1, 18, 32, 384).permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2)
        n = len(stoks[i:i + 8])
        toks.append(h.reshape(n, 6, 144, 384).cpu())
        rays.append(torch.tensor(np.stack(rr), dtype=torch.float32).reshape(n, 6, 144, 6))
    torch.save({"tokens": torch.cat(toks), "rays": torch.cat(rays), "sample_tokens": stoks}, dst)
    print(sc, len(stoks), flush=True)
print("AUG_DONE")
