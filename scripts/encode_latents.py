"""Encode answer keys at 512x256 for the SD-1.5 renderer (latent 4 x 32 x 64; SD's UNet needs sizes divisible by 8).
Keys are 512x288 RGBA; the top 32 rows (mostly sky / far background) are cropped. Alpha = trust/coverage weight.
Writes data/<out>/<scene>.pt with lat (n,4,32,64) fp16, cover (n,32,64) fp16, sample_tokens. Tokens stay in data/enc_full.
"""
import argparse, glob, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL

p = argparse.ArgumentParser()
p.add_argument("--targets", required=True)
p.add_argument("--out", required=True)
p.add_argument("--shard", default="0/1")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
scenes = sorted(d for d in os.listdir(a.targets) if os.path.isdir(f"{a.targets}/{d}"))
si, sn = map(int, a.shard.split("/"))
for sc in scenes[si::sn]:
    dst = f"{a.out}/{sc}.pt"
    if os.path.exists(dst):
        continue
    files = sorted(f for f in glob.glob(f"{a.targets}/{sc}/*.png"))
    if not files:
        continue
    lats, covs, toks = [], [], []
    for i in range(0, len(files), 8):
        ch = files[i:i + 8]
        t = np.stack([np.asarray(Image.open(f).convert("RGBA").resize((512, 288), Image.BILINEAR))[32:] for f in ch]).astype(np.float32) / 255
        t = torch.from_numpy(t).to(dev)
        rgb = t[..., :3] * t[..., 3:] if "lidar" in a.targets else t[..., :3]  # splat keys keep their colour; alpha only weights
        with torch.no_grad():
            z = vae.encode(rgb.permute(0, 3, 1, 2) * 2 - 1).latent_dist.mean * 0.18215
        lats.append(z.half().cpu())
        covs.append(F.avg_pool2d(t[..., 3][:, None], 8)[:, 0].half().cpu())
        toks += [os.path.basename(f)[4:-4] for f in ch]
    torch.save({"lat": torch.cat(lats), "cover": torch.cat(covs), "sample_tokens": toks}, dst)
    print(sc, len(files), flush=True)
print("ENCODE512_DONE")
