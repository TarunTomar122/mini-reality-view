"""Stage 3: shrink everything into numbers.

Per keyframe:
  6 photos  -> resize 448x252 -> DINOv2-small (frozen) -> 6 x 576 tokens x 384   (32x18 patches of 14 px)
  target    -> 256x144 RGB (holes black) -> SD VAE encoder -> 4 x 18 x 32 latent (x 0.18215)
  coverage  -> fraction of each 8x8 block the lidar actually covered -> 18 x 32 (used to weight the loss)
Saved per scene as data/enc/<scene>.pt
"""
import argparse, glob, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from nuscenes.nuscenes import NuScenes
from transformers import AutoModel
from diffusers import AutoencoderKL
from mrv.nusc import CAMS

p = argparse.ArgumentParser()
p.add_argument("--dataroot", default="data/nuscenes")
p.add_argument("--version", default="v1.0-mini")
p.add_argument("--targets", default="data/targets_lidar")
p.add_argument("--out", default="data/enc")
p.add_argument("--shard", default="0/1")
p.add_argument("--tokens_from", default="", help="copy camera tokens from this enc dir instead of re-running DINOv2")
p.add_argument("--pool", type=int, default=1, help="average-pool the 32x18 token grid before saving (2 -> 16x9)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
nusc = NuScenes(a.version, a.dataroot, verbose=False)
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).half().eval()
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

scenes = sorted(os.listdir(a.targets))
si, sn = map(int, a.shard.split("/"))
for scene in scenes[si::sn]:
    dst = f"{a.out}/{scene}.pt"
    if os.path.exists(dst):
        continue
    files = sorted(glob.glob(f"{a.targets}/{scene}/*.png"))
    toks, lats, covs, stoks, times = [], [], [], [], []
    for i in range(0, len(files), 8):
        chunk = files[i:i + 8]
        imgs, tg = [], []
        for f in chunk:
            st = os.path.basename(f)[4:-4]
            s = nusc.get("sample", st)
            stoks.append(st); times.append(s["timestamp"])
            for cam in ([] if a.tokens_from else CAMS):
                sd = nusc.get("sample_data", s["data"][cam])
                im = Image.open(f"{a.dataroot}/{sd['filename']}").convert("RGB").resize((448, 252), Image.BICUBIC)
                imgs.append(np.asarray(im))
            t = np.asarray(Image.open(f).resize((256, 144), Image.BILINEAR)).astype(np.float32) / 255
            tg.append(t)
        with torch.no_grad():
            h = None
            if not a.tokens_from:
                x = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2).float() / 255
                x = ((x - MEAN) / STD).half()
                h = dino(pixel_values=x).last_hidden_state[:, 1:]  # (B*6, 576, 384)
            if h is None:
                pass
            elif a.pool > 1:
                h = F.avg_pool2d(h.view(-1, 18, 32, 384).permute(0, 3, 1, 2), a.pool).flatten(2).transpose(1, 2)
            if h is not None:
                toks.append(h.reshape(len(chunk), 6, -1, 384).cpu())
            t = torch.from_numpy(np.stack(tg)).to(dev)  # (B,144,256,4)
            rgb = t[..., :3] * t[..., 3:]
            z = vae.encode(rgb.permute(0, 3, 1, 2) * 2 - 1).latent_dist.mean * 0.18215
            lats.append(z.half().cpu())
            covs.append(F.avg_pool2d(t[..., 3][:, None], 8)[:, 0].half().cpu())
    if a.tokens_from:
        src = torch.load(f"{a.tokens_from}/{scene}.pt")
        assert src["sample_tokens"] == stoks, scene
        toks = [src["tokens"]]
    torch.save({"tokens": torch.cat(toks), "lat": torch.cat(lats), "cover": torch.cat(covs),
                "sample_tokens": stoks, "timestamps": times}, dst)
    print(scene, len(files), flush=True)
print("ENCODE_DONE")
