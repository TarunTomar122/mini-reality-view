"""Temporal data, step 3 (council/PLAN-temporal.md): cache inputs and answers for every usable dense tick.

  python scripts/temporal_encode.py --manifest data/temporal_manifest.json --keys data/sim/keys_tpv10 \
      --out data/enc_tpv10 [--shard i/n]

Per drive -> <out>/<scene>.pt with
  ticks (n,) int, seq (n,) int (sequence id; consecutive ticks in one seq can be chained), ok (n,) bool,
  tokens (n, 6, 144, 384) fp16 (DINOv2-small, 448x252, 2x2 pooled, same as scripts/encode.py --pool 2),
  lat (n, 4, 32, 64) fp16 (SD VAE of the game key, top 32 rows cropped, x0.18215, same as encode_latents.py),
  key_token: sample token for ticks that are keyframes (else "").
Calibrated rays stay in data/calib_rays.pt (per scene). Rejected ticks keep their latent (useful as targets) but
get zero tokens and ok=False.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--manifest", default="data/temporal_manifest.json")
p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--dataroot", default="data/nuscenes_full")
p.add_argument("--out", default="data/enc_tpv10")
p.add_argument("--shard", default="0/1")
a = p.parse_args()
dev = "cuda"
from transformers import AutoModel
from diffusers import AutoencoderKL
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).half().eval()
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
man = json.load(open(a.manifest))
si, sn = map(int, a.shard.split("/"))
os.makedirs(a.out, exist_ok=True)
for sc in sorted(man)[si::sn]:
    if os.path.exists(f"{a.out}/{sc}.pt"):
        continue
    rows = man[sc]
    toks, lats = [], []
    with torch.no_grad():
        for r in rows:
            if r["ok"]:
                ims = [np.asarray(Image.open(f"{a.dataroot}/{f}").convert("RGB").resize((448, 252), Image.BICUBIC)) for f in r["files"]]
                x = torch.from_numpy(np.stack(ims)).to(dev).permute(0, 3, 1, 2).float() / 255
                h = dino(pixel_values=((x - MEAN) / STD).half()).last_hidden_state[:, 1:]
                toks.append(F.avg_pool2d(h.float().view(6, 18, 32, 384).permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2).half().cpu())
            else:
                toks.append(torch.zeros(6, 144, 384, dtype=torch.float16))
            key = np.asarray(Image.open(f"{a.keys}/{sc}/t{r['tick']:04d}.jpg").convert("RGB").resize((512, 288), Image.BILINEAR))[32:]
            k = torch.from_numpy(key).to(dev).permute(2, 0, 1)[None].float() / 127.5 - 1
            lats.append((vae.encode(k).latent_dist.mean * 0.18215)[0].half().cpu())
    torch.save({"ticks": torch.tensor([r["tick"] for r in rows]), "seq": torch.tensor([r["seq"] for r in rows]),
                "ok": torch.tensor([r["ok"] for r in rows]), "tokens": torch.stack(toks), "lat": torch.stack(lats),
                "key_token": [r["token"] or "" for r in rows]}, f"{a.out}/{sc}.pt")
    print(sc, len(rows), "ticks", sum(r["ok"] for r in rows), "usable", flush=True)
print("TEMPORAL_ENCODE_DONE")
