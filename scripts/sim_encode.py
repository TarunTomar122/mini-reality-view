"""Experiment S: camera tokens for the sim rig images (clean + one phone-augmented copy), with matching rays.

Reads data/sim/cams/<scene>/<k>_<token>_<CAM>.jpg (448x252 renders placed with the scene's real calibration) and writes
  data/enc_sim/<scene>.pt      {tokens (n,6,144,384) fp16, rays (n,6,144,6), sample_tokens}
  data/enc_sim_aug/<scene>.pt  same, after the shared phone augmentation (mrv/aug.py), rays recomputed for the crop.
Sim keys are encoded separately with encode_latents.py (--targets data/sim/keys --out data/lat512_sim).
"""
import argparse, glob, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from nuscenes.nuscenes import NuScenes
from transformers import AutoModel
from mrv.nusc import CAMS
from mrv.aug import augment, rays_for

p = argparse.ArgumentParser()
p.add_argument("--cams", default="data/sim/cams")
p.add_argument("--dataroot", default="data/nuscenes_full")
p.add_argument("--version", default="v1.0-trainval")
p.add_argument("--shard", default="0/1")
a = p.parse_args()
dev = "cuda"
nusc = NuScenes(a.version, a.dataroot, verbose=False)
by_name = {s["name"]: s for s in nusc.scene}
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).half().eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
os.makedirs("data/enc_sim", exist_ok=True); os.makedirs("data/enc_sim_aug", exist_ok=True)
scenes = sorted(os.listdir(a.cams))
si, sn = map(int, a.shard.split("/"))
rng = np.random.default_rng(4242 + si)
S = 1600 / 448  # sim renders are 448x252; calibration is in 1600x900 pixels


@torch.no_grad()
def tokens(imgs):
    x = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2).float() / 255
    h = dino(pixel_values=((x - MEAN) / STD).half()).last_hidden_state[:, 1:]
    return F.avg_pool2d(h.view(-1, 18, 32, 384).permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2).reshape(-1, 6, 144, 384).cpu()


for sc in scenes[si::sn]:
    if os.path.exists(f"data/enc_sim_aug/{sc}.pt"):
        continue
    keys = sorted({os.path.basename(f).rsplit("_CAM_", 1)[0] for f in glob.glob(f"{a.cams}/{sc}/*.jpg")})
    s0 = nusc.get("sample", by_name[sc]["first_sample_token"])
    cs = {c: nusc.get("calibrated_sensor", nusc.get("sample_data", s0["data"][c])["calibrated_sensor_token"]) for c in CAMS}
    T, R, TA, RA, stoks = [], [], [], [], []
    for i in range(0, len(keys), 8):
        clean, aug, rr, rra = [], [], [], []
        for kk in keys[i:i + 8]:
            stoks.append(kk[4:])
            for c in CAMS:
                im = Image.open(f"{a.cams}/{sc}/{kk}_{c}.jpg").convert("RGB")
                clean.append(np.asarray(im.resize((448, 252))))
                rr.append(rays_for(cs[c], (0, 0, 1600, 900)))
                ai, (ox, oy, cw, ch) = augment(im, rng)
                aug.append(ai)
                rra.append(rays_for(cs[c], (ox * S, oy * S, cw * S, ch * S)))
        n = len(keys[i:i + 8])
        T.append(tokens(clean)); TA.append(tokens(aug))
        R.append(torch.tensor(np.stack(rr), dtype=torch.float32).reshape(n, 6, 144, 6))
        RA.append(torch.tensor(np.stack(rra), dtype=torch.float32).reshape(n, 6, 144, 6))
    torch.save({"tokens": torch.cat(T).half(), "rays": torch.cat(R), "sample_tokens": stoks}, f"data/enc_sim/{sc}.pt")
    torch.save({"tokens": torch.cat(TA).half(), "rays": torch.cat(RA), "sample_tokens": stoks}, f"data/enc_sim_aug/{sc}.pt")
    print(sc, len(stoks), flush=True)
print("SIM_ENCODE_DONE")
