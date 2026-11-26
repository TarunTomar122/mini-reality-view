"""Car map diagnostic (for the council after the 2k stop rule fired): how much does the map branch change the
renderer's noise prediction, and does it lower the loss?

  python scripts/carmap_diag.py --ckpts runs/cmX1/ckpt_2000.pt runs/cmX1/ckpt_4000.pt --out runs/judge/cm_diag.json

Held-out decision keyframes (30 drives x up to 4 frames), fixed noise; at t in {100, 300, 500, 700, 900}:
epsilon MSE with the right map, with the map of the drive half the list away, and with the branch off; the size of
the branch's change to the prediction (RMS of eps_right - eps_off) vs the RMS of eps_off; and the branch residual
RMS at each scale. Also the same error restricted to latent cells the right map covers (car pixels).
"""
import argparse, glob, json, os
import numpy as np
import torch, torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.carmap import rasterize, pool, CarMapBranch, C_MAP, grid_channels

p = argparse.ArgumentParser()
p.add_argument("--ckpts", nargs="+", required=True)
p.add_argument("--boxes", default="tmp/bev_boxes_decision_zero.json")
p.add_argument("--grid", default="data/bev_tpv")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--out", required=True)
a = p.parse_args()
dev, AMP = "cuda", torch.bfloat16
SD = "sd-legacy/stable-diffusion-v1-5"
D = json.load(open(a.lists))["decision"]
BOX = json.load(open(a.boxes)); RAYS = torch.load("data/calib_rays.pt")
rows = []
for sc in D:
    k = torch.load(f"data/lat512_tpv/{sc}.pt"); e = torch.load(f"data/enc_full/{sc}.pt")
    pos = {t: i for i, t in enumerate(e["sample_tokens"])}
    js = [j for j, t in enumerate(k["sample_tokens"]) if t in BOX and t in pos]
    for j in [js[i] for i in np.linspace(0, len(js) - 1, min(4, len(js))).round().astype(int)]:
        t = k["sample_tokens"][j]; rows.append((sc, t, e["tokens"][pos[t]], k["lat"][j]))
print("rows", len(rows), flush=True)
sched = DDPMScheduler.from_pretrained(SD, subfolder="scheduler"); acp = sched.alphas_cumprod.to(dev)
res = {}
for ckp in a.ckpts:
    ck = torch.load(ckp, map_location="cpu"); r = ck["args"]["rank"]
    unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
    unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); unet.load_state_dict(ck["lora"], strict=False)
    unet.to(dev, AMP).eval()
    ad = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)); ad.load_state_dict(ck["adapter"]); ad.to(dev).eval()
    br = CarMapBranch(c_in=C_MAP + (34 if ck["grid"] else 0)).to(dev).eval(); br.load_state_dict(ck["branch"])
    red = None
    if ck["grid"]:
        red = nn.Conv2d(512, 32, 1).to(dev).eval(); red.load_state_dict(ck["reduce"])

    def mapof(i):
        sc, t = rows[i][0], rows[i][1]
        m = pool(rasterize(BOX.get(t)))[None].to(dev)
        if red is not None:
            g = torch.load(f"{a.grid}/{sc}.pt"); G = grid_channels(g["bev"][g["sample_tokens"].index(t)])[None].to(dev)
            m = torch.cat([m, red(G[:, :512]), G[:, 512:]], 1)
        return m
    out = {}
    for T in (100, 300, 500, 700, 900):
        acc = {"mse_right": [], "mse_swap": [], "mse_off": [], "car_mse_right": [], "car_mse_off": [], "delta_rms": [], "eps_rms": [], "res_rms": []}
        for i in range(len(rows)):
            sc, t, tok, lat = rows[i]
            g = torch.Generator().manual_seed(7 * i + T)
            eps = torch.randn(1, 4, 32, 64, generator=g).to(dev); x0 = lat[None].float().to(dev)
            ab = acp[T]; xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            tt = torch.full((1,), T, device=dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
                c = ad(tok[None].float().to(dev), RAYS[sc][None].float().to(dev), torch.full((1,), 2, device=dev))
                M = mapof(i); Ms = mapof((i + len(rows) // 2) % len(rows))
                R = br(M); Rs = br(Ms)
                pr = unet(xt, tt, encoder_hidden_states=c, down_intrablock_additional_residuals=[x.to(AMP) for x in R]).sample.float()
                ps = unet(xt, tt, encoder_hidden_states=c, down_intrablock_additional_residuals=[x.to(AMP) for x in Rs]).sample.float()
                po = unet(xt, tt, encoder_hidden_states=c).sample.float()
            cov = (M[0, 0:2].max(0).values > 0.05)
            acc["mse_right"].append(((pr - eps) ** 2).mean().item()); acc["mse_swap"].append(((ps - eps) ** 2).mean().item())
            acc["mse_off"].append(((po - eps) ** 2).mean().item())
            if cov.any():
                acc["car_mse_right"].append(((pr - eps) ** 2)[0][:, cov].mean().item()); acc["car_mse_off"].append(((po - eps) ** 2)[0][:, cov].mean().item())
            acc["delta_rms"].append((pr - po).pow(2).mean().sqrt().item()); acc["eps_rms"].append(po.pow(2).mean().sqrt().item())
            acc["res_rms"].append([x.float().pow(2).mean().sqrt().item() for x in R])
        out[T] = {k: (np.mean(v, 0).tolist() if k == "res_rms" else float(np.mean(v))) for k, v in acc.items()}
        print(os.path.basename(os.path.dirname(ckp)), os.path.basename(ckp), T, {k: (np.round(v, 5).tolist() if isinstance(v, list) else round(v, 5)) for k, v in out[T].items()}, flush=True)
    res[ckp] = out
    del unet
json.dump(res, open(a.out, "w"), indent=1)
print("CARMAP_DIAG_DONE")
