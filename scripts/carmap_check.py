"""Car map R single-step check (council/PLAN-carmap.md v4): does the map conv USE the map? Dev drives only.

  python scripts/carmap_check.py --ckpt runs/cmR/ckpt_1000.pt --boxes 'tmp/bevF_boxes_train_*.json' --out runs/judge/cmR_1000_check.json

Per keyframe of the dev drives (data/lat512_tpv, tokens data/enc_full, key car masks data/keycar_tpv): fixed noise per
(keyframe, t), cameras ON, t in {700, 900}; eps prediction with the drive's own map and with the map of a fixed donor
(the dev drive half the list away, same relative position). Records car-cell eps-MSE (key car mask weights) right vs
swapped, the share of car-bearing keyframes where right < swapped, the mean relative reduction (swapped - right) /
swapped, and RMS(eps_right - eps_swapped). Hard gate at 1k: share >= 0.75 and mean reduction >= 2%.
"""
import argparse, glob, json, os, zlib
import numpy as np
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.carmap import rasterize, pool, MapConvIn, N_MAP, CarMapBranchBF

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--boxes", required=True)
p.add_argument("--drives", default="", help="comma list (default: the dev list)")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--keycar", default="data/keycar_tpv")
p.add_argument("--out", required=True)
a = p.parse_args()
dev, AMP = "cuda", torch.bfloat16
SD = "sd-legacy/stable-diffusion-v1-5"
drives = a.drives.split(",") if a.drives else json.load(open(a.lists))["dev"]
BOX = {}
for f in [x for pat in a.boxes.split(",") for x in glob.glob(pat)]:  # comma-separated globs
    BOX.update(json.load(open(f)))
RAYS = torch.load("data/calib_rays.pt")
rows = {}
for sc in drives:
    k = torch.load(f"data/lat512_tpv/{sc}.pt"); e = torch.load(f"data/enc_full/{sc}.pt"); kc = torch.load(f"{a.keycar}/{sc}.pt")
    pos = {t: i for i, t in enumerate(e["sample_tokens"])}; kpos = {t: i for i, t in enumerate(kc["sample_tokens"])}
    missing = [t for t in k["sample_tokens"] if t not in BOX]
    assert not missing, f"{sc}: {len(missing)} keyframes without boxes (coverage must be complete)"
    rows[sc] = [(t, e["tokens"][pos[t]], k["lat"][j], kc["mask"][kpos[t]].float()) for j, t in enumerate(k["sample_tokens"])]
ck = torch.load(a.ckpt, map_location="cpu"); r = ck["args"]["rank"]
if not ("branch4" in ck or "mapconv" in ck):
    raise SystemExit("checkpoint has neither a map conv (R) nor a bias-free 4-scale branch (F)")
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); unet.load_state_dict(ck["lora"], strict=False)
unet.to(dev, AMP).eval()
BR4 = None
if "branch4" in ck:  # fallback F: bias-free 4-scale branch
    BR4 = CarMapBranchBF().to(dev).float().eval(); BR4.load_state_dict(ck["branch4"])
    with torch.no_grad():  # an empty map must give exactly zero at all four scales
        z_ = BR4(torch.zeros(1, N_MAP, 32, 64, device=dev))
    if any(float(r_.abs().max()) != 0.0 for r_ in z_):
        raise SystemExit("branch4: empty map does not give exactly zero residuals")
else:
    unet.conv_in = MapConvIn(unet.conv_in); unet.conv_in.map.load_state_dict(ck["mapconv"]); unet.conv_in.map.to(dev).float().eval()
ad = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)); ad.load_state_dict(ck["adapter"]); ad.to(dev).eval()
acp = DDPMScheduler.from_pretrained(SD, subfolder="scheduler").alphas_cumprod.to(dev)
st = torch.full((1,), 2, dtype=torch.long, device=dev)
res = {"ckpt": a.ckpt, "drives": drives, "per": []}
for di, sc in enumerate(drives):
    donor = drives[(di + len(drives) // 2) % len(drives)]
    for n_, (tok, tk, lat, w) in enumerate(rows[sc]):
        dt = rows[donor][min(round(n_ / max(1, len(rows[sc]) - 1) * (len(rows[donor]) - 1)), len(rows[donor]) - 1)][0]
        Mr = pool(rasterize(BOX[tok]))[:N_MAP][None].to(dev); Ms = pool(rasterize(BOX[dt]))[:N_MAP][None].to(dev)
        W = w.to(dev)[None, None]
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
            c = ad(tk[None].float().to(dev), RAYS[sc][None].float().to(dev), st)
        for T in (700, 900):
            g = torch.Generator().manual_seed(zlib.crc32(f"check|{sc}|{tok}|{T}".encode()))
            eps = torch.randn(1, 4, 32, 64, generator=g).to(dev); ab = acp[T]
            xt = ab.sqrt() * lat[None].float().to(dev) + (1 - ab).sqrt() * eps; tt = torch.full((1,), T, device=dev)
            out = {}
            for name, M in (("right", Mr), ("swap", Ms), ("off", None)):
                with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
                    if BR4 is None:
                        unet.conv_in.m = M
                        out[name] = unet(xt, tt, encoder_hidden_states=c).sample.float()
                    else:
                        res_ = None if M is None else [r_.to(AMP) for r_ in BR4(M)]
                        out[name] = unet(xt, tt, encoder_hidden_states=c, down_intrablock_additional_residuals=res_).sample.float()
            if BR4 is None:
                unet.conv_in.m = None
            ce = lambda pr: float((((pr - eps) ** 2).mean(1, keepdim=True) * W).sum() / W.sum()) if W.sum() > 0 else None
            res["per"].append({"scene": sc, "token": tok, "t": T, "car_cells": float((W > 0).float().sum()),
                               "car_right": ce(out["right"]), "car_swap": ce(out["swap"]), "car_off": ce(out["off"]),
                               "rms_right_swap": float((out["right"] - out["swap"]).pow(2).mean().sqrt()),
                               "rms_right_off": float((out["right"] - out["off"]).pow(2).mean().sqrt())})
for T in (700, 900):
    P = [q for q in res["per"] if q["t"] == T and q["car_right"] is not None]
    share = float(np.mean([q["car_right"] < q["car_swap"] for q in P])) if P else None
    red = float(np.mean([(q["car_swap"] - q["car_right"]) / q["car_swap"] for q in P])) if P else None
    res[f"t{T}"] = {"keyframes": len(P), "share_right_better": share, "mean_rel_reduction": red,
                    "rms_right_swap": float(np.mean([q["rms_right_swap"] for q in res["per"] if q["t"] == T])),
                    "rms_right_off": float(np.mean([q["rms_right_off"] for q in res["per"] if q["t"] == T]))}
res["gate_1k"] = all(res[f"t{T}"]["share_right_better"] is not None and res[f"t{T}"]["share_right_better"] >= 0.75
                     and res[f"t{T}"]["mean_rel_reduction"] >= 0.02 for T in (700, 900))
json.dump(res, open(a.out, "w"), indent=1)
print("CHECK", json.dumps({k: v for k, v in res.items() if k != "per"}))
