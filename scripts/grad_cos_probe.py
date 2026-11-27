"""Gradient-cosine probe (Fable, review 22:30): on fixed train clips, do the plain MSE loss and the adjacent-tick difference loss
push the temporal blocks in different directions? If cos(grad_MSE, grad_DIFF) ~ 1 the difference loss adds nothing (A3's arm
would repeat vlive). Read-only on checkpoints; separate from the tested code.

  python scripts/grad_cos_probe.py --ckpt runs/vlive/ckpt_8000.pt --out runs/probe/gradcos_8000.json
"""
import argparse, json, os
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.causal_video import add_causal_temporal, set_mode

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True)
p.add_argument("--enc", default="data/enc_tpv10"); p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt"); p.add_argument("--clips", type=int, default=8)
a = p.parse_args()
dev = "cuda"; SD = "sd-legacy/stable-diffusion-v1-5"; AMP = torch.bfloat16
ck = torch.load(a.ckpt, map_location="cpu"); r = ck["args"]["rank"]; K = ck["K"]; RHO = ck["rho"]; Fn = 16
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); unet.load_state_dict(ck["lora"], strict=False)
unet.requires_grad_(False); unet.to(dev, AMP).eval()
TW = add_causal_temporal(unet, K)
for w, sd_ in zip(TW, ck["temporal"]):
    w.temporal.load_state_dict(sd_)
for w in TW:
    w.temporal.to(dev).float().requires_grad_(True)
params = [q for w in TW for q in w.temporal.parameters()]
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
acp = DDPMScheduler.from_pretrained(SD, subfolder="scheduler").alphas_cumprod.to(dev)
RAYS = torch.load(a.rays); TRAIN = json.load(open(a.lists))["train"]
starts = []
for sc in sorted(TRAIN)[:40]:
    e = torch.load(f"{a.enc}/{sc}.pt")
    ok, seq, ticks, hard = e["ok"].tolist(), e["seq"].tolist(), e["ticks"].tolist(), e["hard"].tolist()
    for i in range(0, len(ticks) - Fn + 1, 40):
        if all(ok[i + k] and not hard[i + k] and seq[i + k] == seq[i] and ticks[i + k] == ticks[i] + k for k in range(Fn)):
            starts.append((sc, e, i)); break
g = torch.Generator().manual_seed(99)
res = {}
for tv in (900, 700, 300):
    cos_l = []; ratio_l = []
    for sc, e, i in starts[:a.clips]:
        tok = e["tokens"][i:i + Fn].float().to(dev); x0 = e["lat"][i:i + Fn].float().to(dev)
        ray = RAYS[sc][None].expand(Fn, *RAYS[sc].shape).to(dev).float()
        eps = (RHO ** 0.5 * torch.randn(1, 4, 32, 64, generator=g) + (1 - RHO) ** 0.5 * torch.randn(Fn, 4, 32, 64, generator=g)).to(dev)
        tt = torch.full((Fn,), tv, device=dev); ab = acp[tt].view(-1, 1, 1, 1); xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
            ctx = adapter(tok, ray, torch.full((Fn,), 2, dtype=torch.long, device=dev))
        set_mode(TW, "clip", Fn)
        grads = []
        for kind in ("mse", "diff"):
            with torch.autocast("cuda", dtype=AMP):
                pr = unet(xt, tt, encoder_hidden_states=ctx).sample.float()
            if kind == "mse":
                loss = ((pr[1:] - eps[1:]) ** 2).mean()
            else:
                loss = (((pr[2:] - pr[1:-1]) - (eps[2:] - eps[1:-1])) ** 2).mean()
            gr = torch.autograd.grad(loss, params)
            grads.append(torch.cat([x.flatten() for x in gr]))
        cos_l.append(torch.nn.functional.cosine_similarity(grads[0], grads[1], dim=0).item())
        ratio_l.append((grads[1].norm() / grads[0].norm()).item())
    res[tv] = {"cos_mean": round(sum(cos_l) / len(cos_l), 4), "cos_min": round(min(cos_l), 4), "norm_ratio_diff_over_mse": round(sum(ratio_l) / len(ratio_l), 4), "n": len(cos_l)}
    print(tv, res[tv], flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"ckpt": a.ckpt, "rho": RHO, "t": res}, open(a.out, "w"), indent=1); print("GRADCOS_DONE")
