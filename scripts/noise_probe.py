"""Offline probe: does the causal adapter use past CONTENT or the shared NOISE? (council review 21:30: Astra "vary prefix noise
and content independently, recipient tick fixed"; Neon "past-same-content-fresh-noise arm"; Fable "difference-loss probe").
Separate from the tested training / rollout code; read-only on checkpoints.

Clips: the trainer's 8 fixed dev probe clips (K+1 = 5 ticks; same seeds / selection as scripts/train_causal.py, recipient =
last tick, donor = a clip from a DIFFERENT dev drive). Noise: eps_i = sqrt(rho) e_clip + sqrt(1-rho) e_i (rho from the ckpt).
Arms (loss = eps-MSE on the recipient tick only; the recipient's own latent, cameras and noise never change):
  on            true past content + true (shared) past noise
  off           temporal blocks off
  noise_only    DONOR past content + recipient's shared past noise           (only the noise carries over)
  content_only  true past content + FRESH independent past noise             (only the content carries over)
  neither       donor past content + fresh past noise
Fable's difference loss: over ticks 1..K of the true clip, MSE of (pred_i - pred_{i-1}) vs (eps_i - eps_{i-1}); the shared
noise cancels, so only content use can lower it (reported on / off).
Shares: content_gain = off - content_only, noise_gain = off - noise_only, full_gain = off - on (positive = helps).

  python scripts/noise_probe.py --ckpt runs/vlive/ckpt_2000.pt --out runs/probe/noise_2000.json
"""
import argparse, json
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.causal_video import add_causal_temporal, set_mode

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True)
p.add_argument("--enc", default="data/enc_tpv10"); p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt"); p.add_argument("--reps", type=int, default=4, help="noise draws per clip")
p.add_argument("--noise", default="ckpt", choices=["ckpt", "private"], help="probe noise: the checkpoint's rho mix (shared) or fully private per tick (rho 0); A3 reporting")
a = p.parse_args()
dev = "cuda"; SD = "sd-legacy/stable-diffusion-v1-5"
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
ck = torch.load(a.ckpt, map_location="cpu"); r = ck["args"]["rank"]; K = ck["K"]; RHO = ck["rho"] if a.noise == "ckpt" else 0.0; L = K + 1
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); unet.load_state_dict(ck["lora"], strict=False)
unet.to(dev, AMP).eval()
TW = add_causal_temporal(unet, K)
for w, sd_ in zip(TW, ck["temporal"]):
    w.temporal.load_state_dict(sd_)
for w in TW:
    w.temporal.to(dev, AMP).eval()
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
acp = DDPMScheduler.from_pretrained(SD, subfolder="scheduler").alphas_cumprod.to(dev)
RAYS = torch.load(a.rays); DEVL = json.load(open(a.lists))["dev"]

# the trainer's probe clips (same generator recipe as scripts/train_causal.py)
gp = torch.Generator().manual_seed(4242); pc = {}
for sc in DEVL:
    e = torch.load(f"{a.enc}/{sc}.pt")
    ok, seq, ticks, hard = e["ok"].tolist(), e["seq"].tolist(), e["ticks"].tolist(), e["hard"].tolist()
    st_ = [i for i in range(len(ticks) - L + 1) if all(ok[i + k] and not hard[i + k] and seq[i + k] == seq[i] and ticks[i + k] == ticks[i] + k for k in range(L))]
    pc[sc] = [(e["tokens"][i:i + L].float(), e["lat"][i:i + L].float()) for i in (st_[k_] for k_ in torch.randperm(len(st_), generator=gp)[:2].tolist())]
PAIRS = []
for pi in range(8):
    rsc, dsc = DEVL[pi % len(DEVL)], DEVL[(pi + 1) % len(DEVL)]
    PAIRS.append(((rsc,) + pc[rsc][(pi // len(DEVL)) % len(pc[rsc])], (dsc,) + pc[dsc][0]))

g = torch.Generator().manual_seed(777)
res = {}
with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
    for tv in (900, 700, 300):
        acc = {k: 0.0 for k in ("on", "off", "noise_only", "content_only", "neither", "diff_on", "diff_off")}; n = 0
        for (rsc, tk, lt), (dsc, otk, olt) in PAIRS:
            for _ in range(a.reps):
                e_clip = torch.randn(1, 4, 32, 64, generator=g); e_own = torch.randn(L, 4, 32, 64, generator=g)
                e_fresh = torch.randn(L, 4, 32, 64, generator=g)  # independent noise for the past ticks (content_only / neither)
                eps = (RHO ** 0.5 * e_clip + (1 - RHO) ** 0.5 * e_own).to(dev)
                tt = torch.full((L,), tv, device=dev); ab = acp[tt].view(-1, 1, 1, 1)
                for kind in ("on", "off", "noise_only", "content_only", "neither"):
                    x0 = lt.clone(); tkk = tk.clone(); rays = RAYS[rsc][None].expand(L, *RAYS[rsc].shape).clone(); ep = eps.clone()
                    if kind in ("noise_only", "neither"):
                        x0[:-1] = olt[:-1]; tkk[:-1] = otk[:-1]; rays[:-1] = RAYS[dsc][None]
                    if kind in ("content_only", "neither"):
                        ep[:-1] = e_fresh[:-1].to(dev)
                    xt = ab.sqrt() * x0.to(dev) + (1 - ab).sqrt() * ep
                    ctx = adapter(tkk.to(dev), rays.to(dev).float(), torch.full((L,), 2, dtype=torch.long, device=dev))
                    set_mode(TW, "off" if kind == "off" else "clip", L)
                    pr = unet(xt, tt, encoder_hidden_states=ctx).sample.float()
                    acc[kind] += ((pr[-1] - ep[-1]) ** 2).mean().item()
                    if kind in ("on", "off"):  # Fable's difference loss over the true clip
                        acc["diff_" + kind] += (((pr[1:] - pr[:-1]) - (ep[1:] - ep[:-1])) ** 2).mean().item()
                n += 1
        acc = {k: v / n for k, v in acc.items()}
        acc["content_gain"] = acc["off"] - acc["content_only"]; acc["noise_gain"] = acc["off"] - acc["noise_only"]
        acc["full_gain"] = acc["off"] - acc["on"]; acc["diff_gain"] = acc["diff_off"] - acc["diff_on"]
        res[tv] = {k: round(v, 6) for k, v in acc.items()}
        print(tv, res[tv], flush=True)
out = {"ckpt": a.ckpt, "step": ck.get("step"), "rho": RHO, "noise": a.noise, "K": K, "reps": a.reps, "pairs": [[x[0], y[0]] for x, y in PAIRS], "t": res}
import os; os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
print("NOISE_PROBE_DONE", a.ckpt)
