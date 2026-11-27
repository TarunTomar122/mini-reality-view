"""Joint multi-frame video model V (council/video-next-fable.md + swaps, both reviewers): temporal blocks (mrv/video.py) in
the FROZEN E3 UNet (LoRA + CameraAdapter frozen too). Clips of 16 consecutive usable 10 Hz ticks of the 144 train drives
(data/enc_tpv10: DINO tokens + key latents per tick); batch 2 clips; one timestep per clip; PYoCo mixed noise
eps_i = (eps_clip + eps_i') / sqrt(2); epsilon loss; AdamW 1e-4 (500 warm-up), clip 1.0, bf16; camera dropout 10% per
whole clip; checkpoints 2k / 4k / 8k / 12k (extend to 20k if dev still improves 8k -> 12k).

  python scripts/train_video.py --init runs/tpvE3/ckpt_10000.pt --out runs/vidV [--steps 12000] [--budget_min 0]
"""
import argparse, json, os, time
import numpy as np
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.video import add_temporal, set_frames

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--out", required=True)
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--frames", type=int, default=16)
p.add_argument("--clips", type=int, default=2)
p.add_argument("--steps", type=int, default=12000)
p.add_argument("--save_at", default="2000,4000,8000,12000,16000,20000")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--grad_ckpt", action="store_true")
p.add_argument("--budget_min", type=float, default=0)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev, AMP = "cuda", torch.bfloat16
SD = "sd-legacy/stable-diffusion-v1-5"
TRAIN = json.load(open(a.lists))["train"]
RAYS = torch.load(a.rays)
Fn = a.frames

# ---------- clips: Fn consecutive usable ticks in one sequence ----------
E, starts = {}, []
for sc in sorted(TRAIN):
    e = torch.load(f"{a.enc}/{sc}.pt"); E[sc] = e
    ok, seq, ticks = e["ok"].tolist(), e["seq"].tolist(), e["ticks"].tolist()
    for i in range(len(ticks) - Fn + 1):
        if all(ok[i + k] for k in range(Fn)) and all(seq[i + k] == seq[i] and ticks[i + k] == ticks[i] + k for k in range(Fn)):
            starts.append((sc, i))
print(f"clips: {len(starts)} start positions of {Fn} ticks from {len(E)} drives", flush=True)
if len({s for s, _ in starts}) != len(TRAIN):
    print("note: drives without a full clip:", sorted(set(TRAIN) - {s for s, _ in starts}), flush=True)

# ---------- model: frozen E3 + temporal blocks ----------
ck0 = torch.load(a.init, map_location="cpu"); r = ck0["args"]["rank"]
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); unet.load_state_dict(ck0["lora"], strict=False)
unet.requires_grad_(False); unet.to(dev, AMP)
TW = add_temporal(unet)
for w in TW:
    w.temporal.to(dev).float().requires_grad_(True)
if a.grad_ckpt:
    unet.enable_gradient_checkpointing()
unet.train()
adapter = CameraAdapter(n_styles=ck0["args"].get("n_styles", 3)); adapter.load_state_dict(ck0["adapter"])
adapter.requires_grad_(False); adapter.to(dev).eval()
params = [p_ for w in TW for p_ in w.temporal.parameters()]
print("temporal params:", sum(p_.numel() for p_ in params), flush=True)
opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
acp = DDPMScheduler.from_pretrained(SD, subfolder="scheduler").alphas_cumprod.to(dev)
start, ck = 0, f"{a.out}/ckpt.pt"
if os.path.exists(ck):
    if not a.resume:
        raise SystemExit(f"{ck} exists: pass --resume or use a fresh --out")
    s_ = torch.load(ck, map_location="cpu")
    for w, sd_ in zip(TW, s_["temporal"]):
        w.temporal.load_state_dict(sd_)
    opt.load_state_dict(s_["opt"]); start = s_["step"]; print("resumed at", start, flush=True)


def save(step, path):
    d = {"temporal": [w.temporal.state_dict() for w in TW], "opt": opt.state_dict(), "step": step, "init": a.init,
         "frames": Fn, "args": {**ck0["args"], "video": vars(a)}, "adapter": ck0["adapter"], "lora": ck0["lora"]}
    torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)


SAVE = set(int(x) for x in a.save_at.split(","))
log = open(f"{a.out}/log.jsonl", "a"); t0 = time.time(); t10 = t0
style = torch.full((a.clips * Fn,), 2, dtype=torch.long, device=dev)
for step in range(start, a.steps):
    if step - start == 10:
        torch.cuda.synchronize(); t10 = time.time()
    g = torch.Generator().manual_seed(a.seed * 100003 + step)
    pick = torch.randint(0, len(starts), (a.clips,), generator=g).tolist()
    t = torch.randint(0, 1000, (a.clips,), generator=g)
    eps_clip = torch.randn(a.clips, 1, 4, 32, 64, generator=g); eps_own = torch.randn(a.clips, Fn, 4, 32, 64, generator=g)
    eps = ((eps_clip + eps_own) / 2 ** 0.5).flatten(0, 1)  # PYoCo mixed noise
    drop = (torch.rand(a.clips, generator=g) < 0.1).repeat_interleave(Fn)
    tok, lat, ray = [], [], []
    for sc, i in (starts[k] for k in pick):
        e = E[sc]
        tok.append(e["tokens"][i:i + Fn].float()); lat.append(e["lat"][i:i + Fn].float()); ray.append(RAYS[sc][None].expand(Fn, *RAYS[sc].shape))
    tok = torch.cat(tok).to(dev); x0 = torch.cat(lat).to(dev); ray = torch.cat(ray).to(dev).float()
    tt = t.repeat_interleave(Fn).to(dev); eps = eps.to(dev)
    ab = acp[tt].view(-1, 1, 1, 1); xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    for g_ in opt.param_groups:
        g_["lr"] = 1e-4 * min(1, (step + 1) / 500)
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        ctx = adapter(tok, ray, style, drop.to(dev))
    set_frames(TW, Fn)
    with torch.autocast("cuda", dtype=AMP):
        pred = unet(xt, tt, encoder_hidden_states=ctx).sample
    loss = ((pred.float() - eps) ** 2).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if step % 100 == 0:
        rec = {"step": step, "loss": round(loss.item(), 4), "gn": round(gn.item(), 4),
               "out_norm": round(sum(w.temporal.out.weight.norm().item() ** 2 for w in TW) ** 0.5, 4),
               "min": round((time.time() - t0) / 60, 1), "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    if (step + 1) % 1000 == 0 or step + 1 == a.steps:
        save(step + 1, ck)
    if step + 1 in SAVE:
        save(step + 1, f"{a.out}/ckpt_{step + 1}.pt")
if a.budget_min:
    per = (time.time() - t10) / max(1, a.steps - start - 10)
    proj = (12000 * per + 120) / 60
    print(f"TIMING per update {per:.3f} s -> projected 12k {proj:.0f} min (budget {a.budget_min:.0f})", flush=True)
    if proj > a.budget_min:
        print("TIMING_OVER_BUDGET", flush=True); raise SystemExit(3)
print("VIDEO_TRAIN_DONE")
