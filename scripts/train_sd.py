"""Step 3: Stable Diffusion 1.5 as the renderer.

SD 1.5's UNet already knows what streets, cars and trees look like. We keep it frozen and teach it our task with:
  - a camera adapter that turns the 6 cameras' DINOv2 tokens (+ their viewing rays) into 864 tokens of width 768,
    used where SD normally reads its text prompt (cross-attention);
  - LoRA (low-rank updates) on the UNet's attention layers;
  - a style token: 0 = lidar answer key, 1 = splat answer key (we render with 1).
Answer keys are 512x256 (latents 4 x 32 x 64, from encode_latents.py); the loss is weighted by the key's coverage /
trust map. SD's own noise schedule and epsilon prediction are kept so its prior stays useful.
10% of the time the camera tokens are replaced by a learned "no cameras" token set, for classifier-free guidance.
"""
import argparse, copy, glob, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler, AutoencoderKL
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS

SD = "sd-legacy/stable-diffusion-v1-5"
p = argparse.ArgumentParser()
p.add_argument("--tokens", default="data/enc_full", help="enc dir with pooled camera tokens per scene")
p.add_argument("--keys", default="data/lat512_full_low", help="style-0 key latents (encode_latents.py)")
p.add_argument("--keys2", default="", help="style-1 key latents (e.g. splat keys)")
p.add_argument("--mix", type=float, default=0.5)
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--tokens_aug", default="", help="comma list of encode_aug.py dirs: phone-like token copies (with their own rays) added as extra training rows")
p.add_argument("--init", default="", help="start adapter + LoRA from this checkpoint")
p.add_argument("--rank", type=int, default=64)
p.add_argument("--steps", type=int, default=30000)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--hole_w", type=float, default=0.05)
p.add_argument("--cover_min", default="", help="comma list, one per key set: latent cells whose coverage is below this get zero "
               "weight (lidar keys are black between points, so partly covered cells teach dark speckles); '' = off")
p.add_argument("--out", required=True)
p.add_argument("--render_scenes", type=int, default=8)
p.add_argument("--sample_steps", type=int, default=25)
p.add_argument("--cfg", type=float, default=2.0)
p.add_argument("--render_style", type=int, default=1)
p.add_argument("--eval_only", action="store_true")
p.add_argument("--set", action="append", default=[],
               help="keydir:tokdir:style:rows[:augdir+augdir]; repeatable. Fixed rows per batch per set, paired random draws.")
p.add_argument("--n_styles", type=int, default=2)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--save_at", default="", help="comma list of steps to keep a copy of the checkpoint (ckpt_<step>.pt)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
print("amp", AMP, torch.cuda.get_device_name(0), flush=True)
from nuscenes.utils.splits import val
VAL = set(val)
RAYS = torch.load(a.rays)


# ---------------- data ----------------
def load_set(keydir, style, tokdir=None, augdirs=None):
    tokdir = tokdir or a.tokens
    augdirs = [x for x in a.tokens_aug.split(",") if x] if augdirs is None else augdirs
    T, L, C, R, S, names = [], [], [], [], [], []
    for f in sorted(glob.glob(f"{keydir}/*.pt")):
        sc = os.path.basename(f)[:-3]
        k = torch.load(f)
        if not os.path.exists(f"{tokdir}/{sc}.pt"):
            continue
        e = torch.load(f"{tokdir}/{sc}.pt")
        pos = {t: i for i, t in enumerate(e["sample_tokens"])}
        ix = [pos[t] for t in k["sample_tokens"]]
        T.append(e["tokens"][ix]); L.append(k["lat"]); C.append(k["cover"])
        R.append(e["rays"][ix] if "rays" in e else RAYS[sc][None].expand(len(ix), -1, -1, -1))
        S += [style] * len(ix); names += [sc] * len(ix)
        for ad in augdirs:
            if sc in VAL or not os.path.exists(f"{ad}/{sc}.pt"):
                continue  # augmented copies only for training scenes
            g = torch.load(f"{ad}/{sc}.pt")
            gp = {t: i for i, t in enumerate(g["sample_tokens"])}
            gx = [gp[t] for t in k["sample_tokens"]]
            T.append(g["tokens"][gx]); L.append(k["lat"]); C.append(k["cover"]); R.append(g["rays"][gx])
            S += [style] * len(gx); names += [sc] * len(gx)
    return dict(T=torch.cat(T), L=torch.cat(L), C=torch.cat(C), R=torch.cat(R), S=torch.tensor(S), names=names)


def split(d):
    tr = torch.tensor([n not in VAL for n in d["names"]])
    return {k: (v[tr] if torch.is_tensor(v) else [x for x, m in zip(v, tr) if m]) for k, v in d.items()}


ROWS = []
if a.set:
    sets = []
    for spec in a.set:
        parts = spec.split(":")
        keydir, tokdir, style, rows = parts[0], parts[1], int(parts[2]), int(parts[3])
        augs = parts[4].split("+") if len(parts) > 4 and parts[4] else []
        sets.append(load_set(keydir, style, tokdir, augs)); ROWS.append(rows)
        sets[-1]["tokdir"] = tokdir
    a.bs = sum(ROWS)
    print("batch rows per set", ROWS, "-> bs", a.bs, flush=True)
else:
    sets = [load_set(a.keys, 0)] + ([load_set(a.keys2, 1)] if a.keys2 else [])
if a.cover_min:
    for s, thr in zip(sets, map(float, a.cover_min.split(","))):
        kept = (s["C"] >= thr).float().mean().item()
        s["C"] = s["C"] * (s["C"] >= thr)
        print(f"cover_min {thr}: {kept:.3f} of latent cells keep weight", flush=True)
train = [split({k: v for k, v in s.items() if k != "tokdir"}) for s in sets]
for s, t in zip(sets, train):
    print("set", len(s["L"]), "frames,", len(t["L"]), "train", flush=True)
# keep only the val rows each set needs for the final render, so the training rows are not held twice in RAM
# (61k token rows are ~40 GB; holding sets + train copies plus allocator growth hit 126 GB and an OOM kill)
for s in sets:
    vm = torch.tensor([n in VAL for n in s["names"]])
    for k in ("T", "L", "C", "R", "S"):
        s[k] = s[k][vm]
    s["names"] = [n for n, m in zip(s["names"], vm) if m]
import gc; gc.collect()
lat_scale = 1.0  # latents are already x0.18215 (SD convention)


# ---------------- model ----------------
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet", torch_dtype=torch.float32)
unet.requires_grad_(False)
unet.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights="gaussian",
                            target_modules=LORA_TARGETS))
for n_, p_ in unet.named_parameters():
    if not p_.requires_grad:
        p_.data = p_.data.to(AMP)  # frozen base weights in half precision, LoRA weights stay fp32
unet.enable_gradient_checkpointing()
unet.to(dev)
adapter = CameraAdapter(n_styles=a.n_styles).to(dev)
params = [p_ for p_ in unet.parameters() if p_.requires_grad] + list(adapter.parameters())
print("trainable params (M)", round(sum(p_.numel() for p_ in params) / 1e6, 1), flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
sched = DDPMScheduler.from_pretrained(SD, subfolder="scheduler")
acp = sched.alphas_cumprod.to(dev)
ck = f"{a.out}/ckpt.pt"
start = 0
if a.init and not os.path.exists(ck):
    s_ = torch.load(a.init, map_location=dev)
    adapter.load_state_dict(s_["adapter"]); print("init from", a.init, unet.load_state_dict(s_["lora"], strict=False).unexpected_keys[:3], flush=True)
if os.path.exists(ck):
    s_ = torch.load(ck, map_location=dev)
    adapter.load_state_dict(s_["adapter"])
    unet.load_state_dict(s_["lora"], strict=False)
    opt.load_state_dict(s_["opt"]); start = s_["step"]
    print("resumed at", start, flush=True)


def lora_state():
    return {k: v for k, v in unet.state_dict().items() if "lora" in k}


GENS = [torch.Generator().manual_seed(a.seed * 1000 + i) for i in range(len(ROWS))]


def batch_paired():
    """Fixed rows per set, each set with its own random stream (indices, timesteps, noise, dropout), so runs that
    share a set (e.g. the real rows of arms A, B, C) see identical draws for it."""
    out = []
    for si, (t_, k) in enumerate(zip(train, ROWS)):
        g = GENS[si]
        idx = torch.randint(0, len(t_["L"]), (k,), generator=g)
        x0 = t_["L"][idx].float()
        tt = torch.randint(0, sched.config.num_train_timesteps, (k,), generator=g)
        eps = torch.randn(x0.shape, generator=g)
        drop = torch.rand(k, generator=g) < 0.1
        out.append((t_["T"][idx], x0, t_["C"][idx], t_["R"][idx], t_["S"][idx], tt, eps, drop))
    return [torch.cat([o[i] for o in out]).to(dev) for i in range(8)]


def batch(n):
    parts = []
    for si, t in enumerate(train):
        k = n - sum(len(x[0]) for x in parts) if si == len(train) - 1 else int(round(n * (1 - a.mix)))
        idx = torch.randint(0, len(t["L"]), (k,))
        parts.append((t["T"][idx], t["L"][idx], t["C"][idx], t["R"][idx], t["S"][idx]))
    return [torch.cat([x[i] for x in parts]).to(dev) for i in range(5)]


# ---------------- train ----------------
# Pause safely: `touch <out>/STOP` (checked every 50 steps) or scancel (SIGTERM) -> save a checkpoint and exit.
# Restart with the same --out to continue from that step (data/args may change, model shape may not).
import signal
_stop = {"flag": False}
signal.signal(signal.SIGTERM, lambda *_: _stop.__setitem__("flag", True))
signal.signal(signal.SIGUSR1, lambda *_: _stop.__setitem__("flag", True))


def save(step_done):
    torch.save({"adapter": adapter.state_dict(), "lora": lora_state(), "opt": opt.state_dict(), "step": step_done, "args": vars(a)}, ck + ".tmp")
    os.replace(ck + ".tmp", ck)


log = open(f"{a.out}/log.jsonl", "a"); t_start = time.time()
scaler = torch.amp.GradScaler(enabled=AMP == torch.float16)
unet.train(); adapter.train()
for step in range(start, 0 if a.eval_only else a.steps):
    if ROWS:
        tok, x0, cov, rays, sty, t, eps, drop = batch_paired()
        x0 = x0.float() * lat_scale
    else:
        tok, x0, cov, rays, sty = batch(a.bs)
        x0 = x0.float() * lat_scale
        t = torch.randint(0, sched.config.num_train_timesteps, (a.bs,), device=dev)
        eps = torch.randn_like(x0)
        drop = torch.rand(a.bs, device=dev) < 0.1
    ab = acp[t].view(-1, 1, 1, 1)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    lr = a.lr * min(1, (step + 1) / 500)
    for g in opt.param_groups:
        g["lr"] = lr
    with torch.autocast("cuda", dtype=AMP):
        ctx = adapter(tok.float(), rays.float(), sty.long(), drop)
        pred = unet(xt, t, encoder_hidden_states=ctx).sample
    w = a.hole_w + (1 - a.hole_w) * cov.float()[:, None]
    loss = ((pred.float() - eps) ** 2 * w).mean() / w.mean()
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update()
    if step % 100 == 0:
        r = {"step": step, "loss": round(loss.item(), 4), "gn": round(gn.item(), 3), "lr": lr, "min": round((time.time() - t_start) / 60, 1)}
        print(r, flush=True); log.write(json.dumps(r) + "\n"); log.flush()
    if (step + 1) % 1000 == 0 or step + 1 == a.steps:
        save(step + 1)
    if str(step + 1) in a.save_at.split(","):
        save(step + 1); import shutil; shutil.copy(ck, f"{a.out}/ckpt_{step + 1}.pt")
    if _stop["flag"] or (step % 50 == 0 and os.path.exists(f"{a.out}/STOP")):
        save(step + 1)
        print("PAUSED at", step + 1, flush=True)
        if os.path.exists(f"{a.out}/STOP"):
            os.remove(f"{a.out}/STOP")
        raise SystemExit(0)

# ---------------- render val scenes ----------------
unet.eval(); adapter.eval()
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler")
import lpips
lp = lpips.LPIPS(net="alex").to(dev).eval()


@torch.no_grad()
def render(tok, rays, style, seed=0):
    B = tok.shape[0]
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(B, 4, 32, 64, device=dev, generator=g)
    st = torch.full((B,), style, device=dev, dtype=torch.long)
    with torch.autocast("cuda", dtype=AMP):
        c = adapter(tok, rays, st)
        u = adapter(tok, rays, st, torch.ones(B, dtype=torch.bool, device=dev))
    ddim.set_timesteps(a.sample_steps)
    for t in ddim.timesteps:
        with torch.autocast("cuda", dtype=AMP):
            e = unet(torch.cat([x, x]), t, encoder_hidden_states=torch.cat([c, u])).sample.float()
        ec, eu = e.chunk(2)
        x = ddim.step(eu + a.cfg * (ec - eu), t, x).prev_sample
    return x


@torch.no_grad()
def decode(z):
    return torch.cat([vae.decode(z[i:i + 8] / 0.18215).sample.float() for i in range(0, len(z), 8)]).clamp(-1, 1)


if ROWS:  # render against the real key set of the requested style
    ev = next((s_ for s_ in sets if len(s_["S"]) and int(s_["S"][0]) == a.render_style and s_["tokdir"] == a.tokens), sets[0])
else:
    ev = sets[-1] if a.render_style == 1 and len(sets) > 1 else sets[0]
val_names = sorted({n for n in ev["names"] if n in VAL})[: a.render_scenes]
metrics = {}
for sc in val_names:
    ix = torch.tensor([i for i, n in enumerate(ev["names"]) if n == sc])
    tok, gt, cov, rays = ev["T"][ix].to(dev).float(), ev["L"][ix].to(dev).float(), ev["C"][ix].to(dev).float(), ev["R"][ix].to(dev).float()
    pred = torch.cat([render(tok[i:i + 8], rays[i:i + 8], a.render_style, seed=i) for i in range(0, len(tok), 8)])
    P, G = decode(pred), decode(gt)
    cm = F.interpolate(cov[:, None], scale_factor=8)[:, 0]
    l1 = float(((P - G).abs().mean(1) * cm).sum() / cm.sum() * 127.5)
    lpv = float(torch.cat([lp(P[i:i + 8] * cm[i:i + 8, None], G[i:i + 8] * cm[i:i + 8, None]).flatten() for i in range(0, len(P), 8)]).mean())
    metrics[sc] = {"masked_L1": l1, "LPIPS": lpv}
    print(sc, metrics[sc], flush=True)
    to8 = lambda x: ((x + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    fr = [Image.fromarray(np.concatenate([g_, p_], 1)) for g_, p_ in zip(to8(G), to8(P))]
    fr[0].save(f"{a.out}/{sc}.gif", save_all=True, append_images=fr[1:], duration=250, loop=0)
summ = {"masked_L1": float(np.mean([m["masked_L1"] for m in metrics.values()])), "LPIPS": float(np.mean([m["LPIPS"] for m in metrics.values()])), "scenes": len(metrics)}
json.dump({"summary": summ, "scenes": metrics}, open(f"{a.out}/eval.json", "w"), indent=1)
print("SUMMARY", json.dumps(summ))
print("SD_DONE")
