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
from mrv.model import sincos_2d

SD = "sd-legacy/stable-diffusion-v1-5"
p = argparse.ArgumentParser()
p.add_argument("--tokens", default="data/enc_full", help="enc dir with pooled camera tokens per scene")
p.add_argument("--keys", default="data/lat512_full_low", help="style-0 key latents (encode_latents.py)")
p.add_argument("--keys2", default="", help="style-1 key latents (e.g. splat keys)")
p.add_argument("--mix", type=float, default=0.5)
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--rank", type=int, default=64)
p.add_argument("--steps", type=int, default=30000)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--hole_w", type=float, default=0.05)
p.add_argument("--out", required=True)
p.add_argument("--render_scenes", type=int, default=8)
p.add_argument("--sample_steps", type=int, default=25)
p.add_argument("--cfg", type=float, default=2.0)
p.add_argument("--render_style", type=int, default=1)
p.add_argument("--eval_only", action="store_true")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
print("amp", AMP, torch.cuda.get_device_name(0), flush=True)
from nuscenes.utils.splits import val
VAL = set(val)
RAYS = torch.load(a.rays)


# ---------------- data ----------------
def load_set(keydir, style):
    T, L, C, R, S, names = [], [], [], [], [], []
    for f in sorted(glob.glob(f"{keydir}/*.pt")):
        sc = os.path.basename(f)[:-3]
        k = torch.load(f)
        e = torch.load(f"{a.tokens}/{sc}.pt")
        pos = {t: i for i, t in enumerate(e["sample_tokens"])}
        ix = [pos[t] for t in k["sample_tokens"]]
        T.append(e["tokens"][ix]); L.append(k["lat"]); C.append(k["cover"])
        R.append(RAYS[sc][None].expand(len(ix), -1, -1, -1)); S += [style] * len(ix); names += [sc] * len(ix)
    return dict(T=torch.cat(T), L=torch.cat(L), C=torch.cat(C), R=torch.cat(R), S=torch.tensor(S), names=names)


def split(d):
    tr = torch.tensor([n not in VAL for n in d["names"]])
    return {k: (v[tr] if torch.is_tensor(v) else [x for x, m in zip(v, tr) if m]) for k, v in d.items()}


sets = [load_set(a.keys, 0)] + ([load_set(a.keys2, 1)] if a.keys2 else [])
train = [split(s) for s in sets]
for s, t in zip(sets, train):
    print("set", len(s["L"]), "frames,", len(t["L"]), "train", flush=True)
lat_scale = 1.0  # latents are already x0.18215 (SD convention)


# ---------------- model ----------------
class CameraAdapter(nn.Module):
    def __init__(s, d=768, n_cam=6, layers=2):
        super().__init__()
        s.inp = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, d))
        s.ray = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        s.cam = nn.Parameter(torch.zeros(1, n_cam, 1, d))
        s.register_buffer("pos", sincos_2d(9, 16, d)[None, None], persistent=False)
        s.style = nn.Embedding(2, d)
        s.null = nn.Parameter(torch.zeros(1, n_cam * 144 + 1, d))
        enc = nn.TransformerEncoderLayer(d, 12, 4 * d, batch_first=True, norm_first=True, activation="gelu")
        s.mix = nn.TransformerEncoder(enc, layers)
        s.out = nn.LayerNorm(d)

    def forward(s, tok, rays, style, drop=None):
        B = tok.shape[0]
        h = s.inp(tok) + s.ray(rays) + s.cam + s.pos  # (B,6,144,d)
        h = torch.cat([h.flatten(1, 2), s.style(style)[:, None]], 1)
        h = s.out(s.mix(h))
        if drop is not None:
            h = torch.where(drop[:, None, None], s.null.expand(B, -1, -1).to(h.dtype), h)
        return h


unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet", torch_dtype=torch.float32)
unet.requires_grad_(False)
unet.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights="gaussian",
                            target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]))
for n_, p_ in unet.named_parameters():
    if not p_.requires_grad:
        p_.data = p_.data.to(AMP)  # frozen base weights in half precision, LoRA weights stay fp32
unet.enable_gradient_checkpointing()
unet.to(dev)
adapter = CameraAdapter().to(dev)
params = [p_ for p_ in unet.parameters() if p_.requires_grad] + list(adapter.parameters())
print("trainable params (M)", round(sum(p_.numel() for p_ in params) / 1e6, 1), flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
sched = DDPMScheduler.from_pretrained(SD, subfolder="scheduler")
acp = sched.alphas_cumprod.to(dev)
ck = f"{a.out}/ckpt.pt"
start = 0
if os.path.exists(ck):
    s_ = torch.load(ck, map_location=dev)
    adapter.load_state_dict(s_["adapter"])
    unet.load_state_dict(s_["lora"], strict=False)
    opt.load_state_dict(s_["opt"]); start = s_["step"]
    print("resumed at", start, flush=True)


def lora_state():
    return {k: v for k, v in unet.state_dict().items() if "lora" in k}


def batch(n):
    parts = []
    for si, t in enumerate(train):
        k = n - sum(len(x[0]) for x in parts) if si == len(train) - 1 else int(round(n * (1 - a.mix)))
        idx = torch.randint(0, len(t["L"]), (k,))
        parts.append((t["T"][idx], t["L"][idx], t["C"][idx], t["R"][idx], t["S"][idx]))
    return [torch.cat([x[i] for x in parts]).to(dev) for i in range(5)]


# ---------------- train ----------------
log = open(f"{a.out}/log.jsonl", "a"); t_start = time.time()
scaler = torch.amp.GradScaler(enabled=AMP == torch.float16)
unet.train(); adapter.train()
for step in range(start, 0 if a.eval_only else a.steps):
    tok, x0, cov, rays, sty = batch(a.bs)
    x0 = x0.float() * lat_scale
    t = torch.randint(0, sched.config.num_train_timesteps, (a.bs,), device=dev)
    eps = torch.randn_like(x0)
    ab = acp[t].view(-1, 1, 1, 1)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    drop = torch.rand(a.bs, device=dev) < 0.1
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
    if (step + 1) % 5000 == 0 or step + 1 == a.steps:
        torch.save({"adapter": adapter.state_dict(), "lora": lora_state(), "opt": opt.state_dict(), "step": step + 1, "args": vars(a)}, ck)

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
