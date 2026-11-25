"""Stages 4-5: train the renderer with flow matching, then render held-out scenes.

  noisy = (1 - t) * target + t * noise,  model predicts v = noise - target
--prev adds the previous frame's latent with a little noise (stage 5); 10% of the time it is pure noise,
so the same model can also start a video from nothing.
"""
import argparse, glob, json, os, time, copy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL
from mrv.model import Renderer

p = argparse.ArgumentParser()
p.add_argument("--enc", default="data/enc")
p.add_argument("--out", default="runs/stage4")
p.add_argument("--test_scenes", default="scene-0553,scene-1077")
p.add_argument("--prev", action="store_true")
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--d", type=int, default=384)
p.add_argument("--depth", type=int, default=12)
p.add_argument("--heads", type=int, default=6)
p.add_argument("--hole_w", type=float, default=0.05, help="loss weight where the lidar target has no data")
p.add_argument("--sample_steps", type=int, default=25)
p.add_argument("--eval_only", action="store_true")
p.add_argument("--enc2", default="", help="second encoded set (e.g. splat targets), tagged style 1")
p.add_argument("--mix", type=float, default=0.5, help="fraction of each batch drawn from --enc2")
p.add_argument("--init", default="", help="start from this checkpoint's EMA weights (fresh optimiser)")
p.add_argument("--render_style", type=int, default=1)
p.add_argument("--rays", default="", help="experiment A: data/calib_rays.pt -> per-patch ray embeddings")
p.add_argument("--render", default="", help="scenes to render (default: test scenes)")
p.add_argument("--max_render", type=int, default=6)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
scaler = torch.amp.GradScaler(enabled=AMP == torch.float16)
print("amp", AMP, torch.cuda.get_device_name(0))

# ---------- data: everything fits in GPU memory for mini; full set stays on CPU ----------
if a.test_scenes == "val":  # official nuScenes val split (150 scenes)
    from nuscenes.utils.splits import val as _val
    test = set(_val)
else:
    test = set(a.test_scenes.split(","))
data = {os.path.basename(f)[:-3]: torch.load(f) for f in sorted(glob.glob(f"{a.enc}/*.pt"))}
train_names = [n for n in data if n not in test]
test_names = [n for n in data if n in test]
print("train scenes", len(train_names), "test scenes", len(test_names), test_names[:6])


RAYS = torch.load(a.rays) if a.rays else None


def stack(names):
    T, L, C, P, has_prev, R = [], [], [], [], [], []
    for n in names:
        d = data[n]
        if RAYS is not None:
            R.append(RAYS[n.split(":")[-1]][None].expand(len(d["lat"]), -1, -1, -1))
        T.append(d["tokens"]); L.append(d["lat"]); C.append(d["cover"])
        P.append(torch.cat([d["lat"][:1], d["lat"][:-1]]))
        hp = torch.ones(len(d["lat"]), dtype=torch.bool); hp[0] = False; has_prev.append(hp)
    stack.rays = torch.cat(R) if R else None
    return torch.cat(T), torch.cat(L).float(), torch.cat(C).float(), torch.cat(P).float(), torch.cat(has_prev)


Ttr, Ltr, Ctr, Ptr, HPtr = stack(train_names)
Rtr = stack.rays.to(dev) if RAYS is not None else None
if a.enc2:
    data2 = {os.path.basename(f)[:-3]: torch.load(f) for f in sorted(glob.glob(f"{a.enc2}/*.pt"))}
    tr2 = [n for n in data2 if n not in test]
    te2 = [n for n in data2 if n in test]
    print("enc2 train scenes", len(tr2), "test scenes", te2)
    _keep = data
    data = data2
    T2, L2, C2, P2, HP2 = stack(tr2)
    R2 = stack.rays.to(dev) if RAYS is not None else None
    data = {**_keep, **{"splat:" + k: v for k, v in data2.items()}}
    T2, L2, C2, P2, HP2 = T2.to(dev), L2.to(dev), C2.to(dev), P2.to(dev), HP2.to(dev)
N = len(Ltr)
lat_std = Ltr.std().item()
print("train samples", N, "latent std", round(lat_std, 3))
on_gpu = Ttr.numel() * 2 < 8e9
if on_gpu:
    Ttr = Ttr.to(dev)
Ltr, Ctr, Ptr, HPtr = Ltr.to(dev), Ctr.to(dev), Ptr.to(dev), HPtr.to(dev)

model = Renderer(d=a.d, depth=a.depth, heads=a.heads, prev=a.prev, n_styles=2 if a.enc2 else 0, rays=bool(a.rays)).to(dev)
if a.init and not os.path.exists(f"{a.out}/ckpt.pt"):
    _s = torch.load(a.init, map_location=dev)
    print("init from", a.init, model.load_state_dict(_s["ema"], strict=False))
ema = copy.deepcopy(model).eval().requires_grad_(False)
print("params (M)", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
ck = f"{a.out}/ckpt.pt"
start = 0
if os.path.exists(ck):
    s = torch.load(ck, map_location=dev)
    model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); start = s["step"]
    print("resumed at", start)
    lat_std = s.get("lat_std", lat_std)


def prev_input(prev, has_prev, train=True):
    B = prev.shape[0]
    s = torch.rand(B, device=dev) * 0.3 if train else torch.full((B,), 0.1, device=dev)
    s = torch.where(has_prev, s, torch.ones_like(s))
    if train:
        s = torch.where(torch.rand(B, device=dev) < 0.1, torch.ones_like(s), s)
    eps = torch.randn_like(prev)
    return (1 - s.view(B, 1, 1, 1)) * prev / lat_std + s.view(B, 1, 1, 1) * eps, s


log = open(f"{a.out}/log.jsonl", "a")
t0 = time.time()
model.train()
for step in range(start, 0 if a.eval_only else a.steps):
    b1 = a.bs - (int(a.bs * a.mix) if a.enc2 else 0)
    idx = torch.randint(0, N, (b1,), device=dev if on_gpu else "cpu")
    tok = Ttr[idx].to(dev, non_blocking=True).float()
    idx = idx.to(dev)
    x0, cv, pv, hp = Ltr[idx], Ctr[idx], Ptr[idx], HPtr[idx]
    sty = torch.zeros(b1, dtype=torch.long, device=dev)
    ry = Rtr[idx] if RAYS is not None else None
    if a.enc2:
        j = torch.randint(0, len(L2), (a.bs - b1,), device=dev)
        tok = torch.cat([tok, T2[j].float()]); x0 = torch.cat([x0, L2[j]]); cv = torch.cat([cv, C2[j]])
        if RAYS is not None:
            ry = torch.cat([ry, R2[j]])
        pv = torch.cat([pv, P2[j]]); hp = torch.cat([hp, HP2[j]]); sty = torch.cat([sty, torch.ones(a.bs - b1, dtype=torch.long, device=dev)])
    x0 = x0 / lat_std
    w = a.hole_w + (1 - a.hole_w) * cv[:, None]
    t = torch.sigmoid(torch.randn(a.bs, device=dev) * 1.0)  # logit-normal timesteps
    eps = torch.randn_like(x0)
    xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * eps
    kw = {}
    if a.prev:
        kw["prev"], kw["s_noise"] = prev_input(pv, hp)
    if a.enc2:
        kw["style"] = sty
    if RAYS is not None:
        kw["rays"] = ry
    lr = a.lr * min(1, (step + 1) / 1000)
    for g in opt.param_groups:
        g["lr"] = lr
    with torch.autocast("cuda", dtype=AMP):
        v = model(xt, t, tok, **kw)
    loss = ((v.float() - (eps - x0)) ** 2 * w).mean() / w.mean()
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - 0.999)
    if step % 200 == 0:
        r = {"step": step, "loss": round(loss.item(), 4), "gn": round(gn.item(), 3), "lr": lr, "min": round((time.time() - t0) / 60, 1)}
        print(r, flush=True); log.write(json.dumps(r) + "\n"); log.flush()
    if (step + 1) % 5000 == 0 or step + 1 == a.steps:
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step + 1, "lat_std": lat_std, "args": vars(a)}, ck)

# ---------- render held-out scenes ----------
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()


@torch.no_grad()
def sample(tok, prev=None, s_noise=None, steps=a.sample_steps, seed=0, style=None, rays=None):
    g = torch.Generator(device=dev).manual_seed(seed)
    B = tok.shape[0]
    x = torch.randn(B, 4, 18, 32, device=dev, generator=g)
    ctx = ema.context(tok, rays)
    ts = torch.linspace(1, 0, steps + 1, device=dev)
    for i in range(steps):
        t = ts[i].expand(B)
        with torch.autocast("cuda", dtype=AMP):
            v = ema(x, t, None, prev=prev, s_noise=s_noise, ctx=ctx, style=style).float()
        x = x + (ts[i + 1] - ts[i]) * v
    return x * lat_std


@torch.no_grad()
def decode(z):
    img = vae.decode(z / 0.18215).sample
    return ((img.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


metrics = {}
render_names = a.render.split(",") if a.render else test_names[:a.max_render]
for n in render_names:
    d = data.get("splat:" + n, data[n]) if a.enc2 and a.render_style == 1 else data[n]
    rs = {"style": torch.full((1,), a.render_style, device=dev, dtype=torch.long)} if a.enc2 else {}
    tok = d["tokens"].to(dev).float()
    if RAYS is not None:
        rs["rays"] = RAYS[n][None].to(dev)
    gt = d["lat"].to(dev).float()
    cov = F.interpolate(d["cover"][:, None].float(), scale_factor=8, mode="nearest")[:, 0].numpy()
    if a.prev:  # roll out the whole scene: each output feeds the next step
        outs, prev = [], None
        for i in range(len(tok)):
            if prev is None:
                pin, s = torch.randn(1, 4, 18, 32, device=dev), torch.ones(1, device=dev)
            else:
                s = torch.full((1,), 0.1, device=dev)
                pin = 0.9 * prev / lat_std + 0.1 * torch.randn_like(prev)
            z = sample(tok[i:i + 1], pin, s, seed=i, **rs)
            outs.append(z); prev = z
        pred = torch.cat(outs)
    else:
        bkw = lambda n_: {k: v.expand(n_, *v.shape[1:]) for k, v in rs.items()}
        pred = torch.cat([sample(tok[i:i + 8], seed=i, **bkw(len(tok[i:i + 8]))) for i in range(0, len(tok), 8)])
    P, G = decode(pred), decode(gt)
    err = np.abs(P.astype(float) - G.astype(float)).mean(-1)
    metrics[n] = {"masked_L1": float((err * cov).sum() / cov.sum()), "frames": len(P)}
    os.makedirs(f"{a.out}/{n}", exist_ok=True)
    for i in range(len(P)):
        Image.fromarray(np.concatenate([G[i], P[i]], 1)).save(f"{a.out}/{n}/{i:03d}.png")
    frames = [Image.fromarray(np.concatenate([G[i], P[i]], 1)) for i in range(len(P))]
    frames[0].save(f"{a.out}/{n}.gif", save_all=True, append_images=frames[1:], duration=250, loop=0)
print("METRICS", json.dumps(metrics))
json.dump(metrics, open(f"{a.out}/metrics{'_' + a.render if a.render else ''}.json", "w"), indent=1)
print("TRAIN_DONE")
