"""Stage 8: held-out evaluation of a trained renderer on official val scenes.

For each scene: render every keyframe (video models roll out the whole 20 s scene, each output feeding the next),
then compare with the target where the target has data:
  masked L1 (0-255), LPIPS (AlexNet, lower = more alike), and error vs time (drift).
Baseline: the average training target ("mean image"), to show what "knowing nothing about the cameras" scores.
"""
import argparse, glob, json, os
import numpy as np
import torch
import torch.nn.functional as F
import lpips
from diffusers import AutoencoderKL
from PIL import Image
from mrv.model import Renderer

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--enc", default="data/enc_full")
p.add_argument("--n_scenes", type=int, default=40)
p.add_argument("--steps", type=int, default=4)
p.add_argument("--s_prev", type=float, default=0.1)
p.add_argument("--out", required=True)
p.add_argument("--save_gifs", type=int, default=8)
p.add_argument("--truth_enc", default="", help="take target latents/coverage from here (e.g. splat targets); tokens still from --enc")
p.add_argument("--style", type=int, default=-1)
a = p.parse_args()
dev = "cuda"
os.makedirs(a.out, exist_ok=True)
ck = torch.load(a.ckpt, map_location=dev)
ar, lat_std = ck["args"], ck["lat_std"]
m = Renderer(d=ar["d"], depth=ar["depth"], heads=ar["heads"], prev=ar["prev"], n_styles=2 if ar.get("enc2") else 0).to(dev).eval()
m.load_state_dict(ck["ema"])
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval().half()
lp = lpips.LPIPS(net="alex").to(dev).eval()
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
from nuscenes.utils.splits import val
names = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{a.truth_enc or a.enc}/*.pt"))
val_names = [n for n in names if n in set(val)][: a.n_scenes]
train_names = [n for n in names if n not in set(val)]

# mean-image baseline from 50 training scenes
mean_lat = torch.stack([torch.load(f"{a.enc}/{n}.pt")["lat"].float().mean(0) for n in train_names[:50]]).mean(0, keepdim=True).to(dev)


@torch.no_grad()
def decode(z):
    return torch.cat([vae.decode((z[i:i + 8] / 0.18215).half()).sample.float() for i in range(0, len(z), 8)]).clamp(-1, 1)


@torch.no_grad()
def sample(tok, prev=None, s=None, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(tok.shape[0], 4, 18, 32, device=dev, generator=g)
    ctx = m.context(tok)
    ts = torch.linspace(1, 0, a.steps + 1, device=dev)
    for i in range(a.steps):
        with torch.autocast("cuda", dtype=AMP):
            st = torch.full((x.shape[0],), a.style, device=dev, dtype=torch.long) if a.style >= 0 else None
            v = m(x, ts[i].expand(x.shape[0]), None, prev=prev, s_noise=s, ctx=ctx, style=st).float()
        x = x + (ts[i + 1] - ts[i]) * v
    return x * lat_std


def scores(P, G, cov):
    l1 = ((P - G).abs().mean(1) * cov).flatten(1).sum(1) / cov.flatten(1).sum(1).clamp(min=1) * 127.5
    with torch.no_grad():
        lpv = torch.cat([lp(P[i:i + 16] * cov[i:i + 16, None], G[i:i + 16] * cov[i:i + 16, None]).flatten() for i in range(0, len(P), 16)])
    return l1.cpu().numpy(), lpv.cpu().numpy()


res, per_t_l1, per_t_lp, base_l1, base_lp = {}, [], [], [], []
for si, n in enumerate(val_names):
    d = torch.load(f"{a.enc}/{n}.pt")
    if a.truth_enc:
        dt = torch.load(f"{a.truth_enc}/{n}.pt")
        d = {"tokens": d["tokens"], "lat": dt["lat"], "cover": dt["cover"]}
    tok, gt = d["tokens"].to(dev).float(), d["lat"].to(dev).float()
    cov = F.interpolate(d["cover"][:, None].float().to(dev), scale_factor=8)[:, 0]
    if ar["prev"]:
        outs, prev = [], None
        for i in range(len(tok)):
            if prev is None:
                pin, s = torch.randn(1, 4, 18, 32, device=dev), torch.ones(1, device=dev)
            else:
                s = torch.full((1,), a.s_prev, device=dev)
                pin = (1 - a.s_prev) * prev / lat_std + a.s_prev * torch.randn_like(prev)
            z = sample(tok[i:i + 1], pin, s, seed=i); outs.append(z); prev = z
        pred = torch.cat(outs)
    else:
        pred = torch.cat([sample(tok[i:i + 8], seed=i) for i in range(0, len(tok), 8)])
    P, G = decode(pred), decode(gt)
    l1, lpv = scores(P, G, cov)
    bl1, blp = scores(decode(mean_lat.expand(len(gt), -1, -1, -1)), G, cov)
    res[n] = {"masked_L1": float(l1.mean()), "LPIPS": float(lpv.mean()), "base_L1": float(bl1.mean()), "base_LPIPS": float(blp.mean())}
    per_t_l1.append(l1[:40]); per_t_lp.append(lpv[:40]); base_l1.append(bl1.mean()); base_lp.append(blp.mean())
    print(n, res[n], flush=True)
    if si < a.save_gifs:
        to8 = lambda x: ((x + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        Gi, Pi = to8(G), to8(P)
        fr = [Image.fromarray(np.concatenate([Gi[i], Pi[i]], 1)) for i in range(len(Pi))]
        fr[0].save(f"{a.out}/{n}.gif", save_all=True, append_images=fr[1:], duration=250, loop=0)
L = min(len(x) for x in per_t_l1)
summary = {
    "ckpt": a.ckpt, "steps": a.steps, "video": bool(ar["prev"]), "scenes": len(val_names),
    "masked_L1": float(np.mean([r["masked_L1"] for r in res.values()])),
    "LPIPS": float(np.mean([r["LPIPS"] for r in res.values()])),
    "mean_image_L1": float(np.mean(base_l1)), "mean_image_LPIPS": float(np.mean(base_lp)),
    "L1_vs_time_first5_last5": [float(np.mean([x[:5].mean() for x in per_t_l1])), float(np.mean([x[L - 5:L].mean() for x in per_t_l1]))],
    "L1_per_frame": [float(np.mean([x[t] for x in per_t_l1])) for t in range(L)],
}
json.dump({"summary": summary, "scenes": res}, open(f"{a.out}/eval.json", "w"), indent=1)
print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "L1_per_frame"}))
print("EVAL_DONE")
