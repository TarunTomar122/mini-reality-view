"""Stage 7 baseline: quality and speed vs number of sampling steps for a trained renderer.
Renders scenes with N Euler steps, reports masked L1 against the target and frames/second at batch 1 (incl. VAE decode)."""
import argparse, json, time
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from mrv.model import Renderer

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/stage4/ckpt.pt")
p.add_argument("--enc", default="data/enc_v2")
p.add_argument("--scenes", default="scene-0061,scene-1077")
p.add_argument("--steps", default="1,2,4,8,16,32")
p.add_argument("--out", default="runs/stage7/steps_baseline.json")
a = p.parse_args()
dev = "cuda"
ck = torch.load(a.ckpt, map_location=dev)
ar = ck["args"]
m = Renderer(d=ar["d"], depth=ar["depth"], heads=ar["heads"], prev=ar["prev"]).to(dev).eval()
m.load_state_dict(ck["ema"])
lat_std = ck["lat_std"]
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval().half()
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


@torch.no_grad()
def render(tok, n):
    x = torch.randn(tok.shape[0], 4, 18, 32, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    ctx = m.context(tok)
    ts = torch.linspace(1, 0, n + 1, device=dev)
    kw = {}
    if ar["prev"]:
        kw = {"prev": torch.randn_like(x), "s_noise": torch.ones(x.shape[0], device=dev)}
    for i in range(n):
        with torch.autocast("cuda", dtype=AMP):
            v = m(x, ts[i].expand(x.shape[0]), None, ctx=ctx, **kw).float()
        x = x + (ts[i + 1] - ts[i]) * v
    return vae.decode((x * lat_std / 0.18215).half()).sample.float()


res = {"gpu": torch.cuda.get_device_name(0)}
for sc in a.scenes.split(","):
    d = torch.load(f"{a.enc}/{sc}.pt")
    tok = d["tokens"].to(dev).float()
    with torch.no_grad():
        gt = torch.cat([vae.decode((d["lat"][i:i + 8].to(dev) / 0.18215).half()).sample.float() for i in range(0, len(d["lat"]), 8)])
    cov = F.interpolate(d["cover"][:, None].float().to(dev), scale_factor=8)
    for n in map(int, a.steps.split(",")):
        img = torch.cat([render(tok[i:i + 8], n) for i in range(0, len(tok), 8)])
        l1 = ((img - gt).abs().mean(1, keepdim=True) * cov).sum() / cov.sum() * 127.5
        torch.cuda.synchronize(); t0 = time.time()
        for i in range(10):
            render(tok[i:i + 1], n)
        torch.cuda.synchronize()
        fps = 10 / (time.time() - t0)
        res[f"{sc}/{n}"] = {"masked_L1": round(l1.item(), 2), "fps_b1": round(fps, 1)}
        print(sc, n, res[f"{sc}/{n}"], flush=True)
import os
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print("STEPS_DONE")
