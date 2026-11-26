"""Temporal evaluation, step 1 (council/PLAN-temporal.md + amendments 1-2): roll a method through whole 10 Hz drives.

  python scripts/temporal_rollout.py --ckpt runs/tpvE3/ckpt_10000.pt --method independent --split dev --out runs/troll/indep_dev
  python scripts/temporal_rollout.py --ckpt ... --method warm --s 0.7 --split dev --out runs/troll/warm07_dev
  python scripts/temporal_rollout.py --ckpt runs/tmpB/ckpt_6000.pt --method hist --s 0.3 --split dev --out runs/troll/B_dev
  add --swap to switch to another drive's cameras at tick 50 (5 s) for the lag test.

Noise control: ONE sampling-noise tensor per (drive, seed), reused at every tick, by every method and by cold re-renders;
history corruption uses a separate seeded stream. Rollouts start at the first usable tick, run through ticks rejected
only for offset (their nearest bundle is fed; metrics skip them later) and restart cold after a hard break (camera
> 100 ms off or a repeated bundle). DDIM 25, CFG 2, style 2; for warm start a fraction s of the 25 steps is re-run from
the previous output; for history models the previous output (corrupted at level s) feeds the history conv (both CFG
passes get the same history). Writes <out>/<scene>_s<seed>.pt {ticks, lat (n,4,32,64) fp16, ok, hard, restart}.
"""
import argparse, json, os, zlib
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--method", required=True, choices=["independent", "warm", "hist", "fresh", "keyvae"],
               help="fresh = new noise every tick (metric check, worst case); keyvae = the keys' own VAE latents (metric check, best case)")
p.add_argument("--s", type=float, default=0.7)
p.add_argument("--split", default="dev", choices=["dev", "decision", "final", "train"])
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--seeds", default="0,1")
p.add_argument("--swap", action="store_true", help="lag test: from tick 50 on, feed another drive's cameras")
p.add_argument("--max_ticks", type=int, default=0)
p.add_argument("--shard", default="0/1")
p.add_argument("--out", required=True)
a = p.parse_args()
dev = "cuda"
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
SD = "sd-legacy/stable-diffusion-v1-5"
os.makedirs(a.out, exist_ok=True)
ck = torch.load(a.ckpt, map_location="cpu")
r = ck["args"]["rank"]
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS))
unet.load_state_dict(ck["lora"], strict=False)


class HistConvIn(nn.Module):
    def __init__(s, base):
        super().__init__(); s.base = base; s.hist = nn.Conv2d(5, base.out_channels, 3, padding=1); s.h = None
    def forward(s, x):
        y = s.base(x)
        return y if s.h is None else y + s.hist(s.h.to(y.dtype))


if a.method == "hist":
    assert "hist" in ck, "history method needs a checkpoint with a history conv (arm B / C)"
    unet.conv_in = HistConvIn(unet.conv_in); unet.conv_in.hist.load_state_dict(ck["hist"])
unet.to(dev, AMP).eval()
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); ddim.set_timesteps(25)
TS = ddim.timesteps; acp = ddim.alphas_cumprod.to(dev)
RAYS = torch.load(a.rays)
drives = json.load(open(a.lists))[a.split]
st = torch.full((1,), 2, dtype=torch.long, device=dev)
NOHIST = torch.cat([torch.zeros(1, 4, 32, 64), torch.ones(1, 1, 32, 64)], 1).to(dev)


def denoise(z, ts, c, u, h=None):
    for t in ts:
        if a.method == "hist":
            unet.conv_in.h = torch.cat([h, h]) if h is not None else torch.cat([NOHIST, NOHIST])
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
            e = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([c, u])).sample.float()
        ec, eu = e.chunk(2); z = ddim.step(eu + 2.0 * (ec - eu), t, z).prev_sample
    return z


si, sn = map(int, a.shard.split("/"))
for di, sc in enumerate(drives):
    if di % sn != si or not os.path.exists(f"{a.enc}/{sc}.pt"):
        continue
    E = torch.load(f"{a.enc}/{sc}.pt")
    other = drives[(di + len(drives) // 2) % len(drives)]
    E2 = torch.load(f"{a.enc}/{other}.pt") if a.swap and os.path.exists(f"{a.enc}/{other}.pt") else None
    n = len(E["ticks"]) if not a.max_ticks else min(a.max_ticks, len(E["ticks"]))
    for seed in map(int, a.seeds.split(",")):
        fo = f"{a.out}/{sc}_s{seed}.pt"
        if os.path.exists(fo):
            continue
        g = torch.Generator().manual_seed(zlib.crc32(f"{sc}|{seed}".encode()))  # one sampling-noise tensor per (drive, seed)
        z0 = torch.randn(1, 4, 32, 64, generator=g).to(dev)
        gh = torch.Generator().manual_seed(zlib.crc32(f"hist|{sc}|{seed}".encode()))  # separate history-noise stream
        out, restart, prev = [], [], None
        for i in range(n):
            hard = bool(E["hard"][i])
            if hard:
                out.append(torch.zeros(4, 32, 64, dtype=torch.float16)); restart.append(True); prev = None
                continue
            src, j, sc_r = (E2, min(i, len(E2["ticks"]) - 1), other) if (E2 is not None and i >= 50) else (E, i, sc)
            if E2 is not None and i >= 50 and bool(src["hard"][j]):
                out.append(torch.zeros(4, 32, 64, dtype=torch.float16)); restart.append(True); prev = None
                continue
            tok = src["tokens"][j:j + 1].to(dev).float(); ry = RAYS[sc_r][None].to(dev).float()
            with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
                c = adapter(tok, ry, st); u = adapter(tok, ry, st, torch.ones(1, dtype=torch.bool, device=dev))
            restart.append(prev is None)
            if a.method == "keyvae":  # VAE round trip of the key itself (no model)
                z = src["lat"][j:j + 1].to(dev).float()
            elif a.method == "fresh":  # new sampling noise at every tick (seeded per drive / seed / tick)
                gf = torch.Generator().manual_seed(zlib.crc32(f"fresh|{sc}|{seed}|{i}".encode()))
                z = denoise(torch.randn(1, 4, 32, 64, generator=gf).to(dev), TS, c, u)
            elif a.method == "warm" and prev is not None:
                k0 = int(round(len(TS) * (1 - a.s))); ab = acp[TS[k0]]
                z = denoise(ab.sqrt() * prev + (1 - ab).sqrt() * z0, TS[k0:], c, u)
            elif a.method == "hist" and prev is not None:
                eps = torch.randn(1, 4, 32, 64, generator=gh).to(dev)
                h = torch.cat([(1 - a.s ** 2) ** 0.5 * prev + a.s * eps, torch.full((1, 1, 32, 64), a.s, device=dev)], 1)
                z = denoise(z0.clone(), TS, c, u, h)
            else:
                z = denoise(z0.clone(), TS, c, u)
            prev = z
            out.append(z[0].half().cpu())
        torch.save({"ticks": E["ticks"][:n], "lat": torch.stack(out), "ok": E["ok"][:n], "hard": E["hard"][:n],
                    "restart": torch.tensor(restart), "swap_from": 50 if E2 is not None else -1, "other": other if E2 is not None else ""}, fo)
        print(sc, "seed", seed, n, "ticks", flush=True)
print("ROLLOUT_DONE", a.method, a.s, a.split)
