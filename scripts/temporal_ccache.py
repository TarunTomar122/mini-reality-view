"""Arm C own-rollout cache (council/PLAN-temporal.md): from a frozen snapshot with a history conv, roll the 128
preselected eight-tick train clips (data/temporal_cclips.json, made by train_temporal.py) with DDIM 25 / CFG 2,
history level s = 0.3; the first tick of a clip has no history. Detached (no grad). Writes <out>/<scene>.pt
{ticks, lat (n,4,32,64) fp16} with the own frame of every clip tick, for train_temporal.py --arm C --cache <out>.

  python scripts/temporal_ccache.py --ckpt runs/tmpB/ckpt_1000.pt --out data/rollcache_1000 [--shard 0/2]
"""
import argparse, json, os, zlib
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--clips", default="data/temporal_cclips.json")
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--s", type=float, default=0.3)
p.add_argument("--shard", default="0/1")
p.add_argument("--out", required=True)
a = p.parse_args()
dev = "cuda"
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
SD = "sd-legacy/stable-diffusion-v1-5"
os.makedirs(a.out, exist_ok=True)
ck = torch.load(a.ckpt, map_location="cpu"); assert "hist" in ck, "needs a history checkpoint (arm B / C)"
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=ck["args"]["rank"], lora_alpha=ck["args"]["rank"], target_modules=LORA_TARGETS))
unet.load_state_dict(ck["lora"], strict=False)


class HistConvIn(nn.Module):
    def __init__(s, base):
        super().__init__(); s.base = base; s.hist = nn.Conv2d(5, base.out_channels, 3, padding=1); s.h = None
    def forward(s, x):
        y = s.base(x)
        return y if s.h is None else y + s.hist(s.h.to(y.dtype))


unet.conv_in = HistConvIn(unet.conv_in); unet.conv_in.hist.load_state_dict(ck["hist"]); unet.to(dev, AMP).eval()
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); ddim.set_timesteps(25); TS = ddim.timesteps
RAYS = torch.load(a.rays); st = torch.full((1,), 2, dtype=torch.long, device=dev)
NOHIST = torch.cat([torch.zeros(1, 4, 32, 64), torch.ones(1, 1, 32, 64)], 1).to(dev)
clips = json.load(open(a.clips))["clips"]
by_sc = {}
for sc, i0, _ in clips:
    by_sc.setdefault(sc, []).append(i0)
si, sn = map(int, a.shard.split("/"))
for n_, sc in enumerate(sorted(by_sc)):
    if n_ % sn != si or os.path.exists(f"{a.out}/{sc}.pt"):
        continue
    E = torch.load(f"{a.enc}/{sc}.pt"); ticks, lats = [], []
    for i0 in by_sc[sc]:
        g = torch.Generator().manual_seed(zlib.crc32(f"ccache|{sc}|{i0}".encode()))
        z0 = torch.randn(1, 4, 32, 64, generator=g).to(dev); prev = None
        gh = torch.Generator().manual_seed(zlib.crc32(f"ccache-hist|{sc}|{i0}".encode()))  # separate history-noise stream
        for i in range(i0, i0 + 8):
            tok = E["tokens"][i:i + 1].to(dev).float(); ry = RAYS[sc][None].to(dev).float()
            with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
                c = adapter(tok, ry, st); u = adapter(tok, ry, st, torch.ones(1, dtype=torch.bool, device=dev))
            h = NOHIST if prev is None else torch.cat([(1 - a.s ** 2) ** 0.5 * prev + a.s * torch.randn(1, 4, 32, 64, generator=gh).to(dev),
                                                       torch.full((1, 1, 32, 64), a.s, device=dev)], 1)
            z = z0.clone()
            for t in TS:
                unet.conv_in.h = torch.cat([h, h])
                with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
                    e = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([c, u])).sample.float()
                ec, eu = e.chunk(2); z = ddim.step(eu + 2.0 * (ec - eu), t, z).prev_sample
            prev = z; ticks.append(int(E["ticks"][i])); lats.append(z[0].half().cpu())
    torch.save({"ticks": torch.tensor(ticks), "lat": torch.stack(lats)}, f"{a.out}/{sc}.pt.tmp"); os.replace(f"{a.out}/{sc}.pt.tmp", f"{a.out}/{sc}.pt")
    print(sc, len(ticks), "own frames", flush=True)
print("CCACHE_DONE", a.ckpt)
