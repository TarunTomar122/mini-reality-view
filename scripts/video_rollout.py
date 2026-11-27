"""Render whole 10 Hz drives with the joint multi-frame model V (council/video-next-fable.md). Output format = the signed
temporal_rollout.py format ({ticks, src_ticks, lat, ok, hard, restart, amp}) so temporal_metrics.py (764dd3ec95db) scores it.

Per contiguous segment (broken only at hard ticks, as temporal_rollout.py): every tick starts from noise
eps_i = (z_drive + z_i) / sqrt(2) (z_drive = the signed per-(drive, seed) tensor, z_i from a seeded per-tick stream);
at each DDIM step the frames are denoised in windows of --frames with stride --stride, predictions averaged where windows
overlap (CFG 2, same camera tokens per frame). Not causal (a frame sees up to --frames - 1 ticks ahead).
--off runs the same sampler with the temporal blocks switched off (control: the mixed-noise change alone).

  python scripts/video_rollout.py --ckpt runs/vidV/ckpt_4000.pt --split dev --out runs/troll/V4k_dev [--off]
"""
import argparse, json, os, zlib
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.video import add_temporal, set_frames

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--split", default="dev", choices=["dev", "decision", "final", "train"])
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--seeds", default="0,1")
p.add_argument("--frames", type=int, default=16)
p.add_argument("--stride", type=int, default=8)
p.add_argument("--off", action="store_true")
p.add_argument("--shard", default="0/1")
p.add_argument("--out", required=True)
a = p.parse_args()
dev = "cuda"
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
SD = "sd-legacy/stable-diffusion-v1-5"
os.makedirs(a.out, exist_ok=True)
ck = torch.load(a.ckpt, map_location="cpu"); r = ck["args"]["rank"]
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); unet.load_state_dict(ck["lora"], strict=False)
unet.to(dev, AMP).eval()
TW = add_temporal(unet)
if "temporal" in ck:
    for w, sd_ in zip(TW, ck["temporal"]):
        w.temporal.load_state_dict(sd_)
for w in TW:
    w.temporal.to(dev, AMP).eval()
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); ddim.set_timesteps(25); TS = ddim.timesteps
RAYS = torch.load(a.rays)
drives = json.load(open(a.lists))[a.split]
st1 = torch.full((1,), 2, dtype=torch.long, device=dev)


def windows(n, F, S):
    if n <= F:
        return [(0, n)]
    w = list(range(0, n - F + 1, S))
    if w[-1] != n - F:
        w.append(n - F)
    return [(s, s + F) for s in w]


si, sn = map(int, a.shard.split("/"))
for di, sc in enumerate(drives):
    if di % sn != si or not os.path.exists(f"{a.enc}/{sc}.pt"):
        continue
    E = torch.load(f"{a.enc}/{sc}.pt"); n = len(E["ticks"])
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        C = torch.cat([adapter(E["tokens"][i:i + 1].to(dev).float(), RAYS[sc][None].to(dev).float(), st1) for i in range(n)])
        U = adapter(E["tokens"][0:1].to(dev).float(), RAYS[sc][None].to(dev).float(), st1, torch.ones(1, dtype=torch.bool, device=dev))
    hard = [bool(h) for h in E["hard"].tolist()]
    segs, cur = [], []
    for i in range(n):
        if hard[i]:
            if cur: segs.append(cur); cur = []
        else:
            cur.append(i)
    if cur: segs.append(cur)
    for seed in map(int, a.seeds.split(",")):
        fo = f"{a.out}/{sc}_s{seed}.pt"
        if os.path.exists(fo):
            continue
        g = torch.Generator().manual_seed(zlib.crc32(f"{sc}|{seed}".encode()))
        z0 = torch.randn(1, 4, 32, 64, generator=g)
        gp = torch.Generator().manual_seed(zlib.crc32(f"video-private|{sc}|{seed}".encode()))
        zp = torch.randn(n, 4, 32, 64, generator=gp)
        out = torch.zeros(n, 4, 32, 64, dtype=torch.float16); restart = [False] * n
        for seg in segs:
            restart[seg[0]] = True
            x = ((z0 + zp[seg]) / 2 ** 0.5).to(dev)
            for t in TS:
                acc = torch.zeros_like(x); cnt = torch.zeros(len(seg), 1, 1, 1, device=dev)
                for lo, hi in windows(len(seg), a.frames, a.stride):
                    idx = seg[lo:hi]; m = hi - lo
                    set_frames(TW, None if a.off else m)
                    cc = torch.cat([C[idx], U.expand(m, -1, -1)])
                    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
                        # cond and uncond as two separate clips of m frames each (temporal attention within a clip)
                        e = unet(torch.cat([x[lo:hi], x[lo:hi]]), t, encoder_hidden_states=cc).sample.float()
                    ec, eu = e[:m], e[m:]
                    acc[lo:hi] += eu + 2.0 * (ec - eu); cnt[lo:hi] += 1
                x = ddim.step(acc / cnt, t, x).prev_sample
            out[seg] = x.half().cpu()
        torch.save({"ticks": E["ticks"], "src_ticks": E["ticks"], "lat": out, "ok": E["ok"], "hard": E["hard"],
                    "restart": torch.tensor(restart), "amp": str(AMP), "method": "video_off" if a.off else "video",
                    "ckpt": a.ckpt, "frames": a.frames, "stride": a.stride}, fo)
        print(sc, "seed", seed, n, "ticks", len(segs), "segments", flush=True)
print("VIDEO_ROLLOUT_DONE", a.ckpt, a.split)
