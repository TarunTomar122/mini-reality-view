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
p.add_argument("--drive_list", default="", help="file with drive names: only these (e.g. the long-bet own-rollout pool)")
p.add_argument("--swap", action="store_true", help="lag test: from tick 50 on, feed another drive's cameras")
p.add_argument("--max_ticks", type=int, default=0)
p.add_argument("--shard", default="0/1")
p.add_argument("--batch", type=int, default=12, help="(drive, seed) streams denoised together in lockstep; per-stream noise is unchanged")
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


HAS_HIST = "hist" in ck
if a.method == "hist":
    assert HAS_HIST, "history method needs a checkpoint with a history conv (arm B / C)"
if HAS_HIST:  # B / C: every method (incl. cold / independent) runs through the trained history conv; no history = zeros + plane 1
    unet.conv_in = HistConvIn(unet.conv_in); unet.conv_in.hist.load_state_dict(ck["hist"])
unet.to(dev, AMP).eval()
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); ddim.set_timesteps(25)
TS = ddim.timesteps; acp = ddim.alphas_cumprod.to(dev)
RAYS = torch.load(a.rays)
drives = json.load(open(a.lists))[a.split]
if a.drive_list:
    want_ = open(a.drive_list).read().split()
    drives = [d_ for d_ in drives if d_ in set(want_)]
st = torch.full((1,), 2, dtype=torch.long, device=dev)
NOHIST = torch.cat([torch.zeros(1, 4, 32, 64), torch.ones(1, 1, 32, 64)], 1).to(dev)


def denoise(z, ts, c, u, h=None):
    """z, c, u: batch of B streams; h: (B, 5, 32, 64) history planes (NOHIST rows for streams without history) or None"""
    for t in ts:
        if HAS_HIST:
            hh = h if h is not None else NOHIST.expand(len(z), -1, -1, -1)
            unet.conv_in.h = torch.cat([hh, hh])
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
            e = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([c, u])).sample.float()
        ec, eu = e.chunk(2); z = ddim.step(eu + 2.0 * (ec - eu), t, z).prev_sample
    return z


si, sn = map(int, a.shard.split("/"))


class Stream:
    """one (drive, seed) rollout; advanced tick by tick in lockstep with the other streams of its batch"""
    def __init__(s, di, sc, seed, E, E2, other):
        s.sc, s.seed, s.E, s.E2, s.other = sc, seed, E, E2, other
        s.n = len(E["ticks"]) if not a.max_ticks else min(a.max_ticks, len(E["ticks"]))
        g = torch.Generator().manual_seed(zlib.crc32(f"{sc}|{seed}".encode()))  # one sampling-noise tensor per (drive, seed)
        s.z0 = torch.randn(1, 4, 32, 64, generator=g).to(dev)
        s.gh = torch.Generator().manual_seed(zlib.crc32(f"hist|{sc}|{seed}".encode()))  # separate history-noise stream
        s.out, s.restart, s.prev, s.oks, s.hards, s.srct, s.i, s.done = [], [], None, [], [], [], 0, False

    def next_job(s):
        """skip hard ticks; return (src, j, sc_r) of the next tick needing the model, or None when the drive is finished"""
        while s.i < s.n:
            i = s.i
            if s.E2 is not None and i >= 50 and i >= len(s.E2["ticks"]):
                break  # donor drive ended: stop (never repeat its last bundle)
            src, j, sc_r = (s.E2, i, s.other) if (s.E2 is not None and i >= 50) else (s.E, i, s.sc)
            s.oks.append(bool(src["ok"][j])); s.hards.append(bool(src["hard"][j])); s.srct.append(int(src["ticks"][j]))
            if bool(src["hard"][j]):
                s.out.append(torch.zeros(4, 32, 64, dtype=torch.float16)); s.restart.append(True); s.prev = None; s.i += 1
                continue
            s.restart.append(s.prev is None)
            return src, j, sc_r
        s.done = True
        return None

    def save(s):
        fo = f"{a.out}/{s.sc}_s{s.seed}.pt"; n_out = len(s.out)
        torch.save({"ticks": s.E["ticks"][:n_out], "src_ticks": torch.tensor(s.srct), "lat": torch.stack(s.out), "ok": torch.tensor(s.oks), "hard": torch.tensor(s.hards),
                    "restart": torch.tensor(s.restart), "amp": str(AMP), "snapshot": a.ckpt, "snapshot_step": int(ck.get("step", -1)), "s": a.s, "method": a.method,
                    "swap_from": 50 if s.E2 is not None else -1, "other": s.other if s.E2 is not None else "", "batch": a.batch}, fo + ".tmp")
        os.replace(fo + ".tmp", fo)
        print(s.sc, "seed", s.seed, s.n, "ticks", flush=True)


todo = []
for di, sc in enumerate(drives):
    if di % sn != si or not os.path.exists(f"{a.enc}/{sc}.pt"):
        continue
    for seed in map(int, a.seeds.split(",")):
        if not os.path.exists(f"{a.out}/{sc}_s{seed}.pt"):
            todo.append((di, sc, seed))
print(f"{len(todo)} streams to render, batch {a.batch}", flush=True)
ENC = {}
for b0 in range(0, len(todo), a.batch):
    group = []
    for di, sc, seed in todo[b0:b0 + a.batch]:
        if sc not in ENC:
            ENC[sc] = torch.load(f"{a.enc}/{sc}.pt")
        other = drives[(di + len(drives) // 2) % len(drives)]
        E2 = torch.load(f"{a.enc}/{other}.pt") if a.swap and os.path.exists(f"{a.enc}/{other}.pt") else None
        group.append(Stream(di, sc, seed, ENC[sc], E2, other))
    while True:
        jobs = [(s_, s_.next_job()) for s_ in group if not s_.done]
        jobs = [(s_, jb) for s_, jb in jobs if jb is not None]
        for s_ in group:
            if s_.done and not getattr(s_, "saved", False):
                s_.save(); s_.saved = True
        if not jobs:
            break
        tok = torch.cat([src["tokens"][j:j + 1] for _, (src, j, _) in jobs]).to(dev).float()
        ry = torch.stack([RAYS[sc_r] for _, (_, _, sc_r) in jobs]).to(dev).float()
        B = len(jobs); stB = st.expand(B)
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
            C = adapter(tok, ry, stB); U = adapter(tok, ry, stB, torch.ones(B, dtype=torch.bool, device=dev))
        Z = [None] * B
        if a.method == "keyvae":  # VAE round trip of the key itself (no model)
            for k, (_, (src, j, _)) in enumerate(jobs):
                Z[k] = src["lat"][j:j + 1].to(dev).float()
        elif a.method == "fresh":  # new sampling noise at every tick (seeded per drive / seed / tick)
            zs = torch.cat([torch.randn(1, 4, 32, 64, generator=torch.Generator().manual_seed(zlib.crc32(f"fresh|{s_.sc}|{s_.seed}|{s_.i}".encode()))) for s_, _ in jobs]).to(dev)
            Zb = denoise(zs, TS, C, U); Z = list(Zb.split(1))
        elif a.method == "warm":  # warm streams re-run the last fraction s; cold (restart) streams run all steps: two groups
            k0 = int(round(len(TS) * (1 - a.s))); ab = acp[TS[k0]]
            for warm in (True, False):
                idx = [k for k, (s_, _) in enumerate(jobs) if (s_.prev is not None) == warm]
                if not idx:
                    continue
                if warm:
                    zi = torch.cat([ab.sqrt() * jobs[k][0].prev + (1 - ab).sqrt() * jobs[k][0].z0 for k in idx]); ts_ = TS[k0:]
                else:
                    zi = torch.cat([jobs[k][0].z0 for k in idx]); ts_ = TS
                zo = denoise(zi, ts_, C[idx], U[idx])
                for m, k in enumerate(idx):
                    Z[k] = zo[m:m + 1]
        else:  # independent, or hist (history from the previous output; NOHIST rows after a restart)
            H = None
            if a.method == "hist":
                hs = []
                for s_, _ in jobs:
                    if s_.prev is None:
                        hs.append(NOHIST)
                    else:
                        eps = torch.randn(1, 4, 32, 64, generator=s_.gh).to(dev)
                        hs.append(torch.cat([(1 - a.s ** 2) ** 0.5 * s_.prev + a.s * eps, torch.full((1, 1, 32, 64), a.s, device=dev)], 1))
                H = torch.cat(hs)
            Zb = denoise(torch.cat([s_.z0 for s_, _ in jobs]).clone(), TS, C, U, H); Z = list(Zb.split(1))
        for k, (s_, _) in enumerate(jobs):
            s_.prev = Z[k]; s_.out.append(Z[k][0].half().cpu()); s_.i += 1
    ENC = {k_: v_ for k_, v_ in ENC.items() if any(t_[1] == k_ for t_ in todo[b0 + a.batch:])}
print("ROLLOUT_DONE", a.method, a.s, a.split)
