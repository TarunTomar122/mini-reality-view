"""Car map R (council/PLAN-carmap.md v4, round 2-3 of the car map council): input-level map conv on frozen E3.

  python scripts/train_carmap2.py --init runs/tpvE3/ckpt_10000.pt --boxes 'tmp/bevF_boxes_trainD.json' --out runs/cmR

E3 frozen (LoRA + CameraAdapter). New: one bias-free, zero-initialised 3x3 conv from the nine spatial map planes
(mrv/carmap.py channels 0-8; the constant availability plane is NOT used, so empty neighbourhoods add exactly zero)
onto the 320 channels after conv_in, at every denoising step and in both CFG passes. Weights kept fp32.
Data: flat keyframe keys of the 144 train drives (data/lat512_tpv), DINO tokens (data/enc_full), calibrated rays;
boxes = fixed-calibration camera-only BEVFormer, degraded to val error rates (no extra box dropout).
Loss = 0.5 whole-image eps-MSE + 0.5 eps-MSE on key car cells (data/keycar_tpv: Mask2Former car pixels of the game key,
own car excluded, pooled to 32x64 fractions; batch-normalised = sum(w * err) / sum(w); a batch without car cells uses
the whole-image term only). Timesteps t = floor((1 - u^3) * 1000) (biased to high noise). Camera dropout 50% through
update --drop_switch, then --drop_lo. AdamW 1e-4, 100 warm-up, batch 16, style 2, 4k updates, checkpoints 500 / 1k /
2k / 4k. Draws seeded per step.
"""
import argparse, glob, json, os, time, zlib
import numpy as np
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.carmap import rasterize, pool, MapConvIn, N_MAP

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--boxes", required=True)
p.add_argument("--keys", default="data/lat512_tpv")
p.add_argument("--tokens", default="data/enc_full")
p.add_argument("--keycar", default="data/keycar_tpv")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--save_at", default="500,1000,2000,4000")
p.add_argument("--drop_hi", type=float, default=0.5)
p.add_argument("--drop_switch", type=int, default=1000)
p.add_argument("--drop_lo", type=float, default=0.25)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--budget_min", type=float, default=0, help="timing run: exit 3 if the projected 4k run (map prep + 4,000 updates + checkpoints) exceeds this many minutes")
p.add_argument("--resume", action="store_true", help="continue <out>/ckpt.pt (staged 1k / 2k gates)")
p.add_argument("--out", required=True)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev, AMP = "cuda", torch.bfloat16
SD = "sd-legacy/stable-diffusion-v1-5"
TRAIN = set(json.load(open(a.lists))["train"])
RAYS = torch.load(a.rays)
BOX = {}
for f in [x for pat in a.boxes.split(",") for x in glob.glob(pat)]:  # comma-separated globs
    BOX.update(json.load(open(f)))
def map_planes(frame):
    return pool(rasterize(frame))[:N_MAP]


if __name__ == "__main__":
    rows, miss = [], 0
    for f in sorted(glob.glob(f"{a.keys}/*.pt")):
        sc = os.path.basename(f)[:-3]
        if sc not in TRAIN or not os.path.exists(f"{a.tokens}/{sc}.pt"):
            continue
        k = torch.load(f); e = torch.load(f"{a.tokens}/{sc}.pt"); kc = torch.load(f"{a.keycar}/{sc}.pt")
        pos = {t: i for i, t in enumerate(e["sample_tokens"])}; kpos = {t: i for i, t in enumerate(kc["sample_tokens"])}
        for j, t in enumerate(k["sample_tokens"]):
            if t not in BOX:
                miss += 1; continue
            w = kc["mask"][kpos[t]].float()
            if torch.isnan(w).any():
                miss += 1; continue
            rows.append((sc, t, e["tokens"][pos[t]], k["lat"][j], w))
    drives_ok = {r[0] for r in rows}
    assert drives_ok == TRAIN, f"missing train drives: {sorted(TRAIN - drives_ok)[:5]} ({len(TRAIN - drives_ok)})"
    assert miss == 0, f"{miss} keyframes without boxes or key masks (coverage must be complete)"
    # maps are deterministic (fixed degradation, no box dropout): rasterise every training frame once (z-buffered faces
    # cost ~0.13 s / frame), in parallel, and cache them
    from concurrent.futures import ProcessPoolExecutor
    t_prep0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, len(os.sched_getaffinity(0)))) as ex:  # the job's CPUs, not the node's
        MAPS = dict(zip([r[1] for r in rows], ex.map(map_planes, [BOX[r[1]] for r in rows], chunksize=16)))
    T_PREP = time.time() - t_prep0
    print(f"rasterised {len(MAPS)} maps in {T_PREP / 60:.1f} min", flush=True)
    print(f"rows {len(rows)} from {len({r[0] for r in rows})} drives; skipped {miss} (no boxes / no key mask); "
          f"mean car-cell share {np.mean([float((r[4] > 0).float().mean()) for r in rows]):.3f}", flush=True)

    ck0 = torch.load(a.init, map_location="cpu"); r = ck0["args"]["rank"]
    unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
    unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); unet.load_state_dict(ck0["lora"], strict=False)
    unet.requires_grad_(False); unet.to(dev, AMP).eval()
    unet.conv_in = MapConvIn(unet.conv_in); unet.conv_in.map.to(dev).float().requires_grad_(True)
    adapter = CameraAdapter(n_styles=ck0["args"].get("n_styles", 3)); adapter.load_state_dict(ck0["adapter"])
    adapter.requires_grad_(False); adapter.to(dev).eval()
    params = list(unet.conv_in.map.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    sched = DDPMScheduler.from_pretrained(SD, subfolder="scheduler"); acp = sched.alphas_cumprod.to(dev)
    start = 0; ck = f"{a.out}/ckpt.pt"
    if os.path.exists(ck) and not a.resume:
        raise SystemExit(f"{ck} exists: pass --resume to continue it, or use a fresh --out")
    if os.path.exists(ck):
        s_ = torch.load(ck, map_location="cpu"); unet.conv_in.map.load_state_dict(s_["mapconv"]); opt.load_state_dict(s_["opt"])
        start = s_["step"]; print("resumed at", start, flush=True)

    def save(step, path):
        ts_ = time.time()
        d = {"mapconv": unet.conv_in.map.state_dict(), "opt": opt.state_dict(), "step": step, "init": a.init,
             "args": {**ck0["args"], "carmap2": vars(a)}, "adapter": ck0["adapter"], "lora": ck0["lora"]}
        torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)
        SAVE_T[0] += time.time() - ts_; SAVE_T[1] += 1

    style = torch.full((16,), 2, dtype=torch.long, device=dev); CLIP = [0, 0]
    log = open(f"{a.out}/log.jsonl", "a"); t0 = time.time(); SAVE = set(int(x) for x in a.save_at.split(","))
    t10 = time.time(); SAVE_T = [0.0, 0]
    for step in range(start, a.steps):
        if step - start == 10:
            torch.cuda.synchronize(); t10 = time.time()
        g = torch.Generator().manual_seed(a.seed * 100003 + step)
        idx = torch.randint(0, len(rows), (16,), generator=g)
        u = torch.rand(16, generator=g); t = ((1 - u ** 3) * 1000).floor().clamp(0, 999).long()
        eps = torch.randn(16, 4, 32, 64, generator=g)
        pdrop = a.drop_hi if step < a.drop_switch else a.drop_lo
        drop = torch.rand(16, generator=g) < pdrop
        M, tok, lat, ray, W = [], [], [], [], []
        for i in idx.tolist():
            sc, tk, tk_tok, lt, w = rows[i]
            M.append(MAPS[tk]); tok.append(tk_tok.float()); lat.append(lt.float()); ray.append(RAYS[sc]); W.append(w)
        M = torch.stack(M).to(dev); tok = torch.stack(tok).to(dev); x0 = torch.stack(lat).to(dev); ray = torch.stack(ray).to(dev).float()
        W = torch.stack(W).to(dev)[:, None]
        t = t.to(dev); eps = eps.to(dev); ab = acp[t].view(-1, 1, 1, 1); xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        for g_ in opt.param_groups:
            g_["lr"] = 1e-4 * min(1, (step + 1) / 100)
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
            ctx = adapter(tok, ray, style, drop.to(dev))
        unet.conv_in.m = M
        with torch.autocast("cuda", dtype=AMP):
            pred = unet(xt, t, encoder_hidden_states=ctx).sample
        err = (pred.float() - eps) ** 2
        whole = err.mean()
        wsum = W.sum()
        car = (err.mean(1, keepdim=True) * W).sum() / wsum if wsum > 0 else whole
        loss = 0.5 * whole + 0.5 * car
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        CLIP[0] += 1; CLIP[1] += int(gn.item() > 1.0)  # amendment 3: clip 1.0 kept (as E3 / X1), share of clipped updates reported
        if step % 100 == 0:
            rec = {"step": step, "loss": round(loss.item(), 4), "whole": round(whole.item(), 4), "car": round(car.item(), 4),
                   "gn": round(gn.item(), 4), "wnorm": round(unet.conv_in.map.weight.norm().item(), 4), "cam_drop": pdrop, "clipped_share": round(CLIP[1] / max(1, CLIP[0]), 4),
                   "min": round((time.time() - t0) / 60, 1)}
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % 500 == 0 or step + 1 == a.steps:
            save(step + 1, ck)
        if step + 1 in SAVE:
            save(step + 1, f"{a.out}/ckpt_{step + 1}.pt")
    if a.budget_min:
        n_ = a.steps - start; per = (time.time() - t10) / max(1, n_ - 10) if n_ > 10 else float("nan")
        t_save = SAVE_T[0] / max(1, SAVE_T[1])
        proj = (T_PREP + 4000 * per + 6 * t_save + 120) / 60  # + ~2 min model load
        print(f"TIMING per update {per:.3f} s (steps 10-{n_}), map prep {T_PREP / 60:.1f} min, save {t_save:.1f} s -> projected 4k {proj:.0f} min (budget {a.budget_min:.0f})", flush=True)
        if not proj <= a.budget_min:
            print("TIMING_OVER_BUDGET", flush=True); raise SystemExit(3)
    print("CARMAP2_TRAIN_DONE")
