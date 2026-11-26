"""Car map X1 / X2 (council/PLAN-carmap.md FINAL v3): train only a map branch on top of the frozen E3 renderer.

  python scripts/train_carmap.py --init runs/tpvE3/ckpt_10000.pt --boxes 'tmp/bev_boxes_train_*.json' --out runs/cmX1
  python scripts/train_carmap.py ... --grid data/bev_tpv --out runs/cmX2

Data: flat keyframe keys (data/lat512_tpv) of the 144 temporal train drives (dev drives excluded), DINO tokens
(data/enc_full), calibrated rays; map = BEVFormer boxes (score >= 0.3) painted into the chase view (mrv/carmap.py),
area-pooled to 32x64. X2 adds the BEV grid sampled where each latent cell's ray meets z = 0 and z = 1 m (invalid
cells zero, with a valid plane), 2 x 256 -> 32 channels by a 1x1 conv. Branch residuals at the four encoder scales
(down_intrablock_additional_residuals), zero-initialised, so the branch off = E3 exactly. Same map in both CFG passes
at inference. AdamW 1e-4 (500 warm-up), batch 16, epsilon loss, 10% camera dropout, 15% box dropout (own stream),
style 2, 4,000 updates, checkpoints at 2k and 4k. Draws seeded, so X1 and X2 see identical rows / noise / dropout.
"""
import argparse, glob, json, math, os, time, zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.carmap import rasterize, pool, CarMapBranch, C_MAP

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--boxes", required=True, help="glob of BEVFormer box JSONs (bev_extract.py --boxes_out)")
p.add_argument("--grid", default="", help="X2: folder of cached BEV grids (bev_extract.py --out)")
p.add_argument("--keys", default="data/lat512_tpv")
p.add_argument("--tokens", default="data/enc_full")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--save_at", default="2000,4000")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev, AMP = "cuda", torch.bfloat16
SD = "sd-legacy/stable-diffusion-v1-5"
L = json.load(open(a.lists)); TRAIN = set(L["train"])
RAYS = torch.load(a.rays)
BOX = {}
for f in glob.glob(a.boxes):
    BOX.update(json.load(open(f)))

# grid sampling geometry (X2): ray of each 32x64 latent cell -> ego ground points at z = 0 and z = 1 m -> BEV crop coords
if a.grid:
    W, H, CROP = 512, 288, 32
    fpx = W / 2 / math.tan(math.radians(30)); cp, sp = math.cos(math.radians(8)), math.sin(math.radians(8))
    vv, uu = np.meshgrid((np.arange(32) + 0.5) * 8 + CROP, (np.arange(64) + 0.5) * 8, indexing="ij")
    x_r = (uu - W / 2) / fpx; y_u = -(vv - H / 2) / fpx  # camera right, up per unit depth
    fwd = np.stack([cp + 0 * x_r, -x_r, -sp + 0 * x_r], -1) + y_u[..., None] * np.array([sp, 0, cp])  # ego dirs (x fwd, y left, z up)
    origin = np.array([-7.5, 0, 2.4])
    GRIDS, VALID = [], []
    for zh in (0.0, 1.0):
        t = (zh - origin[2]) / np.where(fwd[..., 2] < -1e-6, fwd[..., 2], np.nan)
        P = origin + fwd * t[..., None]
        gx = (P[..., 0] - (-8.0)) / 49.0 * 2 - 1  # columns: x -8..41 m
        gy = (16.4 - P[..., 1]) / 32.8 * 2 - 1    # rows: +16.4 (left) .. -16.4 m
        ok = np.isfinite(t) & (t > 0) & (np.abs(gx) <= 1) & (np.abs(gy) <= 1)
        GRIDS.append(torch.tensor(np.stack([np.nan_to_num(gx, nan=2.0), np.nan_to_num(gy, nan=2.0)], -1), dtype=torch.float32))
        VALID.append(torch.tensor(ok, dtype=torch.float32))

# ---------- data: keyframe rows of the train drives ----------
rows = []
GRIDCACHE = {}
for f in sorted(glob.glob(f"{a.keys}/*.pt")):
    sc = os.path.basename(f)[:-3]
    if sc not in TRAIN or not os.path.exists(f"{a.tokens}/{sc}.pt"):
        continue
    k = torch.load(f); e = torch.load(f"{a.tokens}/{sc}.pt")
    pos = {t: i for i, t in enumerate(e["sample_tokens"])}
    g = torch.load(f"{a.grid}/{sc}.pt") if a.grid and os.path.exists(f"{a.grid}/{sc}.pt") else None
    gpos = {t: i for i, t in enumerate(g["sample_tokens"])} if g is not None else {}
    for j, t in enumerate(k["sample_tokens"]):
        if t not in BOX or (a.grid and t not in gpos):
            continue
        rows.append((sc, t, e["tokens"][pos[t]], k["lat"][j]))
        if g is not None:
            GRIDCACHE[t] = g["bev"][gpos[t]]
print(f"rows {len(rows)} from {len({r[0] for r in rows})} drives; boxes for {len(BOX)} frames; grid {'on' if a.grid else 'off'}", flush=True)


def map_for(tok, dropper):
    m = pool(rasterize(BOX[tok], drop=dropper))
    if a.grid:
        bev = GRIDCACHE[tok].float()[None]  # (1, 256, 32, 48)
        samp = [F.grid_sample(bev, gr[None], align_corners=False)[0] * v for gr, v in zip(GRIDS, VALID)]
        m = torch.cat([m, *samp, VALID[0][None], VALID[1][None]])
    return m


# ---------- model ----------
ck0 = torch.load(a.init, map_location="cpu"); r = ck0["args"]["rank"]
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); unet.load_state_dict(ck0["lora"], strict=False)
unet.requires_grad_(False); unet.to(dev, AMP).eval()
adapter = CameraAdapter(n_styles=ck0["args"].get("n_styles", 3)); adapter.load_state_dict(ck0["adapter"])
adapter.requires_grad_(False); adapter.to(dev).eval()
c_in = C_MAP + (2 * 256 + 2 if a.grid else 0)
branch = CarMapBranch(c_in=C_MAP + (32 + 2 if a.grid else 0)).to(dev)
reduce = nn.Conv2d(2 * 256, 32, 1).to(dev) if a.grid else None
params = list(branch.parameters()) + (list(reduce.parameters()) if reduce is not None else [])
opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
sched = DDPMScheduler.from_pretrained(SD, subfolder="scheduler"); acp = sched.alphas_cumprod.to(dev)
start = 0
ck = f"{a.out}/ckpt.pt"
if os.path.exists(ck):
    s_ = torch.load(ck, map_location="cpu"); branch.load_state_dict(s_["branch"])
    if reduce is not None:
        reduce.load_state_dict(s_["reduce"])
    opt.load_state_dict(s_["opt"]); start = s_["step"]; print("resumed at", start, flush=True)


def prep(m):
    if not a.grid:
        return m
    return torch.cat([m[:, :C_MAP], reduce(m[:, C_MAP:C_MAP + 512]), m[:, C_MAP + 512:]], 1)


def save(step, path):
    d = {"branch": branch.state_dict(), "opt": opt.state_dict(), "step": step, "init": a.init, "grid": bool(a.grid),
         "args": {**ck0["args"], "carmap": vars(a)}, "adapter": ck0["adapter"], "lora": ck0["lora"]}
    if reduce is not None:
        d["reduce"] = reduce.state_dict()
    torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)


style = torch.full((16,), 2, dtype=torch.long, device=dev)
log = open(f"{a.out}/log.jsonl", "a"); t0 = time.time(); SAVE = set(int(x) for x in a.save_at.split(","))
for step in range(start, a.steps):
    g = torch.Generator().manual_seed(a.seed * 100003 + step)
    idx = torch.randint(0, len(rows), (16,), generator=g); t = torch.randint(0, 1000, (16,), generator=g)
    eps = torch.randn(16, 4, 32, 64, generator=g); drop = torch.rand(16, generator=g) < 0.1
    gb = torch.Generator().manual_seed(zlib.crc32(f"boxdrop|{a.seed}|{step}".encode()))  # separate stream
    maps, tok, lat, ray = [], [], [], []
    for n, i in enumerate(idx.tolist()):
        sc, tk, tk_tok, lt = rows[i]
        u = torch.rand(64, generator=gb)
        maps.append(map_for(tk, lambda k_, u=u: bool(u[k_ % 64] < 0.15)))
        tok.append(tk_tok.float()); lat.append(lt.float()); ray.append(RAYS[sc])
    M = prep(torch.stack(maps).to(dev)); tok = torch.stack(tok).to(dev); x0 = torch.stack(lat).to(dev); ray = torch.stack(ray).to(dev).float()
    t = t.to(dev); eps = eps.to(dev); ab = acp[t].view(-1, 1, 1, 1); xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    for g_ in opt.param_groups:
        g_["lr"] = 1e-4 * min(1, (step + 1) / 500)
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        ctx = adapter(tok, ray, style, drop.to(dev))
    with torch.autocast("cuda", dtype=AMP):
        res = branch(M)
        pred = unet(xt, t, encoder_hidden_states=ctx, down_intrablock_additional_residuals=[x.to(AMP) for x in res]).sample
    loss = ((pred.float() - eps) ** 2).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if step % 100 == 0:
        rec = {"step": step, "loss": round(loss.item(), 4), "gn": round(gn.item(), 3), "min": round((time.time() - t0) / 60, 1)}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    if (step + 1) % 1000 == 0 or step + 1 == a.steps:
        save(step + 1, ck)
    if step + 1 in SAVE:
        save(step + 1, f"{a.out}/ckpt_{step + 1}.pt")
print("CARMAP_TRAIN_DONE")
