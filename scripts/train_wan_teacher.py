"""PLAN-video S1: bidirectional Wan2.1-T2V-1.3B teacher, guide channels + camera tokens (mrv/wan_cond.py), council/PLAN-video.md.

Recipe (signed S1 + owner + Fable 06:35 design notes):
- data: 144 train drives (data/temporal_lists.json "train"; train / dev overlap asserted). Clips of 5 latent frames at 512x256:
  with prob --p_start (0.15) a TRUE-START 17-tick clip (1 + 4x4 frames, first latent = single frame), else a MID-STREAM clip
  (5 consecutive regular latents j0..j0+4, j0 >= 1, of a whole-segment stream encoded in 4 phases; 20 ticks). Every tick of a
  clip is usable (ok, not hard, same seq, consecutive). Targets = Wan-VAE latents of the game-render keys; guides encoded the
  same way (scripts/wan_cache.py); camera tokens memory-mapped from scripts/wan_tokens.py.
- objective: rectified flow as Wan / diffusers FlowMatchEulerDiscreteScheduler: x_s = (1 - s) x0 + s eps, target v = eps - x0,
  timestep = 1000 s, MSE in fp32 over all latents. ONE sigma per clip, drawn from the SHIFTED-UNIFORM distribution
  s = k u / (1 + (k - 1) u), u ~ U(0, 1), k = --shift (default 2.0; why: Wan's own 480p setting is 3.0 (diffusers card / config;
  5.0 at 720p) for 21 latents x 30 x 52 = 32.8k tokens; the SD3 resolution rule scales the shift with sqrt(token count): by
  spatial tokens per frame (512 vs 1,560) that gives 3.0 x 0.57 = 1.7, by whole-clip tokens (2,560 vs 32.8k) 0.84; the
  pixel-aligned guide carries layout, but car placement is decided at high noise, so we keep more high-noise mass than the
  rule: k = 2.0, median sigma 0.67 (3.0 would give 0.75, 1.0 0.50). The sampling shift is swept on one dev drive at the
  first checkpoint (scripts/wan_rollout.py --shift); --sigma_dist logit_normal is an option, not the default).
- conditioning dropout, independent per clip (C9): guide 20 % (guide latent AND mask zeroed), else random holes with prob 0.5
  (1-3 rectangles on the latent grid, guide latent + mask zeroed inside, all frames); tokens 10 %. Both dropped (~2 %) = the
  unconditional row for CFG.
- trainable: LoRA r32 (attn1 / attn2 q k v out, ffn) in all 30 blocks + extra patch conv + token projector + gates (fp32);
  base transformer bf16 and VAE frozen (asserted after the first real optimizer step; stock parity at init logged).
- AdamW lr 1e-4, betas (0.9, 0.999), no weight decay, 500-update linear warm-up, grad clip 1.0, bf16 autocast, gradient
  checkpointing per block. Batch 8 clips per update (--micro < 8 = gradient accumulation, same update).
- RNG: every update's draws come from torch.Generator(seed * 100003 + step) on the CPU (picks, clip type, sigma, noise,
  dropout, holes) -> resumable by construction; torch / CUDA RNG states are saved anyway.
- saves: rolling <out>/ckpt.pt every 500 updates (atomic tmp + replace; the NUMBERED file of a milestone is written FIRST),
  numbered ckpt_<k>.pt at --save_at; each with trainable weights, optimizer, step, hours used, RNG, recipe, code / base /
  data hashes. Hard stop when the cumulative training hours (all jobs) reach --budget_hours (S1: 24, extension 36).
- logs (log.jsonl; every 100 updates, every update with --timing): loss, grad norm, lr, step_s (GPU), data_s, e2e, peak GB,
  batch_sha, cumulative pick sha, data sha, gate / extra-conv norms, hours; every --val_every updates a fixed DEV probe:
  flow loss at sigma 0.3 / 0.6 / 0.9 on 8 fixed mid-stream dev clips + 2 true starts with inputs hybrid / guide-only /
  tokens-only / none / other-drive (guide + tokens of a different dev drive): shows whether each path is used (no verdict).
- --overfit N: train on N fixed clips only (bug gate: the loss must collapse in <= 2k updates). --timing: throwaway weights,
  output dir must contain 'timing' / 'ctest', logs every update.
  python scripts/train_wan_teacher.py --out runs/wan_s1 --steps 30000 --budget_hours 24 [--resume] [--max_steps_this_job N]
"""
import argparse, hashlib, json, os, queue, random, threading, time
import numpy as np
import torch
from mrv import wan_cond as WC

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--cache", default="data/wan_cache_v1")
p.add_argument("--tokens", default="data/wan_tok_v1")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--wan", default=WC.WAN_ID)
p.add_argument("--prompt_embed", default="", help="optional cached umT5 embedding [1, n, 4096] (.pt); default = zero text slot")
p.add_argument("--rank", type=int, default=32)
p.add_argument("--tok_ticks", default="ends", choices=list(WC.SLOTS))
p.add_argument("--use_tokens", type=int, default=1)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--micro", type=int, default=8, help="clips per forward; batch / micro = gradient-accumulation steps")
p.add_argument("--steps", type=int, default=30000)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--shift", type=float, default=2.0)
p.add_argument("--sigma_dist", default="shifted_uniform", choices=["shifted_uniform", "logit_normal"])
p.add_argument("--p_start", type=float, default=0.15)
p.add_argument("--p_guide_drop", type=float, default=0.2)
p.add_argument("--p_tok_drop", type=float, default=0.1)
p.add_argument("--p_holes", type=float, default=0.5)
p.add_argument("--save_every", type=int, default=500)
p.add_argument("--save_at", default="1000,2000,4000,6000,8000,10000,12000,14000,16000,20000,24000,28000,32000")
p.add_argument("--val_every", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--resume", action="store_true")
p.add_argument("--max_steps_this_job", type=int, default=0)
p.add_argument("--budget_hours", type=float, default=24.0, help="hard stop on cumulative training hours over all jobs")
p.add_argument("--overfit", type=int, default=0, help="bug gate: train on this many fixed clips only")
p.add_argument("--grad_ckpt", type=int, default=1)
p.add_argument("--verify_cache", type=int, default=1, help="re-hash every cache / token file against its sidecar at start")
p.add_argument("--timing", action="store_true")
a = p.parse_args()
if a.timing and not any(s in a.out for s in ("timing", "ctest")):
    raise SystemExit("--timing runs must write to a *timing* / *ctest* directory")
if a.batch % a.micro:
    raise SystemExit("--batch must be a multiple of --micro")
os.makedirs(a.out, exist_ok=True)
dev, AMP = "cuda", torch.bfloat16
JOB_T0 = time.time()
CODE_SHA = hashlib.sha1(b"".join(open(f, "rb").read() for f in [__file__, "mrv/wan_cond.py"])).hexdigest()[:12]
LISTS = json.load(open(a.lists))
TRAIN, DEVL = sorted(LISTS["train"]), sorted(LISTS["dev"])
assert not set(TRAIN) & set(DEVL), "train / dev overlap"
NL = 5  # latent frames per clip (17-tick true start / 20-tick mid-stream)
S_SLOTS = WC.SLOTS[a.tok_ticks]


# ------------------------------------------------------------------------------------------------------------- data
def sidecar_ok(d, sc, verify):
    j = json.load(open(f"{d}/{sc}.json"))
    if verify:
        h = hashlib.sha256()
        with open(f"{d}/{sc}.pt", "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != j["sha256"]:
            raise SystemExit(f"{d}/{sc}.pt does not match its sidecar sha256")
    return j["sha256"]


def load_split(drives, verify):
    C, T, shas = {}, {}, []
    miss = [s for s in drives if not os.path.exists(f"{a.cache}/{s}.pt") or (a.use_tokens and not os.path.exists(f"{a.tokens}/{s}.pt"))]
    if miss:
        raise SystemExit(f"missing cache / token files for {len(miss)} drives, e.g. {miss[:3]}")
    for sc in drives:
        shas.append(sc + ":" + sidecar_ok(a.cache, sc, verify))
        C[sc] = torch.load(f"{a.cache}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True)
        if a.use_tokens:
            shas.append(sc + ":tok:" + sidecar_ok(a.tokens, sc, verify))
            T[sc] = torch.load(f"{a.tokens}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True)
            assert torch.equal(T[sc]["ticks"], C[sc]["ticks"]), f"{sc}: token ticks != cache ticks"
    return C, T, shas


def candidates(C):
    mid, start = [], []
    for sc, c in C.items():
        ok = c["ok"].tolist(); hard = c["hard"].tolist()
        for si, st in enumerate(c["streams"]):
            Lz, p0 = st["key"].shape[1], st["pos0"]
            for j0 in range(1, Lz - NL + 1):
                fr = range(p0 + 4 * j0 - 3, p0 + 4 * j0 + 4 * NL - 3)
                if all(ok[i] and not hard[i] for i in fr):
                    mid.append((sc, si, j0))
        start += [(sc, k) for k in range(len(c["starts"]))]
    return mid, start


def clip(C, T, desc, kind):
    """-> key [16,5,32,64] f32, guide [16,5,32,64], mask [4,5,32,64], tok [5,S,6,P,384] f16 | None, ray [5,S,6,P,6] | None"""
    if kind == "mid":
        sc, si, j0 = desc; c = C[sc]; st = c["streams"][si]
        key, gd = st["key"][:, j0:j0 + NL].float(), st["guide"][:, j0:j0 + NL].float()
        spans = [[st["pos0"] + 4 * j - 3 + k for k in range(4)] for j in range(j0, j0 + NL)]
    else:
        sc, k_ = desc; c = C[sc]; s0 = int(c["starts"][k_])
        key, gd = c["start_key"][k_].float(), c["start_guide"][k_].float()
        spans = [[s0 + f for f in sp] for sp in WC.latent_spans(NL, first_single=True)]
    mask = WC.fold_mask(c["mask_lr"].float(), spans)
    tok = ray = None
    if a.use_tokens:
        idx = torch.tensor([i for sp in spans for i in WC.token_slots(sp, a.tok_ticks)])
        tk = T[sc]["tokens"][idx]  # [5*S, 6, P, 384]
        tok = tk.view(NL, S_SLOTS, *tk.shape[1:])
        r = T[sc]["rays"]
        ray = r.view(1, 1, *r.shape).expand(NL, S_SLOTS, *r.shape)
    return key, gd, mask, tok, ray


C_TR, T_TR, SHAS = load_split(TRAIN, a.verify_cache)
MID, START = candidates(C_TR)
DATA_SHA = hashlib.sha256("\n".join(SHAS).encode()).hexdigest()[:16]
VAE_SHAS = {c["meta"]["vae_sha"] for c in C_TR.values()}
assert len(VAE_SHAS) == 1, f"cache built with several VAEs: {VAE_SHAS}"
print(f"clips: {len(MID)} mid-stream + {len(START)} true starts from {len(C_TR)} drives; data_sha {DATA_SHA}", flush=True)
if a.overfit:
    g0 = torch.Generator().manual_seed(1234)
    OF = [("mid", MID[i]) for i in torch.randperm(len(MID), generator=g0)[:a.overfit - a.overfit // 4].tolist()] + \
         [("start", START[i]) for i in torch.randperm(len(START), generator=g0)[:a.overfit // 4].tolist()]


def sample_sigma(g, n):
    if a.sigma_dist == "logit_normal":
        u = torch.sigmoid(torch.randn(n, generator=g))
    else:
        u = torch.rand(n, generator=g)
    return WC.sigma_shift(u, a.shift).clamp(1e-4, 1.0)


def make_batch(step):
    """all random draws of update `step` (CPU generator) + the assembled batch"""
    g = torch.Generator().manual_seed(a.seed * 100003 + step)
    B = a.batch
    is_start = (torch.rand(B, generator=g) < a.p_start).tolist()
    pm = torch.randint(0, len(MID), (B,), generator=g).tolist(); ps = torch.randint(0, max(1, len(START)), (B,), generator=g).tolist()
    sig = sample_sigma(g, B)
    eps = torch.randn(B, 16, NL, 32, 64, generator=g)
    gdrop = torch.rand(B, generator=g) < a.p_guide_drop
    tdrop = torch.rand(B, generator=g) < a.p_tok_drop
    holes = torch.rand(B, generator=g) < a.p_holes
    hole_boxes = [[(int(torch.randint(3, 13, (1,), generator=g)), int(torch.randint(4, 25, (1,), generator=g)),
                    float(torch.rand(1, generator=g)), float(torch.rand(1, generator=g))) for _ in range(3)] for _ in range(B)]
    n_holes = torch.randint(1, 4, (B,), generator=g).tolist()
    descs = []
    for b in range(B):
        if a.overfit:
            descs.append(OF[(pm[b]) % len(OF)])
        elif is_start[b] and START:
            descs.append(("start", START[ps[b]]))
        else:
            descs.append(("mid", MID[pm[b]]))
    K, G, M, TK, RY = [], [], [], [], []
    for kind, d in descs:
        k_, g_, m_, t_, r_ = clip(C_TR, T_TR, d, kind)
        K.append(k_); G.append(g_); M.append(m_); TK.append(t_); RY.append(r_)
    K, G, M = torch.stack(K), torch.stack(G), torch.stack(M)
    for b in range(B):
        if gdrop[b]:
            G[b] = 0; M[b] = 0
        elif holes[b]:
            for hh, ww, fy, fx in hole_boxes[b][:n_holes[b]]:
                y0, x0 = int(fy * (32 - hh)), int(fx * (64 - ww))
                G[b, :, :, y0:y0 + hh, x0:x0 + ww] = 0; M[b, :, :, y0:y0 + hh, x0:x0 + ww] = 0
    TK = torch.stack(TK) if a.use_tokens else None; RY = torch.stack(RY) if a.use_tokens else None
    sha = hashlib.sha1(str(descs).encode() + sig.numpy().tobytes() + eps[:, 0, 0, 0, :4].numpy().tobytes()
                       + gdrop.numpy().tobytes() + tdrop.numpy().tobytes() + holes.numpy().tobytes()).hexdigest()[:10]
    return {"key": K, "guide": G, "mask": M, "tok": TK, "ray": RY, "sigma": sig, "eps": eps, "tkeep": ~tdrop, "sha": sha,
            "descs": descs, "n_start": sum(1 for k, _ in descs if k == "start"), "gdrop": int(gdrop.sum()), "tdrop": int(tdrop.sum())}


# --------------------------------------------------------------------------------------------------------- model
prompt = torch.load(a.prompt_embed, map_location="cpu") if a.prompt_embed else None
model, PAR = WC.build(a.wan, rank=a.rank, tok_ticks=a.tok_ticks, use_tokens=bool(a.use_tokens), dtype=AMP, device=dev, prompt=prompt,
                      grad_ckpt=bool(a.grad_ckpt), parity=True)
print(f"stock parity at init: max|wrapped - stock| = {PAR['max_abs']:.3g} (output std {PAR['ref_std']:.3g})", flush=True)
if PAR["max_abs"] > 0.05 * PAR["ref_std"]:
    raise SystemExit(f"init is not stock Wan: {PAR}")
model.train()
TR = model.trainable()
params = [q for _, q in TR]
NEW = [n for n, _ in TR if "lora_" not in n]
n_lora = sum(q.numel() for n, q in TR if "lora_" in n); n_new = sum(q.numel() for n, q in TR if "lora_" not in n)
print(f"trainable: LoRA {n_lora / 1e6:.1f}M ({sum(1 for n, _ in TR if 'lora_' in n)} tensors), new {n_new / 1e6:.2f}M ({len(NEW)} tensors)", flush=True)
BASE_SHA = WC.base_sha(model)
from diffusers import AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained(a.wan, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).eval()
vae.requires_grad_(False)
VAE_SHA = WC.vae_sha(vae)
if VAE_SHAS != {VAE_SHA}:
    raise SystemExit(f"cache latents were made with another VAE ({VAE_SHAS}) than the one on disk ({VAE_SHA})")
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
RECIPE = {k: getattr(a, k) for k in ("seed", "lists", "cache", "tokens", "rank", "tok_ticks", "use_tokens", "batch", "lr", "warmup", "shift",
                                     "sigma_dist", "p_start", "p_guide_drop", "p_tok_drop", "p_holes", "overfit", "prompt_embed", "wan")}
CFG = {"rank": a.rank, "tok_ticks": a.tok_ticks, "use_tokens": bool(a.use_tokens), "shift": a.shift, "n_extra": WC.N_EXTRA,
       "prompt_embed": a.prompt_embed, "wan": a.wan, "clip_latents": NL}
start, hours_prev, ck = 0, 0.0, f"{a.out}/ckpt.pt"
if os.path.exists(ck):
    if not a.resume:
        raise SystemExit(f"{ck} exists: pass --resume or use a fresh --out")
    s_ = torch.load(ck, map_location="cpu", weights_only=False)
    if s_.get("timing") and not a.timing:
        raise SystemExit("refusing to resume production training from a timing / test checkpoint")
    for k_, v_ in RECIPE.items():
        if s_["recipe"].get(k_) != v_:
            raise SystemExit(f"incompatible resume: recipe {k_} {s_['recipe'].get(k_)} != {v_}")
    for k_, v_ in (("code_sha", CODE_SHA), ("base_sha", BASE_SHA), ("vae_sha", VAE_SHA), ("data_sha", DATA_SHA)):
        if s_.get(k_) != v_:
            raise SystemExit(f"incompatible resume: {k_} {s_.get(k_)} != {v_}")
    model.load_trainable(s_["trainable"]); opt.load_state_dict(s_["opt"])
    start, hours_prev = s_["step"], s_["hours"]
    torch.set_rng_state(s_["rng"]["torch"]); torch.cuda.set_rng_state(s_["rng"]["cuda"]); random.setstate(s_["rng"]["python"]); np.random.set_state(s_["rng"]["numpy"])
    print(f"resumed at update {start}, {hours_prev:.2f} h used", flush=True)
json.dump({**vars(a), "code_sha": CODE_SHA, "base_sha": BASE_SHA, "vae_sha": VAE_SHA, "data_sha": DATA_SHA, "mid_clips": len(MID),
           "start_clips": len(START), "trainable_lora_M": round(n_lora / 1e6, 2), "trainable_new_M": round(n_new / 1e6, 3),
           "stock_parity": PAR, "cfg": CFG, "train_drives": len(TRAIN)}, open(f"{a.out}/args.json", "w"), indent=1)


def hours_used():
    return hours_prev + (time.time() - JOB_T0) / 3600


def save(step, path):
    d = {"trainable": model.trainable_state(), "opt": opt.state_dict(), "step": step, "hours": hours_used(), "recipe": RECIPE, "cfg": CFG,
         "code_sha": CODE_SHA, "base_sha": BASE_SHA, "vae_sha": VAE_SHA, "data_sha": DATA_SHA, "timing": a.timing, "args": vars(a),
         "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}}
    torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)


# --------------------------------------------------------------------------------------------- forward / loss
def to_dev(bt):
    out = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in bt.items()}
    return out


def flow_loss(bt, sl, reduce=True):
    x0, eps, sig = bt["key"][sl], bt["eps"][sl], bt["sigma"][sl]
    s5 = sig.view(-1, 1, 1, 1, 1)
    xt = (1 - s5) * x0 + s5 * eps
    extra = torch.cat([bt["guide"][sl], bt["mask"][sl]], 1)
    tok = bt["tok"][sl] if bt["tok"] is not None else None
    ray = bt["ray"][sl] if bt["ray"] is not None else None
    with torch.autocast("cuda", dtype=AMP):
        v = model(xt, extra, sig * 1000.0, tok, ray, bt["tkeep"][sl])
    err = (v.float() - (eps - x0)) ** 2
    return err.mean() if reduce else err.mean(dim=(1, 2, 3, 4))


# ------------------------------------------------------------------------------------------- dev probe (fixed)
PROBE = None
if a.val_every and all(os.path.exists(f"{a.cache}/{s}.pt") for s in DEVL) and (not a.use_tokens or all(os.path.exists(f"{a.tokens}/{s}.pt") for s in DEVL)):
    C_DV, T_DV, _ = load_split(DEVL, False)
    mid_d, start_d = candidates(C_DV)
    gp = torch.Generator().manual_seed(4242)
    by_drive = {}
    for d in mid_d:
        by_drive.setdefault(d[0], []).append(d)
    drv = sorted(by_drive)
    sel = [("mid", by_drive[drv[i % len(drv)]][int(torch.randint(0, len(by_drive[drv[i % len(drv)]]), (1,), generator=gp))]) for i in range(8)]
    sel += [("start", start_d[int(i)]) for i in torch.randperm(len(start_d), generator=gp)[:2].tolist()] if start_d else []
    def _stack(items):
        cl = [clip(C_DV, T_DV, d, k) for k, d in items]
        return [torch.stack([c[i] for c in cl]) if cl[0][i] is not None else None for i in range(5)]
    # donor clips: same kind, a DIFFERENT dev drive
    donors = []
    for k, d in sel:
        pool = [x for x in (mid_d if k == "mid" else start_d) if x[0] != d[0]]
        donors.append((k, pool[int(torch.randint(0, len(pool), (1,), generator=gp))]))
    PK, PG, PM, PT, PR = _stack(sel); _, DG, DM, DT, DR = _stack(donors)
    PEPS = torch.randn(len(sel), 16, NL, 32, 64, generator=gp)
    PROBE = {"key": PK, "guide": PG, "mask": PM, "tok": PT, "ray": PR, "dguide": DG, "dmask": DM, "dtok": DT, "dray": DR, "eps": PEPS,
             "clips": [(k, list(d)) for k, d in sel]}
    print(f"dev probe: {len(sel)} fixed clips from {len({d[0] for _, d in sel})} dev drives", flush=True)
    del C_DV, T_DV


@torch.no_grad()
def probe(step):
    model.eval(); res = {}
    n = PROBE["key"].shape[0]; on = torch.ones(n, dtype=torch.bool); off = torch.zeros(n, dtype=torch.bool)
    zg, zm = torch.zeros_like(PROBE["guide"]), torch.zeros_like(PROBE["mask"])
    modes = {"hybrid": (PROBE["guide"], PROBE["mask"], PROBE["tok"], PROBE["ray"], on),
             "guide_only": (PROBE["guide"], PROBE["mask"], PROBE["tok"], PROBE["ray"], off),
             "tokens_only": (zg, zm, PROBE["tok"], PROBE["ray"], on),
             "none": (zg, zm, PROBE["tok"], PROBE["ray"], off),
             "other_drive": (PROBE["dguide"], PROBE["dmask"], PROBE["dtok"], PROBE["dray"], on)}
    for sv in (0.3, 0.6, 0.9):
        for name, (g_, m_, t_, r_, kp) in modes.items():
            bt = to_dev({"key": PROBE["key"], "eps": PROBE["eps"], "sigma": torch.full((n,), sv), "guide": g_, "mask": m_, "tok": t_, "ray": r_, "tkeep": kp})
            ls = []
            for i in range(0, n, a.micro):
                ls.append(flow_loss(bt, slice(i, i + a.micro), reduce=False).cpu())
            res[f"{sv}"] = res.get(f"{sv}", {}); res[f"{sv}"][name] = round(torch.cat(ls).mean().item(), 5)
    model.train()
    rec = {"update": step, "probe": res, "code_sha": CODE_SHA}
    print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()


# ------------------------------------------------------------------------------------------------ prefetch
end = a.steps if not a.max_steps_this_job else min(a.steps, start + a.max_steps_this_job)
Q = queue.Queue(maxsize=3)


def producer():
    try:
        for s in range(start, end):
            t0 = time.time(); bt = make_batch(s)
            for k in ("key", "guide", "mask", "eps", "tok", "ray"):
                if bt[k] is not None:
                    bt[k] = bt[k].pin_memory()
            bt["data_s"] = time.time() - t0
            Q.put((s, bt))
    except BaseException as e:  # surface data errors in the main loop instead of hanging on Q.get()
        Q.put((-1, e))


threading.Thread(target=producer, daemon=True).start()
SAVE = set(int(x) for x in a.save_at.split(","))
log = open(f"{a.out}/log.jsonl", "a")
PICK_H = hashlib.sha1()
T_WIN, N_WIN = time.time(), 0
acc = {"data": 0.0, "gpu": 0.0, "wait": 0.0, "n": 0}
stopped = None
for step in range(start, end):
    tw = time.time(); s_, bt = Q.get()
    if s_ == -1:
        raise SystemExit(f"data producer failed: {bt!r}")
    assert s_ == step
    acc["wait"] += time.time() - tw; acc["data"] += bt["data_s"]
    PICK_H.update(bt["sha"].encode())
    t1 = time.time()
    bt = to_dev(bt)
    for g_ in opt.param_groups:
        g_["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
    if step == start:
        PRE_NEW = {n: q.detach().clone() for n, q in TR if "lora_" not in n or "lora_B" in n}
        FROZEN_REF = model.frozen(); FROZEN_PRE = [q.detach().cpu().clone() for _, q in FROZEN_REF]
        VAE_PRE = WC.vae_sha(vae)
    opt.zero_grad(set_to_none=True)
    nacc = a.batch // a.micro; loss_v = 0.0
    for i in range(nacc):
        l_ = flow_loss(bt, slice(i * a.micro, (i + 1) * a.micro)) / nacc
        l_.backward(); loss_v += l_.item()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    if not torch.isfinite(gn):
        raise SystemExit(f"non-finite grad norm at update {step + 1}")
    opt.step()
    if step == start:  # first-update guard: base + VAE bit-identical after a REAL optimizer step; new parameters moved
        same = all(torch.equal(q.detach().cpu(), q0) for (_, q), q0 in zip(FROZEN_REF, FROZEN_PRE))
        assert same, "frozen base transformer changed!"
        assert WC.vae_sha(vae) == VAE_PRE == VAE_SHA, "VAE changed!"
        cur = dict(TR); moved = {n: not torch.equal(cur[n], q0) for n, q0 in PRE_NEW.items()}
        # attn2 to_q / to_k LoRA get no gradient in update 1: the text slot is one row (softmax over one key ignores q and k) and
        # the token path is gated by exactly 0; they start moving once the gates are non-zero
        must = [n for n in moved if ".patch_embedding.extra." in n or ("lora_B" in n and not any(x in n for x in (".attn2.inner.to_q.", ".attn2.inner.to_k.")))
                or (n.endswith(".attn2.gate") and bool(bt["tkeep"].any()))]
        if start == 0:
            bad = [n for n in must if not moved[n]]
            assert not bad, f"{len(bad)} new / LoRA-B tensors did not move in update 1, e.g. {bad[:3]}"
        print(f"frozen check after update {step + 1}: {len(FROZEN_REF)} base tensors + VAE bit-identical; "
              f"{sum(moved.values())}/{len(moved)} new + LoRA-B tensors moved (projector / LoRA-A start moving once gates / B are non-zero)", flush=True)
        del FROZEN_PRE, PRE_NEW
    torch.cuda.synchronize(); acc["gpu"] += time.time() - t1; acc["n"] += 1
    if (step + 1) % 100 == 0 or step == start or step == end - 1 or a.timing:
        gates = [m.gate for m in model.xattn]
        rec = {"update": step + 1, "loss": round(loss_v, 5), "gn": round(gn.item(), 4), "lr": opt.param_groups[0]["lr"],
               "sigma_mean": round(bt["sigma"].mean().item(), 3), "n_start": bt["n_start"], "gdrop": bt["gdrop"], "tdrop": bt["tdrop"],
               "step_s": round(acc["gpu"] / acc["n"], 3), "data_s": round(acc["data"] / acc["n"], 3), "wait_s": round(acc["wait"] / acc["n"], 3),
               "e2e_s": round((time.time() - T_WIN) / max(1, step + 1 - start - N_WIN), 3),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "gate_norm": round(sum(g.norm().item() ** 2 for g in gates) ** 0.5, 4),
               "extra_norm": round(model.tf.patch_embedding.extra.weight.norm().item(), 4), "batch_sha": bt["sha"],
               "pick_sha_cum": PICK_H.hexdigest()[:10], "data_sha": DATA_SHA, "hours": round(hours_used(), 3), "code_sha": CODE_SHA}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        acc = {"data": 0.0, "gpu": 0.0, "wait": 0.0, "n": 0}; T_WIN, N_WIN = time.time(), step + 1 - start
    if step + 1 in SAVE:  # numbered milestone FIRST, then the rolling file
        save(step + 1, f"{a.out}/ckpt_{step + 1}.pt")
    if (step + 1) % a.save_every == 0 or step + 1 == end:
        save(step + 1, ck)
    if PROBE is not None and ((step + 1) % a.val_every == 0 or step == start):
        probe(step + 1)
    if hours_used() >= a.budget_hours:
        save(step + 1, ck); stopped = step + 1
        rec = {"budget_stop": step + 1, "hours": round(hours_used(), 3), "budget_hours": a.budget_hours}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        break
print("WAN_TEACHER_BUDGET_STOP" if stopped else "WAN_TEACHER_DONE", stopped or end)
