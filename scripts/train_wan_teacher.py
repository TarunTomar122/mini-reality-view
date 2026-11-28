"""PLAN-video S1: bidirectional Wan2.1-T2V-1.3B teacher, guide channels + camera tokens (mrv/wan_cond.py), council/PLAN-video.md;
revised for Astra's review council/codereview-wan-s1-astra.md (items 3-8, 11, 12) and the underlay guide (415c470).

Recipe
- data: 144 train drives (data/temporal_lists.json "train"; train / dev overlap asserted; sealed splits never read). Clips of 5
  latent frames at 512x256: with prob --p_start (0.15) a TRUE-START 17-tick clip, else a MID-STREAM clip (5 regular latents
  j0..j0+4, j0 >= 1, of a whole-segment stream encoded in 4 phases; 20 ticks). Streams run through soft rejects; every tick of
  a TRAINING clip is ok (mrv/wan_plan.py). Candidate counts per drive are printed and stored in args.json before sampling.
  Targets = Wan-VAE latents of the keys; guides encoded identically + source planes (depth / ground share, 2 x 4 folded);
  camera tokens + per-tick rays memory-mapped (scripts/wan_tokens.py). Every cache / token file is re-hashed against its
  sidecar and all carry ONE guide version (train and dev).
- objective: rectified flow (x_s = (1 - s) x0 + s eps, v = eps - x0, timestep 1000 s), MSE fp32; one sigma per clip,
  shifted-uniform with shift --shift 2.0 (Wan 480p uses 3.0 for 32.8k tokens; the SD3 sqrt(token) rule gives 1.7 per frame /
  0.84 per clip for our 2,560; kept above the rule because car placement is decided at high noise; median sigma 0.67).
- dropout, independent per clip: guide 20 % (latent + source planes zeroed), else holes w.p. 0.5; tokens 10 %. Updates 1 and 2
  (absolute steps 0 / 1) use NO dropout / holes, so the first-update guard and the gate-open gradient check are deterministic.
- trainable: LoRA r32 on the 10 targets x 30 blocks (asserted) + extra patch conv + projector + gates, fp32, seeded init; base
  bf16 + VAE frozen. AdamW 1e-4, no weight decay, 500-update warm-up, clip 1.0, bf16 autocast, per-block checkpointing,
  global batch 8 (--micro < 8 = accumulation).
- init diagnostics (item 4): stock parity with the same 512-row padded context and precision, 1-row shortcut, guide / token
  inertness, explicit finiteness (mrv.wan_cond.build(parity=True), check_parity).
- guard (item 5): after update 1 (fresh run): base transformer + VAE bit-identical; extra conv, all 30 gates (tokens on) and
  LoRA-B of 8 targets x 30 blocks moved (attn2 q / k LoRA cannot move yet: one text row, gate 0). Update 2: projector and
  attn2 q / k LoRA-B gradients non-zero in all 30 blocks (the gates are open).
- RNG: every update's draws from torch.Generator(seed * 100003 + step) (CPU) -> resumable by construction; the chained data hash
  pick_chain = sha1(prev | batch_sha) is saved and continued, so a resumed run must end on the uninterrupted run's chain.
- budget (item 11): ONE ledger (--ledger, jsonl heartbeats every 60 s) is debited by every S1 job: timing, overfit, resume test,
  training, panels (scripts/wan_panel.py); a job's charge = last beat - start, so preempted work is still debited. Before
  every update: stop (save) if used + this update + --reserve_hours (panel time) >= --budget_hours. Verdicts: below 12k updates
  unresolved; 24 h read -> pass / one 12 h extension only for car F1 in [0.4156, 0.4456) with every other gate passed / stop;
  after the extension the bar stays 0.4456 (mrv.wan_plan.decide_s1, applied by scripts/wan_panel.py).
- saves: numbered ckpt_<k>.pt (--save_at) FIRST, then the rolling ckpt.pt every 500 updates; atomic; trainable weights,
  optimizer, step, RNG, pick chain, recipe, code / base / VAE / data / guide-version hashes.
- logs (log.jsonl): loss, grad norm, lr, sigma, step_s, data_s, wait_s, e2e, peak alloc / reserved GB, batch / chain / data sha,
  gate / extra norms, ledger hours; every --val_every a fixed dev probe (flow loss at sigma 0.3 / 0.6 / 0.9, inputs hybrid /
  guide-only / tokens-only / none / other-drive, + output sensitivity to guide and tokens).
Modes
- --timing (item 7): warmed real updates at global batch 8 for microbatch 1, 2, 4, 8 (stops at OOM): wall / fwd+bwd / Adam /
  data wait / peak allocated + reserved; then probe, save, one 30-step rollout window (1 and 2 rows), Wan VAE encode (17 f) /
  decode (5 latents), SD-VAE encode, each timed; forecast of updates in the remaining budget vs the 12k floor -> timing.json.
- --overfit 8 (item 8): 8 DISTINCT fixed clips (6 mid-stream + 2 true starts) as every batch; fixed-noise / fixed-sigma eval of
  the SAME 8 every 100 updates; PASS = final eval loss <= --overfit_ratio (0.25) x the update-0 eval loss AND decoded samples
  (30 Euler steps, hybrid) reach car F1 >= --overfit_f1_frac (0.8) x the Wan-VAE round-trip car F1 against the keys (signed
  scorer's segmentation + 0.3 matcher), the round trip itself > 0; both gates read at update 2,000 only (--steps 2000 required;
  a shorter job prints OVERFIT_PENDING and resumes); the update-0 eval loss is stored in every checkpoint; sampling runs in the
  fitting microbatch -> overfit.json, exit 4 on FAIL.
  python scripts/train_wan_teacher.py --out runs/wan_s1 --steps 40000 --budget_hours 24 --ledger runs/wan_s1_ledger.jsonl --resume
"""
import argparse, hashlib, json, os, queue, random, threading, time
import numpy as np
import torch
from mrv import wan_cond as WC
from mrv import wan_plan as WP

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--cache", default="data/wan_cache_v1")
p.add_argument("--tokens", default="data/wan_tok_v1")
p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--guides", default="data/guides_v1")
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--wan", default=WC.WAN_ID)
p.add_argument("--prompt_embed", default="")
p.add_argument("--rank", type=int, default=32)
p.add_argument("--tok_ticks", default="last", choices=list(WC.SLOTS))
p.add_argument("--use_tokens", type=int, default=1)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--micro", type=int, default=8)
p.add_argument("--steps", type=int, default=40000)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--shift", type=float, default=2.0)
p.add_argument("--sigma_dist", default="shifted_uniform", choices=["shifted_uniform", "logit_normal"])
p.add_argument("--p_start", type=float, default=0.15)
p.add_argument("--p_guide_drop", type=float, default=0.2)
p.add_argument("--p_tok_drop", type=float, default=0.1)
p.add_argument("--p_holes", type=float, default=0.5)
p.add_argument("--save_every", type=int, default=500)
p.add_argument("--save_at", default="1000,2000,4000,6000,8000,10000,12000,14000,16000,20000,24000,28000,32000,36000,40000")
p.add_argument("--val_every", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--resume", action="store_true")
p.add_argument("--max_steps_this_job", type=int, default=0)
p.add_argument("--budget_hours", type=float, default=24.0, help="S1 slice-hour cap over ALL S1 jobs in --ledger")
p.add_argument("--reserve_hours", type=float, default=2.0, help="kept free for the dev panels (scripts/wan_panel.py)")
p.add_argument("--ledger", default="runs/wan_s1_ledger.jsonl")
p.add_argument("--kind", default="", help="ledger label (default: timing / overfit / train from the flags)")
p.add_argument("--overfit", type=int, default=0)
p.add_argument("--overfit_ratio", type=float, default=0.25)
p.add_argument("--overfit_f1_frac", type=float, default=0.8)
p.add_argument("--timing_micro", default="1,2,4,8")
p.add_argument("--grad_ckpt", type=int, default=1)
p.add_argument("--verify_cache", type=int, default=1)
p.add_argument("--timing", action="store_true", help="timing / test run: dir must contain 'timing' or 'ctest'")
a = p.parse_args()
TEST = a.timing or a.overfit
if TEST and not any(s in a.out for s in ("timing", "ctest")):
    raise SystemExit("--timing / --overfit runs must write to a *timing* / *ctest* directory")
if a.batch % a.micro:
    raise SystemExit("--batch must be a multiple of --micro")
if a.overfit and a.batch != a.overfit:
    raise SystemExit("--overfit N trains on exactly N clips per update: --batch N")
OVERFIT_STEPS = 2000  # declared: both overfit gates are read at update 2,000 (S1b item 6)
if a.overfit and a.steps != OVERFIT_STEPS:
    raise SystemExit(f"--overfit is declared at {OVERFIT_STEPS} updates: --steps {OVERFIT_STEPS}")
os.makedirs(a.out, exist_ok=True)
dev, AMP = "cuda", torch.bfloat16
KIND = a.kind or ("timing" if a.timing and not a.overfit else ("overfit" if a.overfit else "train"))
LEDGER = WP.Ledger(a.ledger, KIND).start()
CODE_SHA = WC.code_sha(__file__, "mrv/wan_cond.py", "mrv/wan_plan.py")
LISTS = json.load(open(a.lists))
TRAIN, DEVL = sorted(LISTS["train"]), sorted(LISTS["dev"])
WP.check_split_membership(TRAIN, LISTS, allowed=("train",)); WP.check_split_membership(DEVL, LISTS, allowed=("dev",))
assert not set(TRAIN) & set(DEVL), "train / dev overlap"
assert not (set(TRAIN) | set(DEVL)) & (set(LISTS.get("decision", [])) | set(LISTS.get("final", []))), "sealed drives in train / dev"
NL = 5
S_SLOTS = WC.SLOTS[a.tok_ticks]
if KIND == "train" and LEDGER.hours() + a.reserve_hours >= a.budget_hours:
    raise SystemExit(f"S1 budget exhausted: {LEDGER.hours():.2f} h used + {a.reserve_hours} h reserve >= {a.budget_hours} h")


# ------------------------------------------------------------------------------------------------------------- data
def sidecar(d, sc, verify):
    j = json.load(open(f"{d}/{sc}.json"))
    if not j.get("complete"):
        raise SystemExit(f"{d}/{sc}.json not complete")
    if verify and WC.file_sha(f"{d}/{sc}.pt") != j["sha256"]:
        raise SystemExit(f"{d}/{sc}.pt does not match its sidecar sha256")
    return j


GUIDE_VERSION = None


def load_split(drives, verify):
    global GUIDE_VERSION
    C, T, shas = {}, {}, []
    miss = [s for s in drives if not os.path.exists(f"{a.cache}/{s}.pt") or (a.use_tokens and not os.path.exists(f"{a.tokens}/{s}.pt"))]
    if miss:
        raise SystemExit(f"missing cache / token files for {len(miss)} drives, e.g. {miss[:3]}")
    for sc in drives:
        jc = sidecar(a.cache, sc, verify); shas.append(sc + ":" + jc["sha256"])
        jt = sidecar(a.tokens, sc, verify) if a.use_tokens else None
        for j_ in [jc] + ([jt] if jt else []):
            if GUIDE_VERSION is None:
                GUIDE_VERSION = j_["guide_version"]
            if j_["guide_version"] != GUIDE_VERSION:
                raise SystemExit(f"{sc}: mixed guide versions across drives / caches: {j_['guide_version']} vs {GUIDE_VERSION}")
        # S1b item 2: the latent and token caches of a drive must come from the SAME guide and enc files (equal ticks and
        # versions would admit a stale calibration), and with --verify_cache those files must still be the ones on disk
        if jt is not None and (jt.get("guide_sha") != jc.get("guide_sha") or jt.get("enc_sha") != jc.get("enc_sha")):
            raise SystemExit(f"{sc}: latent cache and token cache were built from different guide / enc files")
        if jt is not None and not (jt.get("shared_pass_ok") and jt.get("weights_sha")):
            raise SystemExit(f"{sc}: token cache lacks the shared-depth / weight-hash evidence")
        if verify:
            if WC.file_sha(f"{a.guides}/{sc}.pt") != jc.get("guide_sha") or WC.file_sha(f"{a.enc}/{sc}.pt") != jc.get("enc_sha"):
                raise SystemExit(f"{sc}: guide / enc file on disk differs from the one the caches were built from")
        C[sc] = torch.load(f"{a.cache}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True)
        if a.use_tokens:
            shas.append(sc + ":tok:" + json.load(open(f"{a.tokens}/{sc}.json"))["sha256"])
            T[sc] = torch.load(f"{a.tokens}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True)
            assert torch.equal(T[sc]["ticks"], C[sc]["ticks"]), f"{sc}: token ticks != cache ticks"
            assert tuple(T[sc]["tokens"].shape[2:]) == (WC.TOK_P, WC.TOK_DIM), f"{sc}: token grid {tuple(T[sc]['tokens'].shape)}"
    return C, T, shas


def candidates(C):
    mid, start, per = [], [], {}
    for sc, c in C.items():
        ok = c["ok"].tolist(); hard = c["hard"].tolist()
        st = [{"pos0": s_["pos0"], "L": s_["key"].shape[1]} for s_ in c["streams"]]
        m_ = [(sc, k, j0) for k, j0 in WP.mid_candidates(st, ok, hard, NL)]
        s_ = [(sc, k) for k in range(len(c["starts"]))]
        mid += m_; start += s_; per[sc] = (len(m_), len(s_))
    return mid, start, per


def clip(C, T, desc, kind):
    """-> key [16,5,32,64], guide [16,5,32,64], src [8,5,32,64], tok [5,S,6,P,384] f16 | None, ray [5,S,6,P,6] | None, ticks"""
    if kind == "mid":
        sc, si, j0 = desc; c = C[sc]; st = c["streams"][si]
        key, gd = st["key"][:, j0:j0 + NL].float(), st["guide"][:, j0:j0 + NL].float()
        spans = [[st["pos0"] + 4 * j - 3 + k for k in range(4)] for j in range(j0, j0 + NL)]
    else:
        sc, k_ = desc; c = C[sc]; s0 = int(c["starts"][k_])
        key, gd = c["start_key"][k_].float(), c["start_guide"][k_].float()
        spans = [[s0 + f for f in sp] for sp in WP.latent_spans(NL, first_single=True)]
    src = WC.fold_src(c["src_lr"].float(), spans)
    tok = ray = None
    if a.use_tokens:
        idx = torch.tensor([i for sp in spans for i in WP.token_slots(sp, a.tok_ticks)])
        tk, ry = T[sc]["tokens"][idx], T[sc]["rays"][idx]
        tok = tk.view(NL, S_SLOTS, *tk.shape[1:]); ray = ry.view(NL, S_SLOTS, *ry.shape[1:]).float()
    return key, gd, src, tok, ray, (sc, [sp for sp in spans])


C_TR, T_TR, SHAS = load_split(TRAIN, a.verify_cache)
MID, START, PER_DRIVE = candidates(C_TR)
DATA_SHA = hashlib.sha256("\n".join(SHAS).encode()).hexdigest()[:16]
VAE_SHAS = {c["meta"]["vae_sha"] for c in C_TR.values()}
assert len(VAE_SHAS) == 1, f"cache built with several VAEs: {VAE_SHAS}"
no_clip = [sc for sc, (m_, s_) in PER_DRIVE.items() if m_ + s_ == 0]
print(f"clips: {len(MID)} mid-stream + {len(START)} true starts; contributing drives {len(TRAIN) - len(no_clip)}/{len(TRAIN)}"
      f" (none from {no_clip[:6]}{'...' if len(no_clip) > 6 else ''}); data_sha {DATA_SHA}; guide version {GUIDE_VERSION}", flush=True)
if not MID:
    raise SystemExit("no mid-stream training clips")
if a.overfit:
    g0 = torch.Generator().manual_seed(1234)
    n_s = min(len(START), a.overfit // 4)
    OF = [("mid", MID[i]) for i in torch.randperm(len(MID), generator=g0)[:a.overfit - n_s].tolist()] + \
         [("start", START[i]) for i in torch.randperm(len(START), generator=g0)[:n_s].tolist()]
    assert len(OF) == a.overfit and len(set(OF)) == a.overfit, "overfit clips must be distinct"


def sample_sigma(g, n):
    u = torch.sigmoid(torch.randn(n, generator=g)) if a.sigma_dist == "logit_normal" else torch.rand(n, generator=g)
    return WC.sigma_shift(u, a.shift).clamp(1e-4, 1.0)


def assemble(items):
    cl = [clip(C_TR, T_TR, d, k) for k, d in items]
    st = lambda i: torch.stack([c[i] for c in cl]) if cl[0][i] is not None else None
    return st(0), st(1), st(2), st(3), st(4), [c[5] for c in cl]


def make_batch(step):
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
    if step < 2:  # deterministic active inputs for the guard (update 1) and the gate-open gradient check (update 2)
        gdrop[:] = False; tdrop[:] = False; holes[:] = False
    if a.overfit:
        descs = list(OF)
    else:
        descs = [("start", START[ps[b]]) if (is_start[b] and START) else ("mid", MID[pm[b]]) for b in range(B)]
    K, G, M, TK, RY, _ = assemble(descs)
    for b in range(B):
        if gdrop[b]:
            G[b] = 0; M[b] = 0
        elif holes[b]:
            for hh, ww, fy, fx in hole_boxes[b][:n_holes[b]]:
                y0, x0 = int(fy * (32 - hh)), int(fx * (64 - ww))
                G[b, :, :, y0:y0 + hh, x0:x0 + ww] = 0; M[b, :, :, y0:y0 + hh, x0:x0 + ww] = 0
    sha = hashlib.sha1(str(descs).encode() + sig.numpy().tobytes() + eps[:, 0, 0, 0, :4].numpy().tobytes()
                       + gdrop.numpy().tobytes() + tdrop.numpy().tobytes() + holes.numpy().tobytes()).hexdigest()[:10]
    return {"key": K, "guide": G, "src": M, "tok": TK, "ray": RY, "sigma": sig, "eps": eps, "tkeep": ~tdrop, "sha": sha,
            "n_start": sum(1 for k, _ in descs if k == "start"), "gdrop": int(gdrop.sum()), "tdrop": int(tdrop.sum()),
            "guide_active": bool((~gdrop).any()), "tok_active": bool((~tdrop).any())}


# --------------------------------------------------------------------------------------------------------- model
prompt = torch.load(a.prompt_embed, map_location="cpu") if a.prompt_embed else None
model, DIAG = WC.build(a.wan, rank=a.rank, tok_ticks=a.tok_ticks, use_tokens=bool(a.use_tokens), dtype=AMP, device=dev, prompt=prompt,
                       grad_ckpt=bool(a.grad_ckpt), parity=True, seed=a.seed)
print("init diagnostics", DIAG, flush=True)
WC.check_parity(DIAG)
model.train()
TR = model.trainable(); params = [q for _, q in TR]
INIT_SHA = WC.tensor_sha(TR)
n_lora = sum(q.numel() for n, q in TR if "lora_" in n); n_new = sum(q.numel() for n, q in TR if "lora_" not in n)
print(f"trainable: LoRA {n_lora / 1e6:.1f}M, new {n_new / 1e6:.2f}M; init sha {INIT_SHA}", flush=True)
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
       "prompt_embed": a.prompt_embed, "wan": a.wan, "clip_latents": NL, "tok_grid": WC.TOK_GRID, "seed": a.seed}
start, ck, CHAIN, OF0 = 0, f"{a.out}/ckpt.pt", "", None
if os.path.exists(ck):
    if not a.resume:
        raise SystemExit(f"{ck} exists: pass --resume or use a fresh --out")
    s_ = torch.load(ck, map_location="cpu", weights_only=False)
    if s_.get("timing") and not TEST:
        raise SystemExit("refusing to resume production training from a timing / test checkpoint")
    for k_, v_ in RECIPE.items():
        if s_["recipe"].get(k_) != v_:
            raise SystemExit(f"incompatible resume: recipe {k_} {s_['recipe'].get(k_)} != {v_}")
    for k_, v_ in (("code_sha", CODE_SHA), ("base_sha", BASE_SHA), ("vae_sha", VAE_SHA), ("data_sha", DATA_SHA), ("guide_version", GUIDE_VERSION)):
        if s_.get(k_) != v_:
            raise SystemExit(f"incompatible resume: {k_} {s_.get(k_)} != {v_}")
    model.load_trainable(s_["trainable"]); opt.load_state_dict(s_["opt"])
    start, CHAIN = s_["step"], s_["pick_chain"]
    OF0 = s_.get("overfit_eval0")
    if a.overfit and OF0 is None:
        raise SystemExit("resumed overfit run without its update-0 eval loss in the checkpoint")
    torch.set_rng_state(s_["rng"]["torch"]); torch.cuda.set_rng_state(s_["rng"]["cuda"]); random.setstate(s_["rng"]["python"]); np.random.set_state(s_["rng"]["numpy"])
    print(f"resumed at update {start}; ledger {LEDGER.hours():.2f} h used", flush=True)
json.dump({**vars(a), "kind": KIND, "code_sha": CODE_SHA, "base_sha": BASE_SHA, "vae_sha": VAE_SHA, "data_sha": DATA_SHA, "init_sha": INIT_SHA,
           "guide_version": GUIDE_VERSION, "mid_clips": len(MID), "start_clips": len(START), "clips_per_drive": PER_DRIVE,
           "drives_without_clips": no_clip, "trainable_lora_M": round(n_lora / 1e6, 2), "trainable_new_M": round(n_new / 1e6, 3),
           "init_diagnostics": DIAG, "cfg": CFG, "ledger_hours_at_start": LEDGER.hours()}, open(f"{a.out}/args.json", "w"), indent=1)


def save(step, path):
    d = {"trainable": model.trainable_state(), "opt": opt.state_dict(), "step": step, "hours": LEDGER.hours(), "recipe": RECIPE, "cfg": CFG,
         "code_sha": CODE_SHA, "base_sha": BASE_SHA, "vae_sha": VAE_SHA, "data_sha": DATA_SHA, "guide_version": GUIDE_VERSION,
         "init_sha": INIT_SHA, "pick_chain": CHAIN, "timing": bool(TEST), "args": vars(a), "overfit_eval0": OF0,
         "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}}
    torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)


def to_dev(bt):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in bt.items()}


def flow_loss(bt, sl, reduce=True):
    x0, eps, sig = bt["key"][sl], bt["eps"][sl], bt["sigma"][sl]
    s5 = sig.view(-1, 1, 1, 1, 1)
    xt = (1 - s5) * x0 + s5 * eps
    extra = torch.cat([bt["guide"][sl], bt["src"][sl]], 1)
    tok = bt["tok"][sl] if bt["tok"] is not None else None
    ray = bt["ray"][sl] if bt["ray"] is not None else None
    with torch.autocast("cuda", dtype=AMP):
        v = model(xt, extra, sig * 1000.0, tok, ray, bt["tkeep"][sl])
    err = (v.float() - (eps - x0)) ** 2
    return err.mean() if reduce else err.mean(dim=(1, 2, 3, 4))


# ------------------------------------------------------------------------------------------- dev probe (fixed)
PROBE = None
if a.val_every and all(os.path.exists(f"{a.cache}/{s}.pt") for s in DEVL) and (not a.use_tokens or all(os.path.exists(f"{a.tokens}/{s}.pt") for s in DEVL)):
    C_DV, T_DV, _ = load_split(DEVL, False)  # also enforces ONE guide version for train and dev
    mid_d, start_d, _ = candidates(C_DV)
    gp = torch.Generator().manual_seed(4242)
    by_drive = {}
    for d in mid_d:
        by_drive.setdefault(d[0], []).append(d)
    drv = sorted(by_drive)
    sel = [("mid", by_drive[drv[i % len(drv)]][int(torch.randint(0, len(by_drive[drv[i % len(drv)]]), (1,), generator=gp))]) for i in range(8)]
    sel += [("start", start_d[int(i)]) for i in torch.randperm(len(start_d), generator=gp)[:2].tolist()] if start_d else []
    donors = []
    for k, d in sel:
        pool = [x for x in (mid_d if k == "mid" else start_d) if x[0] != d[0]]
        donors.append((k, pool[int(torch.randint(0, len(pool), (1,), generator=gp))]))
    def _stack(items):
        cl = [clip(C_DV, T_DV, d, k) for k, d in items]
        return [torch.stack([c[i] for c in cl]) if cl[0][i] is not None else None for i in range(5)]
    PK, PG, PM, PT, PR = _stack(sel); _, DG, DM, DT, DR = _stack(donors)
    PROBE = {"key": PK, "guide": PG, "src": PM, "tok": PT, "ray": PR, "dguide": DG, "dsrc": DM, "dtok": DT, "dray": DR,
             "eps": torch.randn(len(sel), 16, NL, 32, 64, generator=gp)}
    print(f"dev probe: {len(sel)} fixed clips from {len({d[0] for _, d in sel})} dev drives", flush=True)
    del C_DV, T_DV


@torch.no_grad()
def probe(step):
    model.eval(); res = {}
    n = PROBE["key"].shape[0]; on = torch.ones(n, dtype=torch.bool); off = ~on
    zg, zm = torch.zeros_like(PROBE["guide"]), torch.zeros_like(PROBE["src"])
    modes = {"hybrid": (PROBE["guide"], PROBE["src"], PROBE["tok"], PROBE["ray"], on),
             "guide_only": (PROBE["guide"], PROBE["src"], PROBE["tok"], PROBE["ray"], off),
             "tokens_only": (zg, zm, PROBE["tok"], PROBE["ray"], on), "none": (zg, zm, PROBE["tok"], PROBE["ray"], off),
             "other_drive": (PROBE["dguide"], PROBE["dsrc"], PROBE["dtok"], PROBE["dray"], on)}
    for sv in (0.3, 0.6, 0.9):
        res[str(sv)] = {}
        for name, (g_, m_, t_, r_, kp) in modes.items():
            bt = to_dev({"key": PROBE["key"], "eps": PROBE["eps"], "sigma": torch.full((n,), sv), "guide": g_, "src": m_, "tok": t_, "ray": r_, "tkeep": kp})
            ls = [flow_loss(bt, slice(i, i + a.micro), reduce=False).cpu() for i in range(0, n, a.micro)]
            res[str(sv)][name] = round(torch.cat(ls).mean().item(), 5)
    bt = to_dev({"key": PROBE["key"][:2], "eps": PROBE["eps"][:2], "guide": PROBE["guide"][:2], "src": PROBE["src"][:2],
                 "tok": PROBE["tok"][:2] if PROBE["tok"] is not None else None, "ray": PROBE["ray"][:2] if PROBE["ray"] is not None else None})
    xt = 0.4 * bt["key"] + 0.6 * bt["eps"]
    sens = WC.sensitivity(model, xt, torch.cat([bt["guide"], bt["src"]], 1), torch.full((2,), 600.0, device=dev), bt["tok"], bt["ray"], AMP)
    model.train()
    ok_s, why_s = WC.check_sensitivity(sens, bool(a.use_tokens))
    rec = {"update": step, "probe": res, "sensitivity_sigma0.6": sens, "sensitivity_pass": ok_s, "sensitivity_why": why_s, "code_sha": CODE_SHA}
    print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()


log = open(f"{a.out}/log.jsonl", "a")


# ------------------------------------------------------------------------------------------------- one update
def update(step, bt, timing=False):
    """forward / backward (accumulated) / clip on one global batch (Adam is the caller's). Returns loss, grad norm, total s and,
    with timing=True, separately synchronised forward and backward seconds (summed over microbatches)."""
    for g_ in opt.param_groups:
        g_["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
    torch.cuda.synchronize(); t0 = time.time(); tf = tb = 0.0
    opt.zero_grad(set_to_none=True)
    nacc = a.batch // a.micro; loss_v = 0.0
    for i in range(nacc):
        if timing:
            torch.cuda.synchronize(); ta = time.time()
        l_ = flow_loss(bt, slice(i * a.micro, (i + 1) * a.micro)) / nacc
        if timing:
            torch.cuda.synchronize(); tb_ = time.time(); tf += tb_ - ta
        l_.backward()
        if timing:
            torch.cuda.synchronize(); tb += time.time() - tb_
        loss_v += l_.item()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    if not torch.isfinite(gn):
        raise SystemExit(f"non-finite grad norm at update {step + 1}")
    torch.cuda.synchronize(); t1 = time.time()
    return (loss_v, gn, t1 - t0, tf, tb) if timing else (loss_v, gn, t1 - t0)


def start_producer(s0, s1):
    """the production prefetch: one thread assembles + pins batches s0..s1-1 into a 3-deep queue"""
    Q_ = queue.Queue(maxsize=3)

    def run():
        try:
            for s in range(s0, s1):
                t0 = time.time(); bt = make_batch(s)
                for k in ("key", "guide", "src", "eps", "tok", "ray"):
                    if bt[k] is not None:
                        bt[k] = bt[k].pin_memory()
                bt["data_s"] = time.time() - t0
                Q_.put((s, bt))
        except BaseException as e:  # surface data errors instead of hanging on Q.get()
            Q_.put((-1, e))
    threading.Thread(target=run, daemon=True).start()
    return Q_


def qget(Q_, step):
    s_, bt = Q_.get()
    if s_ == -1:
        raise SystemExit(f"data producer failed: {bt!r}")
    assert s_ == step, (s_, step)
    return bt


# ------------------------------------------------------------------------------------------------- timing mode
if a.timing and not a.overfit:
    rep = {"gpu": torch.cuda.get_device_name(), "batch": a.batch, "tok_ticks": a.tok_ticks, "use_tokens": bool(a.use_tokens), "micro": {},
           "startup_s": round(time.time() - LEDGER.t0, 1)}  # process start -> ready to train (loads, hashes, parity)
    s_ = 0
    # (1) microbatch sweep at global batch 8 (synchronous batches; forward / backward / Adam timed separately)
    for mb in [int(x) for x in a.timing_micro.split(",") if a.batch % int(x) == 0]:
        a.micro = mb; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        rows = []
        try:
            for k in range(6):  # 2 warm + 4 timed
                tw = time.time(); bt = to_dev(make_batch(s_)); torch.cuda.synchronize(); tdata = time.time() - tw
                loss_v, gn, tfb, tf, tb = update(s_, bt, timing=True)
                ta = time.time(); opt.step(); torch.cuda.synchronize(); tad = time.time() - ta
                s_ += 1
                if k >= 2:
                    rows.append({"sync_batch_s": tdata, "fwd_s": tf, "bwd_s": tb, "fwd_bwd_clip_s": tfb, "adam_s": tad, "gpu_s": tfb + tad})
            rep["micro"][mb] = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}
            rep["micro"][mb].update(peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
        except torch.cuda.OutOfMemoryError as e:
            rep["micro"][mb] = {"oom": str(e)[:200]}
        if "oom" in rep["micro"][mb]:  # free outside the except block (its traceback pins the failed step's tensors)
            import gc
            bt = None; opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
            print("TIMING micro", mb, "OOM", flush=True); break
        print("TIMING micro", mb, rep["micro"][mb], flush=True)
    fits = {k: v for k, v in rep["micro"].items() if "gpu_s" in v}
    if not fits:
        raise SystemExit(f"no microbatch fits: {rep}")
    best = min(fits, key=lambda k: fits[k]["gpu_s"]); rep["best_micro"] = best
    a.micro = best; torch.cuda.empty_cache()  # S1b item 4: everything below runs at the fastest FITTING microbatch, not the OOM one
    # (2) production loop: the real prefetch thread + queue, best microbatch; queue wait = time blocked on Q.get
    Qp = start_producer(s_, s_ + 10); torch.cuda.reset_peak_memory_stats(); prow = []
    for k in range(10):
        t0 = time.time(); bt = qget(Qp, s_); tw = time.time() - t0
        bt = to_dev(bt); loss_v, gn, tfb = update(s_, bt); opt.step(); torch.cuda.synchronize()
        if k >= 2:
            prow.append({"queue_wait_s": tw, "producer_assembly_s": bt["data_s"], "wall_s": time.time() - t0})
        s_ += 1
    rep["production_loop"] = {k: round(float(np.mean([r[k] for r in prow])), 3) for k in prow[0]}
    rep["production_loop"].update(micro=best, updates=len(prow), peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                  peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
    print("TIMING production loop", rep["production_loop"], flush=True)
    t = time.time(); probe(0) if PROBE else None; rep["probe_s"] = round(time.time() - t, 2)
    t = time.time(); save(s_, f"{a.out}/ckpt.pt"); rep["save_s"] = round(time.time() - t, 2)
    # (3) rollout pieces: one 30-step window (5 latents) at 1 and 2 rows, VAE encode / decode, SD encode
    model.eval(); bt = to_dev(make_batch(10 ** 6))
    for rows_ in (1, 2):
        z = torch.randn(rows_, 16, NL, 32, 64, device=dev); ex = torch.cat([bt["guide"][:rows_], bt["src"][:rows_]], 1)
        tk = bt["tok"][:rows_] if bt["tok"] is not None else None; ry = bt["ray"][:rows_] if bt["ray"] is not None else None
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
            for i in range(30):
                model(z, ex, torch.full((rows_,), 1000.0 * (1 - i / 30), device=dev), tk, ry, torch.ones(rows_, dtype=torch.bool, device=dev))
        torch.cuda.synchronize(); rep[f"window30_rows{rows_}_s"] = round(time.time() - t, 2)
    vae.to(dev)
    v17 = torch.rand(1, 3, 17, 256, 512, device=dev) * 2 - 1
    torch.cuda.synchronize(); t = time.time(); zz = WC.wan_encode(vae, v17); torch.cuda.synchronize(); rep["wan_encode_17f_s"] = round(time.time() - t, 3)
    t = time.time(); WC.wan_decode(vae, zz); torch.cuda.synchronize(); rep["wan_decode_5lat_s"] = round(time.time() - t, 3)
    from diffusers import AutoencoderKL
    sdv = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True).to(dev).eval()
    t = time.time()
    with torch.no_grad():
        sdv.encode(v17[0].permute(1, 0, 2, 3)).latent_dist.mode()
    torch.cuda.synchronize(); rep["sd_encode_17f_s"] = round(time.time() - t, 3)
    rem = a.budget_hours - LEDGER.hours() - a.reserve_hours
    wall = rep["production_loop"]["wall_s"]
    ups = rem * 3600 / wall
    rep["forecast"] = {"basis": "production-loop wall (prefetch queue wait included); provisional until slurm timing adds scorer and deductions",
                       "remaining_hours_after_reserve": round(rem, 2), "updates_in_remaining": int(ups), "floor_12k_reachable": ups >= WP.FLOOR_UPDATES,
                       "max_s_per_update_for_12k": round(rem * 3600 / WP.FLOOR_UPDATES, 2), "ledger_by_kind": LEDGER.by_kind()}
    json.dump(rep, open(f"{a.out}/timing.json", "w"), indent=1)
    print("WAN_TIMING", json.dumps(rep), flush=True)
    LEDGER.stop(); raise SystemExit(0)


# ------------------------------------------------------------------------------------------------ overfit eval
if a.overfit:
    K_, G_, M_, T_, R_, OF_SPANS = assemble(OF)
    OFB = {"key": K_, "guide": G_, "src": M_, "tok": T_, "ray": R_, "eps": torch.randn(a.overfit, 16, NL, 32, 64, generator=torch.Generator().manual_seed(999)),
           "tkeep": torch.ones(a.overfit, dtype=torch.bool)}

    @torch.no_grad()
    def overfit_eval(step):
        model.eval(); ls = []
        for sv in (0.3, 0.6, 0.9):
            bt = to_dev({**OFB, "sigma": torch.full((a.overfit,), sv)})
            ls.append(torch.cat([flow_loss(bt, slice(i, i + a.micro), reduce=False).cpu() for i in range(0, a.overfit, a.micro)]).mean().item())
        model.train()
        rec = {"update": step, "overfit_eval": [round(x, 5) for x in ls], "mean": round(float(np.mean(ls)), 5)}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        return float(np.mean(ls))

# ------------------------------------------------------------------------------------------------ prefetch + loop
end = a.steps if not a.max_steps_this_job else min(a.steps, start + a.max_steps_this_job)
Q = start_producer(start, end)
SAVE = set(int(x) for x in a.save_at.split(","))
T_WIN, N_WIN = time.time(), 0
acc = {"data": 0.0, "gpu": 0.0, "wait": 0.0, "n": 0}
stopped, last_wall = None, None
if a.overfit and start == 0:
    OF0 = overfit_eval(0)  # persisted in every checkpoint: a resumed run keeps the update-0 baseline
for step in range(start, end):
    if KIND == "train":  # remaining-hours check BEFORE the update (item 11)
        used = LEDGER.hours(); est = (last_wall or 0.0) / 3600
        if used + est + a.reserve_hours >= a.budget_hours:
            if step > start:
                save(step, ck)
            stopped = step
            rec = {"budget_stop": step, "ledger_hours": round(used, 3), "reserve": a.reserve_hours, "budget_hours": a.budget_hours, "by_kind": LEDGER.by_kind()}
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            break
    tstep = time.time()
    tw = time.time(); bt = qget(Q, step)
    acc["wait"] += time.time() - tw; acc["data"] += bt["data_s"]
    CHAIN = hashlib.sha1((CHAIN + "|" + bt["sha"]).encode()).hexdigest()
    bt = to_dev(bt)
    if step == start:
        FROZEN_REF = model.frozen(); FROZEN_PRE = [q.detach().cpu().clone() for _, q in FROZEN_REF]
        PRE_NEW = {n: q.detach().clone() for n, q in TR if "lora_" not in n or "lora_B" in n}
        VAE_PRE = WC.vae_sha(vae)
    loss_v, gn, tfb = update(step, bt)
    if step == 1 and a.use_tokens:  # gates opened at update 1: the token path must now carry gradient everywhere (item 5)
        cur = dict(TR)
        pj = [n for n in cur if n.startswith("proj.") and cur[n].grad is not None and cur[n].grad.abs().sum() > 0]
        qk = [i for i in range(len(model.tf.blocks)) if all(
            cur[f"tf.blocks.{i}.attn2.inner.{t}.lora_B.default.weight"].grad is not None and
            cur[f"tf.blocks.{i}.attn2.inner.{t}.lora_B.default.weight"].grad.abs().sum() > 0 for t in ("to_q", "to_k"))]
        n_pj = sum(1 for n in cur if n.startswith("proj."))
        assert len(pj) == n_pj and len(qk) == len(model.tf.blocks), f"token path gradients after the gate opened: projector {len(pj)}/{n_pj}, attn2 q/k LoRA {len(qk)}/30 blocks"
        print(f"gate-open check (update 2): projector {len(pj)}/{n_pj} tensors and attn2 q/k LoRA-B in {len(qk)}/30 blocks have gradients", flush=True)
    opt.step()
    if step == start:  # first-update guard (item 5)
        assert all(torch.equal(q.detach().cpu(), q0) for (_, q), q0 in zip(FROZEN_REF, FROZEN_PRE)), "frozen base transformer changed!"
        assert WC.vae_sha(vae) == VAE_PRE == VAE_SHA, "VAE changed!"
        if step == 0:
            cur = dict(TR); moved = {n: not torch.equal(cur[n], q0) for n, q0 in PRE_NEW.items()}
            must = [n for n in moved if ".patch_embedding.extra." in n or (n.endswith(".attn2.gate") and a.use_tokens)
                    or ("lora_B" in n and not any(x in n for x in (".attn2.inner.to_q.", ".attn2.inner.to_k.")))]
            bad = [n for n in must if not moved[n]]
            n_b = sum(1 for n in must if "lora_B" in n); n_g = sum(1 for n in must if n.endswith(".attn2.gate"))
            assert n_b == 8 * 30 and (n_g == 30 or not a.use_tokens), f"guard coverage: {n_b} LoRA-B, {n_g} gates"
            assert not bad, f"{len(bad)} tensors did not move in update 1 (all inputs active), e.g. {bad[:3]}"
            print(f"guard after update 1: {len(FROZEN_REF)} base tensors + VAE bit-identical; extra conv, {n_g} gates, {n_b} LoRA-B moved", flush=True)
        else:
            print(f"guard after resumed update {step + 1}: {len(FROZEN_REF)} base tensors + VAE bit-identical", flush=True)
        del FROZEN_PRE, PRE_NEW
    acc["gpu"] += tfb; acc["n"] += 1
    if (step + 1) % 100 == 0 or step == start or step == end - 1 or TEST:
        gates = [m.gate for m in model.xattn]
        rec = {"update": step + 1, "loss": round(loss_v, 5), "gn": round(gn.item(), 4), "lr": opt.param_groups[0]["lr"],
               "sigma_mean": round(bt["sigma"].mean().item(), 3), "n_start": bt["n_start"], "gdrop": bt["gdrop"], "tdrop": bt["tdrop"],
               "step_s": round(acc["gpu"] / acc["n"], 3), "data_s": round(acc["data"] / acc["n"], 3), "wait_s": round(acc["wait"] / acc["n"], 3),
               "e2e_s": round((time.time() - T_WIN) / max(1, step + 1 - start - N_WIN), 3),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 1),
               "gate_norm": round(sum(g.norm().item() ** 2 for g in gates) ** 0.5, 4), "extra_norm": round(model.tf.patch_embedding.extra.weight.norm().item(), 4),
               "batch_sha": bt["sha"], "pick_chain": CHAIN[:12], "data_sha": DATA_SHA, "ledger_hours": round(LEDGER.hours(), 3), "code_sha": CODE_SHA}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        acc = {"data": 0.0, "gpu": 0.0, "wait": 0.0, "n": 0}; T_WIN, N_WIN = time.time(), step + 1 - start
    if step + 1 in SAVE and not a.overfit:  # numbered milestone FIRST, then the rolling file
        save(step + 1, f"{a.out}/ckpt_{step + 1}.pt")
    if (step + 1) % a.save_every == 0 or step + 1 == end:
        save(step + 1, ck)
    if PROBE is not None and not a.overfit and ((step + 1) % a.val_every == 0 or step == start):
        probe(step + 1)
    if a.overfit and (step + 1) % 100 == 0:
        overfit_eval(step + 1)
    last_wall = time.time() - tstep

# ------------------------------------------------------------------------------------------------ overfit verdict
if a.overfit and end < OVERFIT_STEPS:
    print(f"OVERFIT_PENDING at update {end} (gates are read at {OVERFIT_STEPS}); resume with --resume", flush=True)
    LEDGER.stop(); raise SystemExit(0)
if a.overfit:
    OFN = overfit_eval(end)
    from mrv import guide as GD
    vae.to(dev)
    model.eval(); bt = to_dev({**OFB})
    z = torch.randn(a.overfit, 16, NL, 32, 64, generator=torch.Generator().manual_seed(31337)).to(dev)
    sg = WC.sigma_shift(torch.linspace(1.0, 0.0, 31), a.shift).to(dev)
    ex = torch.cat([bt["guide"], bt["src"]], 1)
    with torch.no_grad():  # sampled in the fitting microbatch size (S1b item 6)
        for i0 in range(0, a.overfit, a.micro):
            sl = slice(i0, i0 + a.micro); zz = z[sl]
            for i in range(30):
                with torch.autocast("cuda", dtype=AMP):
                    v = model(zz, ex[sl], (sg[i] * 1000).expand(zz.shape[0]), bt["tok"][sl] if bt["tok"] is not None else None,
                              bt["ray"][sl] if bt["ray"] is not None else None, bt["tkeep"][sl]).float()
                zz = zz + (sg[i + 1] - sg[i]) * v
            z[sl] = zz
    NS = GD.import_signed_scorer(a.out)  # signed scorer namespace: segment / iou_mat / hungarian (unchanged file)
    tp = {"sample": [0, 0, 0], "roundtrip": [0, 0, 0]}; strips = []
    for b, ((kind, d), (sc, spans)) in enumerate(zip(OF, OF_SPANS)):
        c = C_TR[sc]
        if kind == "mid":  # decode with the stream's true preceding latents as causal context, keep the clip's 20 frames
            st = c["streams"][d[1]]; j0 = d[2]; ctx = st["key"][:, :j0].float().to(dev)[None]
            dec = lambda zc: WC.wan_decode(vae, torch.cat([ctx, zc], 2))[0][:, 4 * j0 - 3:4 * j0 + 4 * NL - 3]
            frames = [i for sp in spans for i in sp]
        else:
            dec = lambda zc: WC.wan_decode(vae, zc)[0]
            frames = [spans[0][0]] + [i for sp in spans[1:] for i in sp]
        xs, xr = dec(z[b:b + 1]), dec(bt["key"][b:b + 1])
        for q, i in enumerate(frames):
            kimg = GD.key_image(a.keys, sc, int(c["ticks"][i]))
            kc = NS["segment"](kimg)[0]
            for name, xx in (("sample", xs), ("roundtrip", xr)):
                from PIL import Image
                im = Image.fromarray(((xx[:, q] + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())
                rc = NS["segment"](im)[0]
                mm = NS["hungarian"](NS["iou_mat"](rc, kc), 0.3)
                tp[name][0] += len(mm); tp[name][1] += len(rc) - len(mm); tp[name][2] += len(kc) - len(mm)
                if q == len(frames) // 2 and name == "sample":
                    strips.append(np.concatenate([np.asarray(im), np.asarray(kimg)], 1))
    f1 = {k: 2 * v[0] / max(1, 2 * v[0] + v[1] + v[2]) for k, v in tp.items()}
    loss_ok = OFN <= a.overfit_ratio * OF0
    rt_ok = f1["roundtrip"] > 0 and tp["roundtrip"][0] + tp["roundtrip"][2] > 0  # key cars exist and the codec keeps some
    f1_ok = rt_ok and f1["sample"] >= a.overfit_f1_frac * f1["roundtrip"]
    res = {"eval_loss_update0": OF0, "eval_loss_final": OFN, "ratio": OFN / OF0, "declared_ratio": a.overfit_ratio, "loss_pass": loss_ok,
           "car_f1_sample": f1["sample"], "car_f1_roundtrip": f1["roundtrip"], "declared_f1_frac": a.overfit_f1_frac, "f1_pass": f1_ok,
           "roundtrip_positive": rt_ok, "counts": tp, "clips": [(k, list(d)) for k, d in OF], "updates": end,
           "declared_updates": OVERFIT_STEPS, "PASS": bool(end == OVERFIT_STEPS and loss_ok and f1_ok)}
    from PIL import Image
    Image.fromarray(np.concatenate(strips, 0)).save(f"{a.out}/overfit_samples.png")
    json.dump(res, open(f"{a.out}/overfit.json", "w"), indent=1)
    print("OVERFIT_GATE", json.dumps(res), flush=True)
    LEDGER.stop(); raise SystemExit(0 if res["PASS"] else 4)
LEDGER.stop()
print("WAN_TEACHER_BUDGET_STOP" if stopped is not None else "WAN_TEACHER_DONE", stopped if stopped is not None else end)
