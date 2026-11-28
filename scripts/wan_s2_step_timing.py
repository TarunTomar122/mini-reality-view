"""S2 groundwork: time ONE causal student forward / backward over a 5-plane window with the K/V cache (mrv/wan_causal.py), from an
S1 checkpoint, on the H200 slice. NOT training: no loop, no loss choice, no optimizer unless --adam 1. The scalar it
backpropagates is a placeholder (mean square of the exit predictions) that only exercises the graph; council has not chosen
the S2 loss (DMD + key term / critic details / warm start are open).

Per plane (block = one latent plane; plane 0 = tick 0, then 4 ticks): 4 denoising substeps (--substeps, x 1000 = sigma; default
Fable's 1.0 / 0.857 / 0.667 / 0.4) on the plane's scratch K/V; gradient through ONE sampled substep per plane (the others under
no_grad), history K/V detached (Self-Forcing pattern, both seats); then the final latent is re-encoded at --t_commit (0 =
clean) and appended to the cache once. Inputs have production shapes (latent 16 x 32 x 64, guide 24 channels, tokens
[slots, 6 views, 777, 384] + rays) and are random unless --data is added later: the cost does not depend on their values.
Records forward (rollout) / backward / optional Adam seconds, peak allocated / reserved memory, per repetition, to --out;
charged to its own ledger (--ledger, default runs/wan_s2_ledger.jsonl; S2 budget, never the S1 ledger).
  sbatch -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=0:45:00 -J wan-s2t -o logs2/%x-%j.out \
         slurm/py.sbatch scripts/wan_s2_step_timing.py --ckpt runs/wan_s1/ckpt.pt --out runs/wan_s2_timing_$SLURM_JOB_ID/timing.json
  python scripts/wan_s2_step_timing.py --tiny 1 --device cpu --out /tmp/s2t.json      # CPU self-test of the code path
"""
import argparse, hashlib, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=""); p.add_argument("--out", required=True)
p.add_argument("--planes", type=int, default=5); p.add_argument("--batch", type=int, default=1)
p.add_argument("--substeps", default="1000,857,667,400"); p.add_argument("--t_commit", type=float, default=0.0)
p.add_argument("--reps", type=int, default=3); p.add_argument("--warm", type=int, default=1)
p.add_argument("--adam", type=int, default=0, help="1: also time one AdamW step on the trainable parameters")
p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"]); p.add_argument("--device", default="cuda")
p.add_argument("--ledger", default="runs/wan_s2_ledger.jsonl"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--tiny", type=int, default=0, help="CPU self-test: tiny random WanCond instead of the S1 checkpoint")
a = p.parse_args()
from mrv import wan_cond as WC, wan_causal as WK
dev = a.device
AMP = torch.bfloat16 if a.dtype == "bf16" else torch.float32
SUB = [float(x) for x in a.substeps.split(",")]
rng = random.Random(a.seed)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


if a.tiny:  # same structure as scripts/wan_causal_tests.py
    from diffusers import WanTransformer3DModel
    tf = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16, out_channels=16,
                               text_dim=WC.TEXT_DIM, freq_dim=32, ffn_dim=48, num_layers=3, cross_attn_norm=True,
                               qk_norm="rms_norm_across_heads", rope_max_seq_len=64)
    model = WC.WanCond(tf, tok_ticks="last", use_tokens=True, prompt=None)
    model.proj = WC.TokenProjector(1, d_tok=8, d=16, d_out=WC.TEXT_DIM, n_views=2)
    for n_, q in model.named_parameters():
        q.requires_grad_(n_.startswith("proj.") or ".patch_embedding.extra." in n_ or n_.endswith(".attn2.gate") or ".attn1." in n_)
    model = model.to(dev)
    HH, WW, S, V, P, D = 8, 8, 1, 2, 3, 8
    CK_SHA = None
else:
    if not a.ckpt:
        raise SystemExit("--ckpt (an S1 checkpoint) is required unless --tiny 1")
    import io
    raw = open(a.ckpt, "rb").read(); CK_SHA = hashlib.sha256(raw).hexdigest()[:16]
    ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False); del raw
    CFG = ck["cfg"]
    prompt = torch.load(CFG["prompt_embed"], map_location="cpu") if CFG.get("prompt_embed") else None
    model = WC.build(CFG["wan"], rank=CFG["rank"], tok_ticks=CFG["tok_ticks"], use_tokens=CFG["use_tokens"], dtype=torch.bfloat16, device=dev,
                     prompt=prompt, grad_ckpt=False, seed=CFG.get("seed", 0))
    model.load_trainable(ck["trainable"])
    HH, WW, S, V, P, D = 32, 64, WC.SLOTS[CFG["tok_ticks"]], WC.N_VIEWS, WC.TOK_P, WC.TOK_DIM
model.train()
cw = WK.CausalWan(model)
TR = [q for q in model.parameters() if q.requires_grad]
opt = torch.optim.AdamW(TR, lr=1e-5) if a.adam else None
from mrv import wan_plan as WP
LEDGER = WP.Ledger(a.ledger, "s2_timing").start()
B = a.batch


def sync():
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def plane_inputs(g):
    extra = torch.randn(B, 24, 1, HH, WW, generator=g).to(dev)
    tok = torch.randn(B, 1, S, V, P, D, generator=g).half().to(dev) if model.use_tokens else None
    ray = torch.randn(B, 1, S, V, P, 6, generator=g).to(dev) if model.use_tokens else None
    return extra, tok, ray, torch.ones(B, dtype=torch.bool, device=dev)


def one_window(seed):
    """the 5-plane own-history rollout with one gradient-bearing substep per plane, then the placeholder backward"""
    g = torch.Generator().manual_seed(seed)
    cw.reset(); model.zero_grad(set_to_none=True)
    exits, outs = [], []
    sync(); t0 = time.time()
    for _p in range(a.planes):
        extra, tok, ray, keep = plane_inputs(g)
        z = torch.randn(B, 16, 1, HH, WW, generator=g).to(dev)
        e = rng.randrange(len(SUB)); exits.append(e)
        for k, tk in enumerate(SUB):
            s_k, s_n = tk / 1000.0, (SUB[k + 1] / 1000.0 if k + 1 < len(SUB) else 0.0)
            with torch.set_grad_enabled(k == e), torch.autocast("cuda" if dev.startswith("cuda") else "cpu", dtype=AMP, enabled=AMP != torch.float32):
                v = cw.step(z, extra, torch.full((B,), tk, device=dev), tok, ray, keep).float()
            if k == e:
                outs.append(z - s_k * v)          # the exit's x0 prediction (graph kept)
            z = (z + (s_n - s_k) * v).detach()    # next substep from the detached state
        with torch.no_grad(), torch.autocast("cuda" if dev.startswith("cuda") else "cpu", dtype=AMP, enabled=AMP != torch.float32):
            cw.commit(z, extra, torch.full((B,), a.t_commit, device=dev), tok, ray, keep)   # clean re-encode, one append
    sync(); t_fwd = time.time() - t0
    placeholder = sum(o.float().pow(2).mean() for o in outs)  # NOT a loss choice: only drives the backward for timing
    t1 = time.time(); placeholder.backward(); sync(); t_bwd = time.time() - t1
    t_adam = None
    if opt is not None:
        t2 = time.time(); opt.step(); sync(); t_adam = time.time() - t2
    return {"forward_rollout_s": round(t_fwd, 3), "backward_s": round(t_bwd, 3), "adam_s": round(t_adam, 3) if t_adam is not None else None,
            "exits": exits, "appends": cw.appends()[0]}


rows = []
for r in range(a.warm + a.reps):
    if dev.startswith("cuda"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    row = one_window(a.seed * 1000 + r)
    if dev.startswith("cuda"):
        row.update(peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
    row["warm"] = r < a.warm
    rows.append(row); print("S2_TIMING rep", r, row, flush=True)
timed = [x for x in rows if not x["warm"]]
mean = lambda k: round(sum(x[k] for x in timed) / len(timed), 3) if timed and timed[0].get(k) is not None else None
rep = {"device": torch.cuda.get_device_name() if dev.startswith("cuda") else dev, "ckpt": a.ckpt or "tiny", "ckpt_sha16": CK_SHA,
       "planes": a.planes, "batch": B, "substeps": SUB, "t_commit": a.t_commit, "window": WK.WINDOW, "dtype": a.dtype,
       "plane_tokens": (HH // 2) * (WW // 2), "trainable_params_M": round(sum(q.numel() for q in TR) / 1e6, 3),
       "mean": {k: mean(k) for k in ("forward_rollout_s", "backward_s", "adam_s", "peak_alloc_gb", "peak_reserved_gb")},
       "rows": rows, "code_sha": {"wan_causal": sha("mrv/wan_causal.py"), "timing": sha(__file__)},
       "note": "placeholder backward (mean square of exit predictions): cost / memory only; no S2 loss chosen"}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(rep, open(a.out, "w"), indent=1)
LEDGER.stop()
print("WAN_S2_TIMING", json.dumps(rep["mean"]), flush=True)
