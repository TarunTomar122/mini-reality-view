"""Acceptance tests for the clean-memory adapter "vclean" (council/AGREED-causal-r1.md, Amendment A4: "T1-T7 re-checked for
the new ring"). Exits 1 if any test fails. T2 / T5 / T6 use scripts/clean_rollout.py (separate slurm runs, as for vlive).

  part 0 (pure torch, CPU, always): one CleanTemporalBlock on random features
    T0  same parameter names / shapes / seeded init as mrv/causal_video.CausalTemporalBlock (vpriv's init; vpriv weights load)
    T1  fresh block: clip / stream read / stream stage == input exactly; perturbed: clip tick 0, an empty ring and every stage
        call still == input exactly; later rows differ
    T3a clip mode: changing inputs + memory of ticks >= j never changes outputs < j (and changes >= j); the memory of tick j
        only reaches ticks j+1 .. j+K (not tick j itself, not j+K+1)
    T4  stream == clip (4 rows = 2 streams x 2 CFG rows, 2 read calls per tick + 1 stage call, 12 ticks: through eviction);
        reset at tick 7: first tick after the reset == input exactly, later ticks == a clip starting at the cut, and differ
        from the unreset stream; one stream alone == interleaved; a cond-only ring slice (load_ring rows=[0]) == row 0
  part 1 (needs diffusers): the same T1 / T3a / T4 through a whole UNet with collect_inputs memory; --small = a tiny random
        UNet on CPU (fp32), else --init = A@20k on CUDA (fp32, TF32 off)
  part 2 (--t7, CUDA + data): T7 frozen guard via scripts/train_clean.py --timing (LoRA + adapter bit-identical after a real
        step, 16/16 out weights moved, t* reported), then --resume compatibility (one more update from the saved checkpoint)
  The authoritative suite is slurm/clean_tests.sh (this file --small and --init, T5 / T6 rollouts, the full T7 resume / recipe
  checks and the 100-update timing record); T2 is slurm/clean_t2.sh.

  python scripts/clean_tests.py --small                     # CPU
  python scripts/clean_tests.py --init runs/longA/ckpt_20000.pt --t7   # A6000
"""
import argparse, copy, os, subprocess, sys
import torch
from mrv.clean_video import (CleanTemporalBlock, add_clean_temporal, set_mode, set_memory, set_stage, publish, reset,
                             collect_inputs, ring_state, load_ring)

p = argparse.ArgumentParser()
p.add_argument("--init", default="", help="A@20k checkpoint: full-model part 1 on CUDA (and T7)")
p.add_argument("--small", action="store_true", help="part 1 on a tiny random UNet on CPU")
p.add_argument("--t7", action="store_true", help="part 2: run scripts/train_clean.py --timing (needs --init, CUDA, data)")
p.add_argument("--t7_updates", type=int, default=2)
p.add_argument("--K", type=int, default=4)
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--rays", default="data/calib_rays.pt")
a = p.parse_args()
K = a.K
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
FAIL = []


def report(name, ok, msg):
    print(("PASS " if ok else "FAIL ") + name + ": " + msg, flush=True)
    if not ok:
        FAIL.append(name)


def perturb_params(named, seed, scale=0.05, dev="cpu"):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n_, q in named:
            if not n_.startswith(("norm", "ln")):  # keep the norms; everything else gets nonzero weights
                q.add_(torch.randn(q.shape, generator=g).to(dev) * scale)


mx = lambda u, v: (u.float() - v.float()).abs().max().item()


class _Hold:  # stands in for a wrapper (ring_state / load_ring take wrappers)
    def __init__(s, t):
        s.temporal = t

# ================= part 0: one block, pure torch, CPU fp32 =================
C0, H0, W0, F0, B0 = 64, 4, 8, 12, 4
from mrv.causal_video import CausalTemporalBlock  # read-only import: init / weight-layout compatibility with vpriv
torch.manual_seed(0); cb = CleanTemporalBlock(C0, K)
torch.manual_seed(0); zb = CausalTemporalBlock(C0, K)
sd1, sd2 = cb.state_dict(), zb.state_dict()
same = list(sd1) == list(sd2) and all(sd1[k].shape == sd2[k].shape and torch.equal(sd1[k], sd2[k]) for k in sd1)
try:
    cb2 = CleanTemporalBlock(C0, K); cb2.load_state_dict(sd2, strict=True); loads = True
except Exception as e_:
    loads = False
report("T0 init / layout == CausalTemporalBlock", same and loads, f"same keys, shapes and seeded values: {same}; strict load of causal weights: {loads}")

g0 = torch.Generator().manual_seed(1)
X0 = torch.randn(B0 * F0, C0, H0, W0, generator=g0); XB = torch.randn(B0 * F0, C0, H0, W0, generator=g0)
M0 = torch.randn(B0 * F0, C0, H0, W0, generator=g0)
rows_at = lambda i: [b * F0 + i for b in range(B0)]  # clip layout: B clips x F ticks, frame-major inside a clip
with torch.no_grad():
    d_fresh = mx(cb.forward_clip(X0, M0, F0), X0)
    cb.ring, cb.staged = [], None
    d_fresh = max(d_fresh, mx(cb.forward_stream(X0[rows_at(0)], True), X0[rows_at(0)])); cb.publish()
    d_fresh = max(d_fresh, mx(cb.forward_stream(X0[rows_at(1)], False), X0[rows_at(1)]))  # ring non-empty, fresh weights
    perturb_params(cb.named_parameters(), 2, 0.1)
    yc = cb.forward_clip(X0, M0, F0)
    tick0 = [b * F0 for b in range(B0)]
    d_t0 = mx(yc[tick0], X0[tick0])
    cb.ring, cb.staged = [], None
    d_empty = mx(cb.forward_stream(X0[rows_at(0)], False), X0[rows_at(0)])
    d_stage = mx(cb.forward_stream(M0[rows_at(0)], True), M0[rows_at(0)])
    later = [b * F0 + i for b in range(B0) for i in range(1, F0)]
    d_later = mx(yc[later], X0[later])
report("T1 block parity", d_fresh == 0.0 and d_t0 == 0.0 and d_empty == 0.0 and d_stage == 0.0 and d_later > 1e-3,
       f"fresh clip/stream {d_fresh:.3g} (=0); perturbed: clip tick 0 {d_t0:.3g}, empty ring {d_empty:.3g}, stage call {d_stage:.3g} (=0); later rows {d_later:.3g} (>0)")

with torch.no_grad():
    j = 6; fut = [b * F0 + i for b in range(B0) for i in range(j, F0)]; past = [b * F0 + i for b in range(B0) for i in range(j)]
    X2, M2 = X0.clone(), M0.clone()
    X2[fut] += torch.randn(X2[fut].shape, generator=g0); M2[fut] += torch.randn(M2[fut].shape, generator=g0)
    y2 = cb.forward_clip(X2, M2, F0)
    dpre, dpost = mx(y2[past], yc[past]), mx(y2[fut], yc[fut])
    j0 = 4; M3 = M0.clone(); M3[rows_at(j0)] += torch.randn(M3[rows_at(j0)].shape, generator=g0)  # memory of tick j0 only
    y3 = cb.forward_clip(X0, M3, F0)
    upto = [b * F0 + i for b in range(B0) for i in range(j0 + 1)]
    win = [b * F0 + i for b in range(B0) for i in range(j0 + 1, j0 + K + 1)]
    beyond = [b * F0 + i for b in range(B0) for i in range(j0 + K + 1, F0)]
    d_upto, d_win, d_beyond = mx(y3[upto], yc[upto]), min(mx(y3[[b * F0 + i]], yc[[b * F0 + i]]) for b in range(B0) for i in range(j0 + 1, j0 + K + 1)), mx(y3[beyond], yc[beyond])
report("T3a block never reads the future", dpre < 1e-6 and dpost > 1e-3 and d_upto < 1e-6 and d_win > 1e-4 and d_beyond < 1e-6,
       f"ticks < {j}: {dpre:.3g} (<1e-6), >= {j}: {dpost:.3g}; memory of tick {j0}: ticks <= {j0} {d_upto:.3g} (=0), "
       f"each of {j0 + 1}..{j0 + K} >= {d_win:.3g} (>0), > {j0 + K} {d_beyond:.3g} (=0)")


def stream_block(blk, xs_reads, mem, reset_at=-1, rows=None):
    """xs_reads: list of (B*F, C, H, W) inputs, one per read call of every tick; returns per read call (B*F, ...) outputs"""
    rows = list(range(B0)) if rows is None else rows
    blk.ring, blk.staged = [], None
    outs = [torch.zeros_like(x) for x in xs_reads]
    for i in range(F0):
        if i == reset_at:
            blk.ring, blk.staged = [], None
        idx = [b * F0 + i for b in rows]
        for x_, o_ in zip(xs_reads, outs):
            o_[idx] = blk.forward_stream(x_[idx], False)
        blk.forward_stream(mem[idx], True); blk.publish()
    return outs


with torch.no_grad():
    ya, yb = stream_block(cb, [X0, XB], M0)
    ra, rb = cb.forward_clip(X0, M0, F0), cb.forward_clip(XB, M0, F0)
    worst = max(mx(ya, ra), mx(yb, rb))
    ring_len = len(cb.ring)
    ca, _ = stream_block(cb, [X0, XB], M0, reset_at=7)
    cut = [b * F0 + i for b in range(B0) for i in range(7, F0)]
    first_after = mx(ca[rows_at(7)], X0[rows_at(7)])
    Fc = F0 - 7  # clip that STARTS at the cut: rows of ticks 7.. of every stream
    fresh = cb.forward_clip(X0[cut], M0[cut], Fc)  # cut rows are clip-major, ticks 7.. inside each clip
    d_reset = mx(ca[cut], fresh)
    after = [b * F0 + i for b in range(B0) for i in range(8, F0)]
    d_hist = mx(ca[after], ya[after])
    (alone,) = stream_block(cb, [X0], M0, rows=[1])
    iso = [1 * F0 + i for i in range(F0)]
    d_iso = mx(alone[iso], ya[iso])
    _ = stream_block(cb, [X0], M0)  # full 4-row ring after 12 ticks; slice the cond row (row 0) and read one more tick
    snap = ring_state([_Hold(cb)])
    xn = torch.randn(B0, C0, H0, W0, generator=g0)
    y_all = cb.forward_stream(xn, False)
    load_ring([_Hold(cb)], snap, rows=[0])
    y_row0 = cb.forward_stream(xn[0:1], False)
    d_slice = mx(y_row0, y_all[0:1])
report("T4 block stream == clip", worst < 1e-4 and ring_len == K and first_after == 0.0 and d_reset < 1e-4 and d_hist > 1e-3 and d_iso < 1e-5 and d_slice < 1e-5,
       f"{F0} ticks x {B0} rows x 2 reads max |diff| {worst:.2e} (<1e-4), ring holds {ring_len} (=K); reset: first tick {first_after:.3g} (=0), "
       f"suffix vs clip from the cut {d_reset:.2e}, vs unreset {d_hist:.2e} (>1e-3); alone vs interleaved {d_iso:.2e}; cond-row slice {d_slice:.2e}")

# ================= part 1: whole UNet =================
try:
    from diffusers import UNet2DConditionModel
    HAVE_DIFFUSERS = True
except Exception:
    HAVE_DIFFUSERS = False
if (a.small or a.init) and not HAVE_DIFFUSERS:
    report("part 1", False, "diffusers not importable")
elif a.small or a.init:
    if a.init:
        dev = "cuda"; SD = "sd-legacy/stable-diffusion-v1-5"
        import json
        from peft import LoraConfig
        from mrv.sd import CameraAdapter, LORA_TARGETS
        ck = torch.load(a.init, map_location="cpu"); r = ck["args"]["rank"]
        base = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
        base.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); base.load_state_dict(ck["lora"], strict=False)
        base = base.to(dev).eval()
        adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
        RAYS = torch.load(a.rays); DEVL = json.load(open("data/temporal_lists.json"))["dev"]
        ENC = {s: torch.load(f"{a.enc}/{s}.pt") for s in DEVL[:2]}

        def ctx_of(k, i0, n, null=False):
            sc = DEVL[k]
            with torch.no_grad():
                tok = ENC[sc]["tokens"][i0:i0 + n].to(dev).float(); ray = RAYS[sc][None].expand(n, *RAYS[sc].shape).to(dev).float()
                return adapter(tok, ray, torch.full((n,), 2, dtype=torch.long, device=dev), torch.ones(n, dtype=torch.bool, device=dev) if null else None)
        LAT = (4, 32, 64); F1 = 12; N_BLK = 16
    else:
        dev = "cpu"
        torch.manual_seed(0)
        base = UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4, block_out_channels=(32, 64), layers_per_block=1,
                                    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                                    cross_attention_dim=32, attention_head_dim=8, norm_num_groups=32).eval()
        CTX = torch.randn(4, 64, 7, 32, generator=torch.Generator().manual_seed(11))  # (source, tick, tokens, dim)

        def ctx_of(k, i0, n, null=False):
            return (CTX[3] if null else CTX[k])[i0:i0 + n].clone()
        LAT = (4, 16, 32); F1 = 12; N_BLK = None
    Wm = copy.deepcopy(base).to(dev).eval(); TW = add_clean_temporal(Wm, K)
    for w in TW:
        w.temporal.to(dev)
    report("part 1 blocks inserted", N_BLK is None or len(TW) == N_BLK, f"{len(TW)} temporal blocks")
    T_STAR = 1  # DDIM-25 last timestep (train_clean.py computes and asserts it from the scheduler)
    g1 = torch.Generator().manual_seed(21)
    rnd = lambda *s: torch.randn(*s, generator=g1).to(dev)
    tvec = lambda tv, n: torch.full((n,), tv, device=dev)

    def unet_mem(xm, c):
        set_mode(TW, "off")
        with torch.no_grad():
            return collect_inputs(TW, lambda: Wm(xm, tvec(T_STAR, xm.shape[0]), encoder_hidden_states=c))

    def run_clip(x, tv, c, mem, n):
        set_mode(TW, "clip", n); set_memory(TW, mem)
        with torch.no_grad():
            o = Wm(x, tvec(tv, x.shape[0]), encoder_hidden_states=c).sample
        set_memory(TW, None); return o

    def call(x, tv, c, stage=False):
        set_mode(TW, "stream"); set_stage(TW, stage)
        with torch.no_grad():
            return Wm(x, tvec(tv, x.shape[0]), encoder_hidden_states=c).sample

    # ---------- T1 ----------
    ctx = ctx_of(0, 0, F1); xt = rnd(F1, *LAT); xm = rnd(F1, *LAT)
    with torch.no_grad():
        ref = base(xt, tvec(500, F1), encoder_hidden_states=ctx).sample
        ref1 = torch.cat([base(xt[i:i + 1], tvec(500, 1), encoder_hidden_states=ctx[i:i + 1]).sample for i in range(F1)])
        set_mode(TW, "off"); o_off = Wm(xt, tvec(500, F1), encoder_hidden_states=ctx).sample
    mem = unet_mem(xm, ctx)
    o_clip = run_clip(xt, 500, ctx, mem, F1)
    reset(TW); o_str = []
    for i in range(F1):
        o_str.append(call(xt[i:i + 1], 500, ctx[i:i + 1]))
        call(xm[i:i + 1], T_STAR, ctx[i:i + 1], stage=True); publish(TW)
    d1 = max(mx(o_off, ref), mx(o_clip, ref), mx(torch.cat(o_str), ref1))
    perturb_params([(n_, q) for w in TW for n_, q in w.temporal.named_parameters()], 1, 0.05, dev)
    mem = unet_mem(xm, ctx)
    o_clip_p = run_clip(xt, 500, ctx, mem, F1)
    reset(TW); o_s0 = call(xt[0:1], 500, ctx[0:1])
    with torch.no_grad():
        a_m0 = base(xm[0:1], tvec(T_STAR, 1), encoder_hidden_states=ctx[0:1]).sample
    o_st = call(xm[0:1], T_STAR, ctx[0:1], stage=True); publish(TW)
    o_st2 = call(xm[1:2], T_STAR, ctx[1:2], stage=True)  # ring non-empty: a stage call is still A exactly
    with torch.no_grad():
        a_m1 = base(xm[1:2], tvec(T_STAR, 1), encoder_hidden_states=ctx[1:2]).sample
    d1b = max(mx(o_clip_p[0], ref[0]), mx(o_s0, ref1[0:1]), mx(o_st, a_m0), mx(o_st2, a_m1))
    d1c = mx(o_clip_p[K:], ref[K:])
    report("T1 UNet parity with A", d1 == 0.0 and d1b == 0.0 and d1c > 1e-3,
           f"fresh off/clip/stream vs A {d1:.3g} (=0); perturbed: clip tick 0, empty-ring call, stage calls vs A {d1b:.3g} (=0); later rows {d1c:.3g} (>0)")

    # ---------- T3a ----------
    j = 6
    xt2, xm2, ctx2 = xt.clone(), xm.clone(), ctx.clone()
    xt2[j:] += rnd(F1 - j, *LAT); xm2[j:] += rnd(F1 - j, *LAT); ctx2[j:] = ctx_of(1, j, F1 - j)
    o_b = run_clip(xt2, 500, ctx2, unet_mem(xm2, ctx2), F1)
    dpre, dpost = mx(o_clip_p[:j], o_b[:j]), mx(o_clip_p[j:], o_b[j:])
    xm3 = xm.clone(); xm3[3] += rnd(*LAT)  # memory of tick 3 only
    o_c = run_clip(xt, 500, ctx, unet_mem(xm3, ctx), F1)
    d_upto, d_next, d_beyond = mx(o_c[:4], o_clip_p[:4]), mx(o_c[4:4 + K], o_clip_p[4:4 + K]), mx(o_c[4 + K:], o_clip_p[4 + K:])
    report("T3a UNet clip never reads the future", dpre < 1e-6 and dpost > 1e-3 and d_upto < 1e-6 and d_next > 1e-4 and d_beyond < 1e-6,
           f"ticks < {j}: {dpre:.3g} (<1e-6), >= {j}: {dpost:.3g} (>1e-3); memory of tick 3: ticks <= 3 {d_upto:.3g}, 4..{3 + K} {d_next:.3g} (>0), later {d_beyond:.3g}")

    # ---------- T4 stream == clip through eviction; reset ----------
    C4 = torch.stack([ctx_of(0, 0, F1), ctx_of(1, 0, F1), ctx_of(0, 0, F1, True), ctx_of(1, 0, F1, True)])  # [s1 c, s2 c, s1 u, s2 u]
    X4 = {s: rnd(4, F1, *LAT) for s in range(2)}; XM = rnd(4, F1, *LAT); TV = {0: 981, 1: 500}
    MEMR = [unet_mem(XM[row], C4[row]) for row in range(4)]
    ref4 = {s: torch.stack([run_clip(X4[s][row], TV[s], C4[row], MEMR[row], F1) for row in range(4)]) for s in range(2)}

    def stream4(reset_at=-1, rows=(0, 1, 2, 3)):
        reset(TW); outs = {s: [] for s in range(2)}; rw = list(rows)
        for i in range(F1):
            if i == reset_at:
                reset(TW)
            for s in range(2):  # DDIM calls that read the ring
                outs[s].append(call(X4[s][rw, i], TV[s], C4[rw, i]))
            call(XM[rw, i], T_STAR, C4[rw, i], stage=True); publish(TW)  # last call at t*: stage, publish after the tick
        return {s: torch.stack(o, 1) for s, o in outs.items()}
    o4 = stream4()
    worst = max(mx(o4[s], ref4[s]) for s in range(2))
    cut = stream4(reset_at=6)
    set_mode(TW, "off")
    with torch.no_grad():
        a6 = Wm(X4[1][:, 6], tvec(500, 4), encoder_hidden_states=C4[:, 6]).sample
    first_after = mx(cut[1][:, 6], a6)
    fresh6 = torch.stack([run_clip(X4[1][row, 6:], 500, C4[row, 6:], [m[6:] for m in MEMR[row]], F1 - 6) for row in range(4)])
    d_reset = mx(cut[1][:, 6:], fresh6)
    d_hist = mx(cut[1][:, 7:], o4[1][:, 7:])
    alone = stream4(rows=(1,))
    d_iso = mx(alone[1][0], o4[1][1])
    report("T4 UNet stream == clip", worst < 1e-4 and first_after == 0.0 and d_reset < 1e-4 and d_hist > 1e-3 and d_iso < 1e-4,
           f"{F1} ticks x 4 rows x 2 reads max |diff| {worst:.2e} (<1e-4); reset at 6: first tick vs A {first_after:.3g} (=0), suffix vs clip from the cut "
           f"{d_reset:.2e} (<1e-4), vs unreset {d_hist:.2e} (>1e-3); stream alone vs interleaved {d_iso:.2e}")

# ================= part 2: T7 frozen guard through the real trainer =================
if a.t7:
    if not a.init:
        report("T7", False, "needs --init")
    else:
        out = f"runs/ctest_clean_t7_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
        base_cmd = [sys.executable, "scripts/train_clean.py", "--init", a.init, "--out", out, "--timing"]
        r1 = subprocess.run(base_cmd + ["--steps", str(a.t7_updates), "--max_steps_this_job", str(a.t7_updates)], capture_output=True, text=True)
        tline = [l_ for l_ in r1.stdout.splitlines() if l_.startswith("t* = ")]
        ok1 = r1.returncode == 0 and "frozen check after first update" in r1.stdout and "CLEAN_TRAIN_DONE" in r1.stdout and tline
        r2 = subprocess.run(base_cmd + ["--steps", str(a.t7_updates + 1), "--resume", "--max_steps_this_job", "1"], capture_output=True, text=True)
        ok2 = r2.returncode == 0 and f"resumed at {a.t7_updates}" in r2.stdout and "frozen check after first update" in r2.stdout
        report("T7 frozen guard + resume", bool(ok1 and ok2),
               f"first run exit {r1.returncode} ({tline[0] if tline else 't* not printed'}), resume exit {r2.returncode}"
               + ("" if ok1 and ok2 else f"\n{r1.stdout[-1500:]}\n{r1.stderr[-1500:]}\n{r2.stdout[-800:]}\n{r2.stderr[-800:]}"))
print("CLEAN_TESTS", "FAIL " + ",".join(FAIL) if FAIL else "ALL PASS", flush=True)
sys.exit(1 if FAIL else 0)
