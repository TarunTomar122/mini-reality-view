"""Acceptance tests for the causal video adapter (council/AGREED-causal-r1.md), run BEFORE training, fp32, on an A6000
(48 GB). Exits 1 if any test fails. T2 / T5 / T6 / T7 run in slurm/causal_tests.sh (rollouts and the real trainer).

  T1 parity: fresh adapter (off / clip / stream) == unwrapped A exactly at matched batch shapes; after perturbing the
     temporal weights, rows without a past (clip tick 0, first stream tick) still == A exactly.
  T3 no look-ahead: (a) one joint clip forward with nonzero weights: perturbing inputs of ticks >= j leaves outputs < j
     unchanged and changes outputs >= j; (b) a 12-tick causal rollout (DDIM 25, CFG 2) with the cameras of ticks >= 8 from
     another drive: ticks < 8 bit-identical, later ticks differ.
  T4 stream == causal clip: 24 ticks, two interleaved streams x two CFG rows (4 rows, distinct inputs), three DDIM slots with
     distinct inputs: max |diff| < 1e-4; clearing the rings at tick 12 changes later outputs (history is used); stream 1 alone
     == stream 1 interleaved (row isolation); single block: a frame K+1 back has no effect, a frame K back has.
"""
import argparse, gc, json, sys, zlib
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.causal_video import add_causal_temporal, set_mode, set_step, reset, CausalTemporalBlock

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--K", type=int, default=4)
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--rays", default="data/calib_rays.pt")
a = p.parse_args()
dev = "cuda"; SD = "sd-legacy/stable-diffusion-v1-5"; K = a.K
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
ck = torch.load(a.init, map_location="cpu"); r = ck["args"]["rank"]
FAIL = []


def load_unet():
    u = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
    u.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); u.load_state_dict(ck["lora"], strict=False)
    return u.to(dev).eval()


def report(name, ok, msg):
    print(("PASS " if ok else "FAIL ") + name + ": " + msg, flush=True)
    if not ok:
        FAIL.append(name)


def perturb(TW, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for w in TW:
            for n_, q in w.temporal.named_parameters():
                if not (n_.startswith("norm") or n_.startswith("ln")):
                    q.add_(torch.randn(q.shape, generator=g).to(dev) * 0.05)


adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
adapter.requires_grad_(False)
RAYS = torch.load(a.rays)
DEVL = json.load(open("data/temporal_lists.json"))["dev"]
sc1, sc2 = DEVL[0], DEVL[1]
E1, E2 = torch.load(f"{a.enc}/{sc1}.pt"), torch.load(f"{a.enc}/{sc2}.pt")
st = lambda n: torch.full((n,), 2, dtype=torch.long, device=dev)


def ctx_of(E, sc, i0, n, null=False):
    with torch.no_grad():
        tok = E["tokens"][i0:i0 + n].to(dev).float(); ray = RAYS[sc][None].expand(n, *RAYS[sc].shape).to(dev).float()
        return adapter(tok, ray, st(n), torch.ones(n, dtype=torch.bool, device=dev) if null else None)


# ---------- T1 ----------
F = 12
ctx = ctx_of(E1, sc1, 0, F); xt = torch.randn(F, 4, 32, 64, generator=torch.Generator().manual_seed(0)).to(dev)
t = torch.full((F,), 500, device=dev)
base = load_unet()
with torch.no_grad():
    ref = base(xt, t, encoder_hidden_states=ctx).sample
    ref1 = torch.cat([base(xt[i:i + 1], t[i:i + 1], encoder_hidden_states=ctx[i:i + 1]).sample for i in range(F)])
del base; gc.collect(); torch.cuda.empty_cache()
W = load_unet(); TW = add_causal_temporal(W, K); [w.temporal.to(dev) for w in TW]


def run_clip(x, tt, c, n):
    set_mode(TW, "clip", n)
    with torch.no_grad():
        return W(x, tt, encoder_hidden_states=c).sample


def run_stream_rows(xs, tt, cs, slot=0):
    """xs / cs: lists over ticks of (R, ...) row-batches; returns (ticks, R, ...)"""
    set_mode(TW, "stream"); set_step(TW, slot); outs = []
    with torch.no_grad():
        for x, c in zip(xs, cs):
            outs.append(W(x, tt[:x.shape[0]], encoder_hidden_states=c).sample)
    return torch.stack(outs)


with torch.no_grad():
    set_mode(TW, "off"); o_off = W(xt, t, encoder_hidden_states=ctx).sample
o_clip = run_clip(xt, t, ctx, F); reset(TW)
o_str = run_stream_rows([xt[i:i + 1] for i in range(F)], t, [ctx[i:i + 1] for i in range(F)])[:, 0]
d1 = max((o_off - ref).abs().max().item(), (o_clip - ref).abs().max().item(), (o_str - ref1).abs().max().item())
perturb(TW, 1)
o_clip_p = run_clip(xt, t, ctx, F); reset(TW)
o_str_p = run_stream_rows([xt[0:1]], t, [ctx[0:1]])[:, 0]
d1b = max((o_clip_p[0] - ref[0]).abs().max().item(), (o_str_p[0] - ref1[0]).abs().max().item())
d1c = (o_clip_p[K:] - ref[K:]).abs().max().item()
report("T1 parity with A", d1 == 0.0 and d1b == 0.0 and d1c > 1e-3,
       f"fresh off/clip/stream vs A {d1:.3g} (=0), perturbed empty-past rows vs A {d1b:.3g} (=0), later rows differ {d1c:.3g} (>0)")

# ---------- T3a no look-ahead inside one joint clip ----------
j = 6
xt2 = xt.clone(); xt2[j:] += torch.randn(xt2[j:].shape, generator=torch.Generator().manual_seed(9)).to(dev)
ctx2 = ctx.clone(); ctx2[j:] = ctx_of(E2, sc2, j, F - j)
o_b = run_clip(xt2, t, ctx2, F)
dpre, dpost = (o_clip_p[:j] - o_b[:j]).abs().max().item(), (o_clip_p[j:] - o_b[j:]).abs().max().item()
report("T3a joint clip never reads the future", dpre < 1e-6 and dpost > 1e-3, f"ticks < {j}: {dpre:.3g} (<1e-6), ticks >= {j}: {dpost:.3g} (>1e-3)")

# ---------- T4 stream == clip: 24 ticks, 2 streams x 2 CFG rows, 3 slots ----------
F4 = 24
g4 = torch.Generator().manual_seed(3)
X = {s: torch.randn(4, F4, 4, 32, 64, generator=g4).to(dev) for s in range(3)}  # per slot: rows [s1 cond, s2 cond, s1 unc, s2 unc]
C = torch.stack([ctx_of(E1, sc1, 0, F4), ctx_of(E2, sc2, 0, F4), ctx_of(E1, sc1, 0, F4, True), ctx_of(E2, sc2, 0, F4, True)])
tt4 = {0: torch.full((F4,), 981, device=dev), 1: torch.full((F4,), 500, device=dev), 2: torch.full((F4,), 21, device=dev)}
worst, worst_iso = 0.0, 0.0
ref4 = {s: torch.stack([run_clip(X[s][row], tt4[s], C[row], F4) for row in range(4)]) for s in range(3)}
reset(TW); outs = {s: [] for s in range(3)}
set_mode(TW, "stream")
with torch.no_grad():
    for i in range(F4):
        for s in range(3):  # slots interleaved within each tick, as in a real rollout
            set_step(TW, s)
            outs[s].append(W(X[s][:, i], tt4[s][:4], encoder_hidden_states=C[:, i]).sample)
for s in range(3):
    worst = max(worst, (torch.stack(outs[s], 1) - ref4[s]).abs().max().item())
reset(TW); alone = []
with torch.no_grad():
    for i in range(F4):
        set_step(TW, 1); alone.append(W(X[1][0:1, i], tt4[1][:1], encoder_hidden_states=C[0:1, i]).sample)
worst_iso = (torch.stack(alone, 1)[0] - torch.stack(outs[1], 1)[0]).abs().max().item()
reset(TW); cut = []
with torch.no_grad():
    for i in range(F4):
        if i == 12:
            reset(TW)
        set_step(TW, 1); cut.append(W(X[1][0:1, i], tt4[1][:1], encoder_hidden_states=C[0:1, i]).sample)
d_hist = (torch.stack(cut, 1)[0, 13:] - ref4[1][0, 13:]).abs().max().item()
fresh12 = run_clip(X[1][0, 12:], tt4[1][12:], C[0, 12:], F4 - 12)  # a clip that STARTS at the cut
d_reset = (torch.stack(cut, 1)[0, 12:] - fresh12).abs().max().item()
blk = CausalTemporalBlock(320, K).to(dev)
with torch.no_grad():
    for n_, q in blk.named_parameters():
        if not (n_.startswith("norm") or n_.startswith("ln")):
            q.add_(torch.randn(q.shape, generator=torch.Generator().manual_seed(5)).to(dev) * 0.1)
    xb = torch.randn(F4, 320, 8, 16, generator=torch.Generator().manual_seed(6)).to(dev)
    yb = blk.forward_clip(xb, F4)
    bump = torch.randn(320, 8, 16, generator=torch.Generator().manual_seed(8)).to(dev)  # non-uniform: survives per-frame GroupNorm
    xk1 = xb.clone(); xk1[12 - K - 1] += bump; yk1 = blk.forward_clip(xk1, F4)
    xk = xb.clone(); xk[12 - K] += bump; yk = blk.forward_clip(xk, F4)
d_old, d_win = (yk1[12] - yb[12]).abs().max().item(), (yk[12] - yb[12]).abs().max().item()
report("T4 stream == causal clip", worst < 1e-4 and worst_iso < 1e-5 and d_hist > 1e-3 and d_reset < 1e-4 and d_old < 1e-6 and d_win > 1e-3,
       f"24 ticks x 4 rows x 3 slots max |diff| {worst:.2e} (<1e-4); stream alone vs interleaved {worst_iso:.2e}; cleared rings change "
       f"later output {d_hist:.2e} (>1e-3) and equal a clip starting at the cut {d_reset:.2e} (<1e-4); one block: frame {K + 1} back {d_old:.2e} (=0), frame {K} back {d_win:.2e} (>0)")

# ---------- T3b rollout prefix invariance ----------
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); ddim.set_timesteps(25); TS = ddim.timesteps
z0 = torch.randn(1, 4, 32, 64, generator=torch.Generator().manual_seed(zlib.crc32(f"{sc1}|0".encode()))).to(dev)


def rollout(swap_from):
    reset(TW); set_mode(TW, "stream"); outs_ = []
    for i in range(12):
        E, s_ = (E2, sc2) if i >= swap_from else (E1, sc1)
        with torch.no_grad():
            c = ctx_of(E, s_, i, 1); u = ctx_of(E, s_, i, 1, True); z = z0.clone()
            for si, tt in enumerate(TS):
                set_step(TW, si)
                e = W(torch.cat([z, z]), tt, encoder_hidden_states=torch.cat([c, u])).sample
                ec, eu = e.chunk(2); z = ddim.step(eu + 2.0 * (ec - eu), tt, z).prev_sample
        outs_.append(z)
    return torch.cat(outs_)


ra, rb = rollout(99), rollout(8)
dpre, dpost = (ra[:8] - rb[:8]).abs().max().item(), (ra[8:] - rb[8:]).abs().max().item()
report("T3b rollout prefix invariance", dpre == 0.0 and dpost > 1e-3, f"ticks < 8: {dpre:.3g} (=0), later {dpost:.3g} (>0)")
print("CAUSAL_TESTS", "FAIL " + ",".join(FAIL) if FAIL else "ALL PASS", flush=True)
sys.exit(1 if FAIL else 0)
