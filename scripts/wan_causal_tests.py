"""CPU tests of mrv/wan_causal.py (S2 groundwork; council/review-20261128-1430-astra.md / -fable.md). No Wan weights: a tiny
random 3-layer diffusers WanTransformer3DModel (the real attention / rotary / per-token-timestep code), bare and wrapped by
mrv.wan_cond.WanCond (guide channels through the extended patch conv, raw-view tokens through DualCrossAttn; the zero-init
guide conv, gates and the projector are randomised so the conditioning is live). Checks:
  1 causality: perturbing FUTURE guides, tokens and noise independently leaves every earlier plane's output bit-identical
    (masked block-causal forward, fixed noise);
  2 cached decoding == slow masked full-prefix recomputation (same eviction semantics) at all four denoising substeps of every
    plane, through the first eviction and across a drive reset, for the bare and the WanCond model;
  3 exactly one append per block per layer (substeps append nothing), oldest-first eviction, reset;
  4 absolute rotary positions == diffusers' own table at those positions; positions keep counting after eviction;
  5 no gradient through detached past K/V; the current plane's inputs and parameters get gradients;
  6 masked segments == separate forwards per drive (no attention across a reset, positions restart);
  7 guards: bare-model call outside a mode, gradient checkpointing with the cache.
  python scripts/wan_causal_tests.py      -> WAN_CAUSAL_TESTS FAILS=0
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from diffusers import WanTransformer3DModel
from mrv import wan_causal as WK

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
FAILS = 0


def check(name, cond, info=""):
    global FAILS
    print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info and not cond else ""))
    FAILS += int(not bool(cond))


H = W = 8                     # latent plane 8 x 8 -> 4 x 4 = 16 tokens per plane
SUBSTEPS = (1000.0, 857.0, 667.0, 400.0)   # Fable: sigma 1.0 / 0.857 / 0.667 / 0.4 (x 1000)
T_COMMIT = 0.0                # clean re-encode (Astra); Fable's small fixed noise is the same call with another value
TOL = 2e-4


def tiny_tf(text_dim=32):
    torch.manual_seed(1)
    return WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16, out_channels=16,
                                 text_dim=text_dim, freq_dim=32, ffn_dim=48, num_layers=3, cross_attn_norm=True,
                                 qk_norm="rms_norm_across_heads", rope_max_seq_len=64).eval()


def tiny_cond():
    from mrv import wan_cond as WC
    tf = tiny_tf(text_dim=WC.TEXT_DIM)
    m = WC.WanCond(tf, tok_ticks="last", use_tokens=True, prompt=None)
    torch.manual_seed(2)
    m.proj = WC.TokenProjector(1, d_tok=8, d=16, d_out=WC.TEXT_DIM, n_views=2)
    with torch.no_grad():
        m.tf.patch_embedding.extra.weight.normal_(0, 0.2)
        for g in m.xattn:
            g.gate.normal_(0, 0.5)
    return m.eval()


class Bare:
    """inputs for the bare transformer: x [1,16,n,H,W], text [1,4,32]"""
    def __init__(s, tf):
        s.tf = tf; s.text = torch.randn(1, 4, 32, generator=torch.Generator().manual_seed(3))

    def args(s, x, t, sl=None):
        return (), {"hidden_states": x, "timestep": t, "encoder_hidden_states": s.text, "return_dict": False}


class Cond:
    """inputs for WanCond: x, guide extra [1,24,n,H,W], tokens [1,n,1,2,3,8], rays [1,n,1,2,3,6]"""
    def __init__(s, m, n):
        g = torch.Generator().manual_seed(4)
        s.extra = torch.randn(1, 24, n, H, W, generator=g); s.tok = torch.randn(1, n, 1, 2, 3, 8, generator=g)
        s.ray = torch.randn(1, n, 1, 2, 3, 6, generator=g); s.keep = torch.ones(1, dtype=torch.bool)

    def args(s, x, t, sl=slice(None)):
        return (x, s.extra[:, :, sl], t, s.tok[:, sl], s.ray[:, sl], s.keep), {}


def per_token(ts, tpp=16):
    """[n] per-plane timesteps -> [1, n * tokens per plane]"""
    return torch.tensor(ts, dtype=torch.float32).repeat_interleave(tpp)[None]


# ---------------------------------------------------------------------------------------------------- 1 causality
m = tiny_cond(); cw = WK.CausalWan(m)
N = 6
inp = Cond(m, N)
x = torch.randn(1, 16, N, H, W, generator=torch.Generator().manual_seed(5))
t = torch.full((1,), 600.0)
with torch.no_grad():
    base = cw.masked(*inp.args(x, t)[0])
G = 3
for what in ("guide", "tokens", "noise"):
    inp2 = Cond(m, N); x2 = x.clone()
    if what == "guide":
        inp2.extra[:, :, G:] += torch.randn_like(inp2.extra[:, :, G:])
    elif what == "tokens":
        inp2.tok[:, G:] += torch.randn_like(inp2.tok[:, G:]); inp2.ray[:, G:] += 1.0
    else:
        x2[:, :, G:] = torch.randn_like(x2[:, :, G:])
    with torch.no_grad():
        out2 = cw.masked(*inp2.args(x2, t)[0])
    check(f"causality: perturbing FUTURE {what} (planes >= {G}) leaves planes < {G} bit-identical",
          torch.equal(out2[:, :, :G], base[:, :, :G]))
    check(f"causality control: the {what} perturbation does change plane {G} (the conditioning is live)",
          not torch.equal(out2[:, :, G], base[:, :, G]))
inp3 = Cond(m, N); inp3.extra[:, :, G - 1] += 1.0
with torch.no_grad():
    out3 = cw.masked(*inp3.args(x, t)[0])
check("causality: a guide change in plane g changes plane g (camera conditioning is per group)", not torch.equal(out3[:, :, G - 1], base[:, :, G - 1]))
check("causality: a guide change in plane g does not reach planes before g", torch.equal(out3[:, :, :G - 1], base[:, :, :G - 1]))
cw.uninstall()


# ---------------------------------------------------------------------------------------------------- 2-3 cache parity
def parity_run(model, mk_inp, label):
    """drive A of 7 planes (first eviction at plane 5) then reset and drive B of 3 planes; every substep compared with the slow
    masked full-prefix recomputation; returns the wrapper for the append / position checks"""
    cw = WK.CausalWan(model)
    worst, ok_all, app_ok, pos_ok = 0.0, True, True, True
    for drive, n in (("A", 7), ("B", 3)):
        cw.reset()
        g = torch.Generator().manual_seed(10 if drive == "A" else 20)
        clean = torch.randn(1, 16, n, H, W, generator=g)          # each plane's final latent
        inp = mk_inp(n)
        for p in range(n):
            for k, tk in enumerate(SUBSTEPS):
                zk = torch.randn(1, 16, 1, H, W, generator=g)       # the current plane's state at substep k
                a, kw = inp.args(zk, torch.full((1,), tk), slice(p, p + 1))
                with torch.no_grad():
                    fast = cw.step(*a, **kw)
                xs = torch.cat([clean[:, :, :p], zk], 2)
                ts = per_token([T_COMMIT] * p + [tk])
                a2, kw2 = inp.args(xs, ts, slice(0, p + 1))
                with torch.no_grad():
                    slow = cw.masked(*a2, positions=list(range(p + 1)), **kw2)[:, :, -1:]
                d = (fast - slow).abs().max().item(); worst = max(worst, d)
                ok_all = ok_all and d <= TOL * max(1.0, slow.abs().max().item())
            a, kw = inp.args(clean[:, :, p:p + 1], torch.full((1,), T_COMMIT), slice(p, p + 1))
            before = cw.appends()
            with torch.no_grad():
                cw.commit(*a, **kw)
            app_ok = app_ok and all(b + 1 == c for b, c in zip(before, cw.appends()))
            exp_pos = list(range(max(0, p + 1 - WK.WINDOW), p + 1))
            pos_ok = pos_ok and all(cw.cached_positions(L) == exp_pos for L in range(len(cw.tf.blocks)))
        if drive == "A":
            check(f"{label}: drive A (7 planes, evictions from plane 5) -> {n} appends per layer, cache holds absolute positions 3..6",
                  cw.appends() == [n] * len(cw.tf.blocks) and cw.cached_positions() == [3, 4, 5, 6], (cw.appends(), cw.cached_positions()))
    check(f"{label}: cached step == slow full-prefix recomputation at all 4 substeps of every plane, through eviction and a reset "
          f"(worst {worst:.2e})", ok_all, worst)
    check(f"{label}: exactly one append per block per layer (substeps append nothing)", app_ok)
    check(f"{label}: cached positions are the last <= 4 finished planes, oldest evicted first, absolute (restart at the reset)", pos_ok)
    return cw


bare_tf = tiny_tf(); bare = Bare(bare_tf)
cwb = parity_run(bare_tf, lambda n: bare, "bare Wan")
cwb.reset()
check("reset: caches empty, appends 0, time index 0", cwb.cached_positions() == [] and cwb.appends() == [0] * 3 and cwb.state.next_pos == 0)
cwb.uninstall()
mc = tiny_cond()
cwc = parity_run(mc, lambda n: Cond(mc, n), "WanCond (guide + tokens)")
cwc.uninstall()

# ---------------------------------------------------------------------------------------------------- 4 absolute positions
rope = bare_tf.rope
for tpos in (0, 1, 5, 17):
    full_c, full_s = rope(torch.zeros(1, 16, tpos + 1, H, W))
    c, s_ = WK.rotary_for(rope, [tpos], H // 2, W // 2)
    check(f"absolute rotary at position {tpos} == diffusers' table for the last of {tpos + 1} frames",
          torch.equal(c, full_c[:, -16:]) and torch.equal(s_, full_s[:, -16:]))
c2, _ = WK.rotary_for(rope, [3, 4, 5], H // 2, W // 2)
check("rotary for arbitrary positions == diffusers' 0..n-1 table sliced", torch.equal(c2, rope(torch.zeros(1, 16, 6, H, W))[0][:, 48:]))
try:
    WK.rotary_for(rope, [64], H // 2, W // 2); ok_ = False
except ValueError:
    ok_ = True
check("a time position beyond the rotary table is refused", ok_)

# ---------------------------------------------------------------------------------------------------- 5 gradient isolation
cwg = WK.CausalWan(bare_tf)
cwg.reset()
x_prev = torch.randn(1, 16, 1, H, W, requires_grad=True)
a, kw = bare.args(x_prev, torch.full((1,), T_COMMIT))
cwg.commit(*a, **kw)   # with grad enabled: the stored K/V must still be detached
check("committed K/V are stored detached", all(not k.requires_grad for L in cwg.state.k for k in L))
x_cur = torch.randn(1, 16, 1, H, W, requires_grad=True)
bare_tf.zero_grad(set_to_none=True)
a, kw = bare.args(x_cur, torch.full((1,), 600.0))
out = cwg.step(*a, **kw)
out.pow(2).mean().backward()
check("no gradient reaches the past plane through its cached K/V", x_prev.grad is None)
check("the current plane's input and the attention weights get gradients",
      x_cur.grad is not None and x_cur.grad.abs().sum() > 0 and bare_tf.blocks[0].attn1.to_k.weight.grad is not None)
cwg.uninstall()

# ---------------------------------------------------------------------------------------------------- 6 masked segments
cws = WK.CausalWan(bare_tf)
g = torch.Generator().manual_seed(30)
xa, xb = torch.randn(1, 16, 3, H, W, generator=g), torch.randn(1, 16, 2, H, W, generator=g)
with torch.no_grad():
    joint = cws.masked(*bare.args(torch.cat([xa, xb], 2), torch.full((1,), 500.0))[0], positions=[0, 1, 2, 0, 1], segments=[0, 0, 0, 1, 1],
                       **bare.args(torch.cat([xa, xb], 2), torch.full((1,), 500.0))[1])
    sa = cws.masked(**bare.args(xa, torch.full((1,), 500.0))[1])
    sb = cws.masked(**bare.args(xb, torch.full((1,), 500.0))[1])
check("masked: two drives in one forward (segments, positions restart) == separate forwards",
      (joint - torch.cat([sa, sb], 2)).abs().max().item() <= TOL, (joint - torch.cat([sa, sb], 2)).abs().max().item())
check("plane ticks: first plane 1 tick, then 4", WK.plane_ticks(4) == [1, 4, 4, 4])

# ---------------------------------------------------------------------------------------------------- 7 guards
try:
    with torch.no_grad():
        bare_tf(**bare.args(xa, torch.full((1,), 500.0))[1]); ok_ = False
except RuntimeError:
    ok_ = True
check("guard: calling the wrapped model outside masked / step / commit is refused", ok_)
bare_tf.gradient_checkpointing = True
try:
    cws.step(**bare.args(xa[:, :, :1], torch.full((1,), 500.0))[1]); ok_ = False
except RuntimeError:
    ok_ = True
bare_tf.gradient_checkpointing = False
check("guard: gradient checkpointing with the cache is refused", ok_)
cws.uninstall()
print(f"WAN_CAUSAL_TESTS FAILS={FAILS}")
sys.exit(1 if FAILS else 0)
