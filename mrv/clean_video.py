"""Clean-memory causal temporal layers for the frozen A@20k renderer, arm "vclean" (council/AGREED-causal-r1.md, Amendment A4;
designs council/next-arm-r1-*.md). Separate from mrv/causal_video.py (fingerprinted by a signed decision; not modified).

Same block as mrv/causal_video.CausalTemporalBlock (per-frame GroupNorm32 -> per-pixel tokens, own q/k/v Linear, 8 heads,
per-head age bias heads x (K+1), +attn, +FF(LayerNorm), zero-init output Linear; identical parameter names, shapes and
construction order, so a seeded init equals vpriv's and vpriv's block weights load strictly). What changes is WHERE the past
keys / values come from: not the past ticks' features at the same diffusion step, but a MEMORY of each past tick's block-input
features (the spatial transformer's output) computed at t* = the last timestep of the DDIM-25 schedule with temporal deltas
OFF. GroupNorm and the k/v projections are applied to the memory features with the same weights as to the query's features.
The query reads its own token (age 0, from the current noisy pass) plus memory rows of ticks j with 1 <= i - j <= K. Rows with
no past (clip tick 0, empty ring) get exactly zero temporal delta, so a fresh block, mode "off", the first tick of a stream and
the first tick after a reset all equal A exactly.

Modes (set_mode):
  "off"    : no temporal computation (also used for the memory pass: collect_inputs hooks the block inputs).
  "clip"   : training. Input rows are B clips x F consecutive ticks (frame-major inside a clip); w.memory (set_memory) is a
             tensor of the same shape holding every row's t* features; query tick i reads self + memory of ticks i-K .. i-1.
  "stream" : rollout. Input rows are the streams' CURRENT tick (fixed row order, e.g. [cond, uncond]). ONE ring per block holds
             the last K ticks' memory (as k / v of the t* features, cond and uncond as separate rows). set_stage(False): read
             the ring (DDIM calls 1..24). set_stage(True): temporal deltas OFF, this call's block inputs are staged (call 25 at
             t*). publish() appends the staged tick to the ring once per tick, after the tick, evicting beyond K. reset()
             clears ring + stage (hard break); camera swaps keep them. k / v are computed at stage time under the caller's
             autocast; with fixed weights (rollout) this equals projecting the stored features at every read.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanTemporalBlock(nn.Module):
    def __init__(s, c, K=4, heads=8):
        super().__init__()
        assert c % heads == 0
        s.c, s.K, s.heads, s.d = c, K, heads, c // heads
        # construction order == CausalTemporalBlock (same RNG consumption -> same seeded init; same state_dict keys)
        s.norm = nn.GroupNorm(32, c)
        s.qkv = nn.Linear(c, 3 * c)
        s.age_bias = nn.Parameter(torch.zeros(heads, K + 1))  # bias[h, age], age 0 = self, 1..K = memory of tick i - age
        s.attn_out = nn.Linear(c, c)
        s.ln = nn.LayerNorm(c)
        s.ff = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))
        s.out = nn.Linear(c, c)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
        s.ring = []      # stream: list (oldest .. newest) of (k, v), each (R, HW, heads, d), detached
        s.staged = None  # stream: this tick's (k, v) from the t* call, appended by publish()

    def _tokens(s, x):
        """(R, C, H, W) -> per-pixel tokens (R, H*W, C) after per-frame GroupNorm"""
        h = s.norm(x.float()).to(x.dtype)
        return h.flatten(2).transpose(1, 2)

    def _delta(s, h, a):
        h = h + s.attn_out(a)
        h = h + s.ff(s.ln(h))
        return s.out(h)

    def forward_clip(s, x, mem, n_frames):
        R, C, H, W = x.shape
        assert mem is not None and mem.shape == x.shape, "clip mode needs the t* memory of every row (set_memory)"
        B, Fn, HW = R // n_frames, n_frames, H * W
        tok = s._tokens(x).reshape(B, Fn, HW, C).permute(0, 2, 1, 3)  # (B, HW, F, C)
        mtok = s._tokens(mem.to(x.dtype)).reshape(B, Fn, HW, C).permute(0, 2, 1, 3)
        q, ks, vs = s.qkv(tok).chunk(3, -1)
        _, km, vm = s.qkv(mtok).chunk(3, -1)  # same projection weights on the memory features (q of memory unused)
        sh = lambda t: t.reshape(B * HW, Fn, s.heads, s.d).transpose(1, 2)  # (N, heads, F, d)
        q = sh(q); k = torch.cat([sh(ks), sh(km)], 2); v = torch.cat([sh(vs), sh(vm)], 2)  # keys: [self 0..F-1, memory 0..F-1]
        i = torch.arange(Fn, device=x.device)
        age = i[:, None] - i[None, :]  # (F_query, F_key)
        b_self = s.age_bias[:, 0][:, None, None].expand(s.heads, Fn, Fn).masked_fill(~(age == 0)[None], float("-inf"))
        b_mem = s.age_bias[:, age.clamp(0, s.K)].masked_fill(~((age >= 1) & (age <= s.K))[None], float("-inf"))
        bias = torch.cat([b_self, b_mem], -1).to(q.dtype)  # (heads, F, 2F); self always allowed -> no all-masked row
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None])  # (N, heads, F, d)
        a = a.transpose(1, 2).reshape(B, HW, Fn, C)
        d = s._delta(tok, a)
        has_past = (i > 0).to(d.dtype)[None, None, :, None]  # tick 0 of a clip has no memory: exactly zero delta
        d = (d * has_past).permute(0, 2, 3, 1).reshape(R, C, H, W)
        return x + d.to(x.dtype)

    def _kv(s, tok):
        R, HW, C = tok.shape
        _, k, v = s.qkv(tok).chunk(3, -1)
        return k.reshape(R, HW, s.heads, s.d), v.reshape(R, HW, s.heads, s.d)

    def forward_stream(s, x, stage):
        R, C, H, W = x.shape
        if stage:  # call at t*: temporal delta OFF (== A); stage this tick's memory for publish()
            k, v = s._kv(s._tokens(x))
            s.staged = (k.detach().clone(), v.detach().clone())
            return x
        if not s.ring:  # empty past: exactly A
            return x
        HW, n = H * W, len(s.ring)
        assert s.ring[0][0].shape[0] == R, f"stream rows changed: ring {s.ring[0][0].shape[0]} vs input {R}"
        tok = s._tokens(x)  # (R, HW, C)
        q, k, v = s.qkv(tok).chunk(3, -1)
        sh = lambda t: t.reshape(R * HW, 1, s.heads, s.d).transpose(1, 2)  # (N, heads, 1, d)
        st = lambda ts: torch.stack(ts, 3).reshape(R * HW, s.heads, n, s.d).to(q.dtype)  # ring -> (N, heads, n, d)
        kk = torch.cat([st([e[0] for e in s.ring]), sh(k)], 2)
        vv = torch.cat([st([e[1] for e in s.ring]), sh(v)], 2)
        ages = torch.arange(n, -1, -1, device=x.device)  # oldest (age n) .. newest (1), self (0)
        bias = s.age_bias[:, ages].to(q.dtype)[None, :, None, :]  # (1, heads, 1, n+1)
        a = F.scaled_dot_product_attention(sh(q), kk, vv, attn_mask=bias)
        d = s._delta(tok, a.transpose(1, 2).reshape(R, HW, C))
        return x + d.transpose(1, 2).reshape(R, C, H, W).to(x.dtype)

    def publish(s):
        if s.staged is not None:
            s.ring = (s.ring + [s.staged])[-s.K:]  # evict beyond K
            s.staged = None


class WithCleanTemporal(nn.Module):
    """wraps a diffusers Transformer2DModel; .mode in {off, clip, stream}; module / parameter names as WithCausalTemporal"""
    def __init__(s, inner, c, K):
        super().__init__()
        s.inner = inner; s.temporal = CleanTemporalBlock(c, K); s.config = inner.config
        s.mode, s.n_frames, s.memory, s.stage = "off", None, None, False

    def forward(s, hidden_states, *args, **kw):
        out = s.inner(hidden_states, *args, **kw)
        x = out[0] if isinstance(out, tuple) else out.sample
        if s.mode == "clip":
            x = s.temporal.forward_clip(x, s.memory, s.n_frames)
        elif s.mode == "stream":
            x = s.temporal.forward_stream(x, s.stage)
        elif s.mode != "off":
            raise ValueError(s.mode)
        if isinstance(out, tuple):
            return (x,) + tuple(out[1:])
        out.sample = x
        return out


def add_clean_temporal(unet, K=4):
    """insert a clean-memory temporal block after each spatial transformer (same traversal as add_causal_temporal)"""
    from diffusers.models.transformers.transformer_2d import Transformer2DModel
    wrappers = []
    for name, mod in list(unet.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, Transformer2DModel):
                w = WithCleanTemporal(child, child.config.in_channels, K)
                setattr(mod, cname, w); wrappers.append(w)
    return wrappers


def set_mode(wrappers, mode, n_frames=None):
    for w in wrappers:
        w.mode, w.n_frames = mode, n_frames


def set_memory(wrappers, mems):
    """clip mode: one (R, C, H, W) t* feature tensor per block (from collect_inputs), or None to clear"""
    mems = [None] * len(wrappers) if mems is None else mems
    assert len(mems) == len(wrappers)
    for w, m in zip(wrappers, mems):
        w.memory = m


def set_stage(wrappers, flag):
    for w in wrappers:
        w.stage = bool(flag)


def publish(wrappers):
    for w in wrappers:
        w.temporal.publish()


def reset(wrappers):
    for w in wrappers:
        w.temporal.ring, w.temporal.staged = [], None


def collect_inputs(wrappers, run):
    """forward hooks on each block's spatial transformer: call run() (a UNet forward, normally no-grad, blocks "off") and
    return the detached per-block temporal-block inputs, in wrapper order"""
    got = [None] * len(wrappers)

    def hook(idx):
        def f(mod, inp, out):
            got[idx] = (out[0] if isinstance(out, tuple) else out.sample).detach()
        return f
    hs = [w.inner.register_forward_hook(hook(i)) for i, w in enumerate(wrappers)]
    try:
        run()
    finally:
        for h in hs:
            h.remove()
    assert all(g is not None for g in got), "a temporal block input was not captured"
    return got


def ring_state(wrappers):
    """snapshot of every block's ring (list of (k, v) per block); tensors are shared, rings are replaced not mutated"""
    return [list(w.temporal.ring) for w in wrappers]


def load_ring(wrappers, state, rows=None):
    """restore a ring snapshot; rows (list of row indices, e.g. [0] = cond only) selects stream rows"""
    for w, ring in zip(wrappers, state):
        w.temporal.ring = [(k[rows], v[rows]) if rows is not None else (k, v) for k, v in ring]
        w.temporal.staged = None
