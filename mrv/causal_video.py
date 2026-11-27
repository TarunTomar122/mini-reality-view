"""Causal (live) temporal layers for the frozen A@20k renderer (council/AGREED-causal-r1.md; designs
council/causal-video-s1/s2-*.md).

After each of the 16 spatial transformers of the UNet, a zero-initialised temporal block lets every pixel of tick i attend to
the same pixel of ticks i-K .. i (itself included, never the future), at the SAME diffusion step (same noise level), with a
learned per-head bias per age 0..K. The block's output projection starts at zero, and rows with no past frame get exactly zero
temporal delta (also after training), so:
  * mode "off"  -> the wrapped spatial transformer's output, unchanged (== A),
  * a fresh block -> == A,
  * the first tick of a stream / after a hard break -> == A.

Modes (set with set_mode):
  "off"    : no temporal computation.
  "clip"   : input rows are B clips x F consecutive ticks (frame-major inside a clip); causal band mask over ticks.
  "stream" : input rows are the streams' CURRENT tick only (fixed row order, e.g. [cond, uncond]); keys / values of the last
             K ticks come from a ring per (layer, DDIM step index) held in the block; the current tick's K / V is appended
             after reading (oldest evicted). set_step(i) selects the ring; reset() clears all rings (hard break).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTemporalBlock(nn.Module):
    def __init__(s, c, K=4, heads=8):
        super().__init__()
        assert c % heads == 0
        s.c, s.K, s.heads, s.d = c, K, heads, c // heads
        s.norm = nn.GroupNorm(32, c)
        s.qkv = nn.Linear(c, 3 * c)
        s.age_bias = nn.Parameter(torch.zeros(heads, K + 1))  # bias[h, age], age = i - j in 0..K
        s.attn_out = nn.Linear(c, c)
        s.ln = nn.LayerNorm(c)
        s.ff = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))
        s.out = nn.Linear(c, c)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
        s.rings = {}  # stream mode: step index -> (k, v), each (N, heads, n<=K, d), detached

    def _tokens(s, x):
        """(R, C, H, W) -> per-pixel tokens (R, H*W, C) after per-frame GroupNorm"""
        R, C, H, W = x.shape
        h = s.norm(x.float()).to(x.dtype)
        return h.flatten(2).transpose(1, 2)

    def _delta(s, h, a):
        """h: tokens (N, C) stacked any way; a: attention result (same shape) -> temporal delta (zero-init out)"""
        h = h + s.attn_out(a)
        h = h + s.ff(s.ln(h))
        return s.out(h)

    def forward_clip(s, x, n_frames):
        R, C, H, W = x.shape
        B, Fn = R // n_frames, n_frames
        tok = s._tokens(x).reshape(B, Fn, H * W, C).permute(0, 2, 1, 3)  # (B, HW, F, C)
        q, k, v = s.qkv(tok).chunk(3, -1)
        sh = lambda t: t.reshape(B * H * W, Fn, s.heads, s.d).transpose(1, 2)  # (N, heads, F, d)
        q, k, v = sh(q), sh(k), sh(v)
        i = torch.arange(Fn, device=x.device)
        age = i[:, None] - i[None, :]  # (F_query, F_key)
        allowed = (age >= 0) & (age <= s.K)
        bias = s.age_bias[:, age.clamp(0, s.K)].to(q.dtype)  # (heads, F, F)
        bias = bias.masked_fill(~allowed[None], float("-inf"))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None])  # (N, heads, F, d)
        a = a.transpose(1, 2).reshape(B, H * W, Fn, C)
        d = s._delta(tok, a)  # (B, HW, F, C)
        has_past = (i > 0).to(d.dtype)[None, None, :, None]  # first tick of the clip: exactly zero delta
        d = (d * has_past).permute(0, 2, 3, 1).reshape(R, C, H, W)
        return x + d.to(x.dtype)

    def forward_stream(s, x, step):
        R, C, H, W = x.shape
        tok = s._tokens(x)  # (R, HW, C)
        q, k, v = s.qkv(tok).chunk(3, -1)
        sh = lambda t: t.reshape(R * H * W, 1, s.heads, s.d).transpose(1, 2)  # (N, heads, 1, d)
        q, k, v = sh(q), sh(k), sh(v)
        past = s.rings.get(step)
        if past is None:  # empty past: exactly A; only store this tick's K / V
            s.rings[step] = (k.detach().clone(), v.detach().clone())
            return x
        kk, vv, n_past = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2), past[0].shape[2]
        ages = torch.arange(n_past, -1, -1, device=x.device)  # oldest .. self(0)
        bias = s.age_bias[:, ages].to(q.dtype)[None, :, None, :]  # (1, heads, 1, n_past+1)
        a = F.scaled_dot_product_attention(q, kk, vv, attn_mask=bias)  # (N, heads, 1, d)
        a = a.transpose(1, 2).reshape(R, H * W, C)
        # append this tick's K / V, keep the last K (clone: bounded storage, no K+1 backing tensor kept alive)
        s.rings[step] = (kk[:, :, -s.K:].detach().clone(), vv[:, :, -s.K:].detach().clone())
        d = s._delta(tok, a)
        return x + d.transpose(1, 2).reshape(R, C, H, W).to(x.dtype)


class WithCausalTemporal(nn.Module):
    """wraps a diffusers Transformer2DModel; .mode in {off, clip, stream}"""
    def __init__(s, inner, c, K):
        super().__init__()
        s.inner = inner; s.temporal = CausalTemporalBlock(c, K); s.config = inner.config
        s.mode, s.n_frames, s.step = "off", None, None

    def forward(s, hidden_states, *args, **kw):
        out = s.inner(hidden_states, *args, **kw)
        x = out[0] if isinstance(out, tuple) else out.sample
        if s.mode == "clip":
            x = s.temporal.forward_clip(x, s.n_frames)
        elif s.mode == "stream":
            x = s.temporal.forward_stream(x, s.step)
        elif s.mode != "off":
            raise ValueError(s.mode)
        if isinstance(out, tuple):
            return (x,) + tuple(out[1:])
        out.sample = x
        return out


def add_causal_temporal(unet, K=4):
    """insert a causal temporal block after each spatial transformer; returns the wrappers (16 for SD 1.5)"""
    from diffusers.models.transformers.transformer_2d import Transformer2DModel
    wrappers = []
    for name, mod in list(unet.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, Transformer2DModel):
                w = WithCausalTemporal(child, child.config.in_channels, K)
                setattr(mod, cname, w); wrappers.append(w)
    return wrappers


def set_mode(wrappers, mode, n_frames=None):
    for w in wrappers:
        w.mode, w.n_frames = mode, n_frames


def set_step(wrappers, step):
    for w in wrappers:
        w.step = step


def reset(wrappers):
    for w in wrappers:
        w.temporal.rings = {}
