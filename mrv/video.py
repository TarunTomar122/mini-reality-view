"""Joint multi-frame video model (council/video-next-*.md, Fable's plan, both reviewers): zero-initialised temporal
attention blocks after each of the 16 spatial transformers of the frozen E3 UNet. Frames pass through the 2D UNet as a
batch (each keeps its own camera tokens); after every spatial transformer, a temporal block lets each pixel attend over
the F frames of its clip. Output projection is zero-initialised, so with fresh blocks the model equals E3 exactly.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def frame_pos(n, d, device):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(d // 2, device=device, dtype=torch.float32)[None]
    ang = pos / (10000 ** (2 * i / d))
    return torch.cat([ang.sin(), ang.cos()], 1)  # (n, d)


class TemporalBlock(nn.Module):
    def __init__(s, c, heads=8):
        super().__init__()
        s.norm = nn.GroupNorm(32, c)
        s.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        s.ln = nn.LayerNorm(c)
        s.ff = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))
        s.out = nn.Linear(c, c)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)

    def forward(s, x, n_frames):
        """x: (B*F, C, H, W) -> same; attention over F for every (b, h, w)"""
        BF, C, H, W = x.shape
        B = BF // n_frames
        h = s.norm(x.float()).to(x.dtype)
        h = h.view(B, n_frames, C, H * W).permute(0, 3, 1, 2).reshape(B * H * W, n_frames, C)
        h = h + frame_pos(n_frames, C, x.device).to(h.dtype)[None]
        a, _ = s.attn(h, h, h, need_weights=False)
        h = h + a
        h = h + s.ff(s.ln(h))
        h = s.out(h)
        h = h.view(B, H * W, n_frames, C).permute(0, 2, 3, 1).reshape(BF, C, H, W)
        return x + h


class WithTemporal(nn.Module):
    """wraps a diffusers Transformer2DModel; .n_frames must be set before each UNet call (1 / None = off)"""
    def __init__(s, inner, c):
        super().__init__()
        s.inner = inner; s.temporal = TemporalBlock(c); s.n_frames = None
        s.config = inner.config

    def forward(s, hidden_states, *args, **kw):
        out = s.inner(hidden_states, *args, **kw)
        x = out[0] if isinstance(out, tuple) else out.sample
        if s.n_frames and s.n_frames > 1:
            x = s.temporal(x, s.n_frames)
        if isinstance(out, tuple):
            return (x,) + tuple(out[1:])
        out.sample = x
        return out


def add_temporal(unet):
    """insert a temporal block after each spatial transformer; returns the list of wrappers"""
    from diffusers.models.transformers.transformer_2d import Transformer2DModel
    wrappers = []
    for name, mod in list(unet.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, Transformer2DModel):
                w = WithTemporal(child, child.config.in_channels)
                setattr(mod, cname, w); wrappers.append(w)
    return wrappers


def set_frames(wrappers, n):
    for w in wrappers:
        w.n_frames = n
