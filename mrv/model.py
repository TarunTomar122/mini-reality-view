"""Small DiT renderer: camera tokens (cross-attention) + noisy latent (+ optional noisy previous frame) -> velocity."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos_2d(h, w, d):
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    om = 1.0 / 10000 ** (torch.arange(d // 4, dtype=torch.float32) / (d // 4))
    out = [torch.sin(x.flatten()[:, None] * om), torch.cos(x.flatten()[:, None] * om),
           torch.sin(y.flatten()[:, None] * om), torch.cos(y.flatten()[:, None] * om)]
    return torch.cat(out, 1)  # (h*w, d)


def t_embed(t, d):
    half = d // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * 1000 * f[None]
    return torch.cat([torch.cos(a), torch.sin(a)], 1)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Attn(nn.Module):
    def __init__(s, d, heads):
        super().__init__()
        s.h = heads
        s.q, s.kv, s.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(s, x, ctx):
        B, N, D = x.shape
        q = s.q(x).view(B, N, s.h, -1).transpose(1, 2)
        k, v = s.kv(ctx).view(B, ctx.shape[1], 2, s.h, -1).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return s.o(y.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(s, d, heads):
        super().__init__()
        s.n1, s.n2, s.n3 = (nn.LayerNorm(d, elementwise_affine=False, eps=1e-6) for _ in range(3))
        s.sa, s.ca = Attn(d, heads), Attn(d, heads)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        s.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(s.ada[1].weight); nn.init.zeros_(s.ada[1].bias)

    def forward(s, x, ctx, c):
        a = s.ada(c).chunk(9, 1)
        h = modulate(s.n1(x), a[0], a[1]); x = x + a[2][:, None] * s.sa(h, h)
        h = modulate(s.n2(x), a[3], a[4]); x = x + a[5][:, None] * s.ca(h, ctx)
        h = modulate(s.n3(x), a[6], a[7]); x = x + a[8][:, None] * s.mlp(h)
        return x


class Renderer(nn.Module):
    def __init__(s, d=384, depth=12, heads=6, lat_ch=4, lat_hw=(18, 32), patch=2, prev=False,
                 tok_dim=384, n_cam=6, tok_hw=(18, 32), tok_pool=2, n_styles=0, rays=False):
        super().__init__()
        s.p, s.lat_ch, s.lat_hw, s.prev, s.tok_pool = patch, lat_ch, lat_hw, prev, tok_pool
        in_ch = lat_ch * (2 if prev else 1)
        s.inp = nn.Linear(in_ch * patch * patch, d)
        gh, gw = lat_hw[0] // patch, lat_hw[1] // patch
        s.register_buffer("pos", sincos_2d(gh, gw, d)[None], persistent=False)
        th, tw = tok_hw[0] // tok_pool, tok_hw[1] // tok_pool
        s.tok_hw = tok_hw
        s.tok_in = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, d))
        s.register_buffer("tok_pos", sincos_2d(th, tw, d)[None, None], persistent=False)
        s.cam_emb = nn.Parameter(torch.zeros(1, n_cam, 1, d))
        # experiment A: per-patch viewing ray (Plücker d, o x d in the ego frame) -> embedding
        s.ray_mlp = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d)) if rays else None
        s.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        s.s_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d)) if prev else None
        s.style = nn.Embedding(n_styles, d) if n_styles else None
        if n_styles:
            nn.init.zeros_(s.style.weight)
        s.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        s.nf = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        s.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        s.out = nn.Linear(d, lat_ch * patch * patch)
        nn.init.zeros_(s.ada_f[1].weight); nn.init.zeros_(s.ada_f[1].bias)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)

    def context(s, tokens, rays=None):
        """tokens: (B, 6, 576, 384) -> (B, 6*144, d) after 2x2 pooling. rays: (B, 6, 144, 6) Plücker rays."""
        B, C, N, D = tokens.shape
        h, w = s.tok_hw
        if N == h * w:  # full 32x18 grid: pool here
            t = tokens.reshape(B * C, h, w, D).permute(0, 3, 1, 2)
            t = F.avg_pool2d(t, s.tok_pool).flatten(2).transpose(1, 2).reshape(B, C, -1, D)
        else:  # already pooled at encode time
            t = tokens
        t = s.tok_in(t) + s.tok_pos + s.cam_emb
        if s.ray_mlp is not None and rays is not None:
            t = t + s.ray_mlp(rays)
        return t.flatten(1, 2)

    def forward(s, x, t, tokens, prev=None, s_noise=None, ctx=None, style=None, rays=None):
        B = x.shape[0]
        if s.prev:
            x = torch.cat([x, prev], 1)
        p, (H, W) = s.p, s.lat_hw
        h = x.unfold(2, p, p).unfold(3, p, p)  # B,C,H/p,W/p,p,p
        h = h.permute(0, 2, 3, 1, 4, 5).reshape(B, (H // p) * (W // p), -1)
        h = s.inp(h) + s.pos
        c = s.t_mlp(t_embed(t, 256))
        if s.prev:
            c = c + s.s_mlp(t_embed(s_noise, 256))
        if s.style is not None and style is not None:
            c = c + s.style(style)
        if ctx is None:
            ctx = s.context(tokens, rays)
        for blk in s.blocks:
            h = blk(h, ctx, c)
        sh, sc = s.ada_f(c).chunk(2, 1)
        h = s.out(modulate(s.nf(h), sh, sc))
        h = h.view(B, H // p, W // p, s.lat_ch, p, p).permute(0, 3, 1, 4, 2, 5)
        return h.reshape(B, s.lat_ch, H, W)
