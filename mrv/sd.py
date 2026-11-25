"""Shared pieces of the Stable Diffusion 1.5 renderer (used by scripts/train_sd.py and scripts/try_my_photos.py)."""
import torch
import torch.nn as nn
from mrv.model import sincos_2d

LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]


class CameraAdapter(nn.Module):
    def __init__(s, d=768, n_cam=6, layers=2):
        super().__init__()
        s.inp = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, d))
        s.ray = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        s.cam = nn.Parameter(torch.zeros(1, n_cam, 1, d))
        s.register_buffer("pos", sincos_2d(9, 16, d)[None, None], persistent=False)
        s.style = nn.Embedding(2, d)
        s.null = nn.Parameter(torch.zeros(1, n_cam * 144 + 1, d))
        enc = nn.TransformerEncoderLayer(d, 12, 4 * d, batch_first=True, norm_first=True, activation="gelu")
        s.mix = nn.TransformerEncoder(enc, layers)
        s.out = nn.LayerNorm(d)

    def forward(s, tok, rays, style, drop=None):
        B = tok.shape[0]
        h = s.inp(tok) + s.ray(rays) + s.cam + s.pos  # (B,6,144,d)
        h = torch.cat([h.flatten(1, 2), s.style(style)[:, None]], 1)
        h = s.out(s.mix(h))
        if drop is not None:
            h = torch.where(drop[:, None, None], s.null.expand(B, -1, -1).to(h.dtype), h)
        return h
