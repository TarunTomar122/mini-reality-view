"""Training noise for the causal adapter (council/AGREED-causal-r1.md; amendment A3 "vpriv").
eps for C clips x F ticks: shared clips get sqrt(rho) e_clip + sqrt(1-rho) e_tick; private clips get e_tick only (no shared
component). With private_clips = 0 this is EXACTLY the AGREED-r1 formula (bit-identical to the pre-A3 trainer).
With private_clips = 1 the private slot alternates each update (step % clips); with private_clips >= clips all clips are private.
"""
import torch


def build_eps(eps_clip, eps_own, rho, private_clips=0, step=0):
    """eps_clip: (C, 1, 4, H, W); eps_own: (C, F, 4, H, W) -> (C * F, 4, H, W)"""
    C, F = eps_own.shape[:2]
    if not private_clips:
        return (rho ** 0.5 * eps_clip + (1 - rho) ** 0.5 * eps_own).reshape(C * F, *eps_own.shape[2:])
    mix = torch.full((C, 1, 1, 1, 1), rho)
    if private_clips >= C:
        mix[:] = 0.0
    else:
        mix[step % C] = 0.0
    return (mix.sqrt() * eps_clip + (1 - mix).sqrt() * eps_own).reshape(C * F, *eps_own.shape[2:])
