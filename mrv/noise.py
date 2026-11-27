"""Training noise for the causal adapter (council/AGREED-causal-r1.md; amendment A3 "vpriv").
Every clip first gets the AGREED-r1 mixture sqrt(rho) e_clip + sqrt(1-rho) e_tick, computed exactly as the r1 trainer did
(Python-float scalars); the private clip(s) are then REPLACED by e_tick alone (no shared component). So shared clips are
bit-identical to vlive's noise and private_clips = 0 is exactly the r1 formula (Astra vpriv review item 3).
private_clips = 1: the private slot alternates each update (step % clips); private_clips >= clips: all clips private.
"""


def build_eps(eps_clip, eps_own, rho, private_clips=0, step=0):
    """eps_clip: (C, 1, 4, H, W); eps_own: (C, F, 4, H, W) -> (C * F, 4, H, W)"""
    C, F = eps_own.shape[:2]
    eps = rho ** 0.5 * eps_clip + (1 - rho) ** 0.5 * eps_own  # exactly the r1 expression
    if private_clips:
        eps = eps.clone()
        slots = range(C) if private_clips >= C else [step % C]
        for s in slots:
            eps[s] = eps_own[s]
    return eps.reshape(C * F, *eps_own.shape[2:])
