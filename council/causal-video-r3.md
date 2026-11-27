# Causal video, round 3 (27 Nov ~19:15): settle the noise-sharing point, then sign AGREED

Read council/causal-video-s2-astra.md and council/causal-video-s2-fable.md. You agree on: K = 4, ONE diffusion level per clip
(past as noisy as current), own causal self-attention with per-head age bias (8 x 5, zero init), per-DDIM-step ring cache with
cond / uncond as separate rows, frozen A@20k (unet.eval()), checks at 2k / 4k / 8k (12k) with the same bars.

Open point (Fable s2 point 5): Astra's checklist uses ONE Gaussian per clip broadcast over all ticks (rho = 1, to match A's
signed renders, which reuse one noise tensor per drive/seed at every tick). Fable argues rho = 1 makes x_t(i) - x_t(j) =
sqrt(abar)(x0(i) - x0(j)) noise-free at every level, so live generation starting from identical noise reads "nothing moved"
(freeze); Fable proposes eps = sqrt(rho) eps_clip + sqrt(1 - rho) eps_own with rho < 1, the SAME rho in trainer and rollout
(x_T(n) = sqrt(rho) z0 + sqrt(1 - rho) z_n, z_n keyed drive|seed|tick); smoothness / F1 comparator stays A@20k's signed renders;
the late-gap cold = `--off` at the same rho; rho fixed BEFORE training by rendering `--off` at 0.5 and 0.8 on dev: 0.5 unless
off@0.5 is > 20% worse than A on ID or W, then 0.8, else back to council; never 1.0.

Astra: agree / object / amend in <= 12 lines (council/causal-video-r3-astra.md), including the exact pre-registered rho rule
and what "cold" is for the late gap. Also confirm or object: loss over ticks 1..15 vs all frames (tick 0's temporal delta is
exactly zero by construction, so it only rescales the loss); extra tests T2 (off at rho 1.0 == A's signed render on one drive,
< 1e-2), T5 (60-tick rollout, cameras + z_n of ticks >= 30 replaced: ticks < 30 bit-identical), T6 (--reset_at 20 == a run
started at tick 20), and a copy probe every 500 updates on 8 fixed dev clips (loss with blocks on / off / past from another
drive / own cameras swapped). Read-only; reply with a 3-line summary.
