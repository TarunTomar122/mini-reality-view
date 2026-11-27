# AGREED (causal / live video, r1) — Astra + Fable, 27 Nov 2026 ~19:20

Sources: council/causal-video-brief.md, causal-video-s1-{astra,fable}.md, causal-video-s2-{astra,fable}.md, causal-video-r3.md,
causal-video-r3-astra.md. Owner: "i want the ability to get live view". Bars below are fixed BEFORE any result; changing one
needs a new council round, never a quiet edit. Every code change is checked by Astra against this file before it trains.

## Design
- Base: FROZEN A@20k (runs/longA/ckpt_20000.pt: SD 1.5 UNet + A's LoRA r64 + A's CameraAdapter), unet.eval(); only 16 new
  temporal blocks train (one after each spatial transformer). Asserted: trainable params == temporal params; LoRA + adapter
  bit-identical after the first update.
- Temporal block: per-frame GroupNorm(32) -> per-pixel tokens; own q/k/v Linear, 8 heads; causal band mask: query tick i reads
  ticks j with 0 <= i - j <= K, K = 4 (self + 4 past), additive per-head age bias (8 x 5, zero init); +attn, +FF(LayerNorm),
  zero-init output Linear; rows with no past -> exactly zero temporal delta. Modes off / clip / stream.
- Stream (live): rings per (block, DDIM step index) of the last K ticks' K/V, cond and uncond as separate rows; read past + self,
  then append, evict to K; hard break -> clear all rings; camera swap keeps them. 0 extra UNet passes vs A.
- Training: 2 clips x 16 consecutive usable ticks (same sequence, no hard break) of the 144 train drives; ONE diffusion
  timestep per clip, U{0..999}; eps_i = sqrt(rho) eps_clip + sqrt(1-rho) eps_i,private; epsilon MSE (fp32) averaged over ticks
  1..15 (tick 0 = context); whole-clip camera dropout 10%; style 2; AdamW 1e-4, wd 0, 500 warm-up, grad clip 1.0, bf16
  autocast, fp32 temporal params. No stale swaps, no generated prefixes, no clean past.
- Rollout: tick loop, current cameras only; x_T(tick) = sqrt(rho) z0 + sqrt(1-rho) z_tick (z0 = the signed per-(drive, seed)
  CRC32 tensor; z_tick keyed drive|seed|absolute tick); rho read from the checkpoint; DDIM 25, eta 0, CFG 2; signed rollout
  fields + K, rho, mode.
- rho (pre-registered, chosen BEFORE update 1): render `--off` (frozen A, blocks off) at rho 0.5 and 0.8 on the 6 dev drives,
  seeds 0/1. Choose 0.5 if both ID and W are <= 1.20 x signed A@20k's; else 0.8 if it meets both; else back to council.
  Never 1.0.
- Logging: every 100 updates loss, grad norm, out-norm, age-bias size, step_s, data_s, peak GB, batch_sha, code_sha;
  args.json; checkpoints store K, rho, init, code_sha.
- Copy probe every 500 updates: paired losses on 8 fixed dev clips at t = 900 / 700 / 300 with blocks on, blocks off, another
  drive's past, current cameras swapped (targets and draws fixed). At t >= 700, flag copying if swapping cameras raises the
  loss less than replacing the past.

## Tests before update 1 (all must pass; Astra signs the diff + results)
T1 fresh / off == A exactly (max |diff| 0). T2 `--off` at rho 1.0 reproduces one dev drive's signed A@20k latents (< 1e-2,
matched noise, dtype, batch layout). T3 future cameras never change past outputs (and later outputs do change). T4 stream ==
causal clip forward with nonzero weights through eviction and after a reset (< 1e-4, fp32). T5 60-tick rollout with cameras +
private draws of ticks 30..59 replaced: ticks 0..29 bit-identical. T6 `--reset_at 20` suffix == a fresh run started at
tick 20. T7 frozen parts unchanged after an optimizer step; only temporal params train. Timing: 100 updates on the A6000
(step_s, peak GB) before scheduling.

## Gate (6 dev drives x seeds 0/1, signed temporal_metrics 764dd3ec95db + temporal_compare fc7ee2b886a5, floor 0)
Checks at 2k / 4k / 8k (12k only if safe at 8k and the smaller reduction improved from 4k). Candidate first, signed A@20k
second, A-derived k=3 count list, cold_ref B:
- Smoothness: ID and W both eligible, the smaller paired reduction strictly > 9.4%.
- Accuracy: car F1 >= A@20k - 0.03.
- Drift: late LPIPS-to-key gap vs cold (= `--off` at the chosen rho, matched cameras and noise) <= +0.03 on eligible ticks
  150-169.
Advance the earliest joint pass to one 30-drive exam vs A@20k. Stop after two consecutive safety misses (F1 or drift). No pass
by 12k (or 8k if not extended) -> close the pilot, back to council. Report contributing drives, paired drive-bootstrap CIs,
leave-one-drive-out sensitivity, lingering ghosts, recall, early vs late gap, and the copy warnings: per-tick latent change
< 0.7 x A's or falling; `--swap` lag > 3 ticks; long minus short memory (`--reset_at 134`) > +0.01 on ticks 150-169.
"Live" claims only after a 2080 Ti timing run (batch one, p50 / p95 camera-to-display, sustained fps, peak VRAM).
