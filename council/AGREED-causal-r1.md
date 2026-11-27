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

## Amendment A1 (owner directive, 27 Nov ~19:00; supersedes the stop timing above)
Owner: "just make sure we dont think something is bad after a short training run these things might need more runs".
- 2k and 4k checks are INFORMATIONAL: full signed scores + videos are reported, but no stop / reject decision is taken on them.
- The pass / safety bars are unchanged; they are first applied at 8k. Two consecutive safety misses (F1 or late gap) counted
  from 8k on (8k, 12k, 16k, ...) stop the run; a pass at 8k or later advances to the exam.
- If the run is trending toward the bars at 8k (smaller reduction rising, F1 / late gap improving), extend to 12k and 16k before
  any "no pass" verdict; only a clearly broken run (NaN, crash, frozen-check failure, copy alarm with rising late gap at 8k+)
  stops earlier.
- The council is informed at the next hourly review; changing these bars again needs the owner or a council round.

## Amendment A2 (rho fallback, signed by Astra and Fable 27 Nov ~20:25 UTC BEFORE the rho 0.8 scores were known)
Sources: council/causal-video-r4-fallback.md, causal-video-r4-{astra,fable}.md, causal-video-r4-swap{,-astra,-fable}.md.
1. rho 0.8 meets the 1.20 rule on admissible evidence -> train at 0.8, nothing else changes.
2. rho 0.8 fails on admissible evidence -> use the rho 0.9 and 0.95 off renders (made the same way, reviewed code, verified)
   and train at the LOWEST of them meeting both ratios <= 1.20. Hard cap 0.95 (never higher; never 1.0). Bug guard: if a higher
   rung scores worse than a lower one on both measures, stop (suspected render / scoring bug) and back to council.
3. Neither 0.9 nor 0.95 meets (or their evidence is not ready 90 min after the 0.8 failure) -> train at rho 0.8 under a
   RECORDED OVERRIDE, bars and comparator unchanged, only if: 0.8 evidence admissible; off@0.8 car F1 >= A@20k - 0.03; 0.8 not
   worse than 0.5 on both ID and W. Otherwise back to council.
4. The fallback code path (decision modes rule / fallback / override; trainer accepts them) needs the test suite re-passed and
   Astra's code signature before any fallback launch. Every check also reports reductions vs the model's own cold baseline.

## Amendment A3 — DRAFT, NOT CONVERGED (Neon guard 22:30: the builder misread Astra's swap note, which signed a computed lambda,
## pairs i = 5..15 and probe-gated queueing; the text below enacted Fable-only terms). Not binding until both seats sign one text.
## (original header: next arm; pre-registered 27 Nov ~22:00 UTC BEFORE any vlive 8k number)
Sources: council/causal-video-a3*.md (proposals, swap, lambda). Probe evidence at the time: scripts/noise_probe.py on vlive
ckpt_2000 (the blocks use the 95%-shared past noise, not past content; pulled/noise_2000.json).
- vlive runs unchanged to 16k as the control (A1 bars and timing).
- TRIGGER: vlive finishes 16k with no joint pass (any of 8k / 12k / 16k) AND both safety bars (F1, late gap) met at 16k.
  Probes are diagnostic only. Same single H200-3g slice (no second slice without the live-VSR session's OK).
- ARM: fresh from frozen A@20k, rho 0.95, K = 4, everything as AGREED r1 except the loss: MSE (ticks 1..15) + lambda x
  adjacent-tick difference loss mean over pairs i = 2..15 of ((pred_i - pred_{i-1}) - (eps_i - eps_{i-1}))^2. No Min-SNR.
- lambda = 1.0 fixed (Fable's signed rule; Astra's original proposal; Astra's latest note signed the computed ratio 0.928,
  within 7% -> Astra confirms 1.0 when signing the A3 code). Launch check: the first batch's ratio difference-term /
  MSE-term must lie in [0.5, 2], else no launch and a council round. Record lambda, the probe ratio (per t) and the probe
  file's sha256 in the launch note / config; log both terms at every log step.
- Before launch: T1-T7 + new fixtures for the difference loss, Astra's code signature. Bars and A1 timing unchanged.

## Amendment A3 (CONVERGED, supersedes the DRAFT above; Astra + Fable, written 27 Nov ~22:45 UTC (earlier header said ~22:55: future-dated, Neon), before vlive's 8k / 16k panels)
Sources: council/causal-video-a3r2.md, causal-video-a3r2-{astra,fable}.md, causal-video-a3r2-swap{,-astra,-fable}.md.
Evidence at the time: V2k jumpier than A and own cold; offline noise probes @2k / @8k (blocks rely on shared past noise, more
so at 8k); gradient-cosine probe (pulled/gradcos_*.json) informational.
- vlive runs unchanged to 16k (control; A1).
- TRIGGER: vlive has no joint pass at 8k, 12k or 16k AND both safety bars (F1, late gap) are met at 16k.
- ARM "vpriv": fresh from frozen A@20k, vlive's seed and draws, plain MSE (ticks 1..15), K = 4, everything as AGREED r1 except
  training noise: BOTH clips of every update use fully private noise per tick (no shared component) -- unless vlive's signed
  16k panel has ID AND W strictly lower than its own cold (off at 0.95), in which case ONE clip per update is private and
  the private slot alternates each update. Rollout noise unchanged (rho 0.95, from the A2 decision); the training mixture is
  stored in the checkpoint separately from the rollout rho. Difference loss (old draft arm) rejected.
- Before launch: T1-T7 + new fixtures (private-noise batches; alternating slot; resume compatibility) + Astra code signature.
- Reports (no gate): the noise probe fed with shared and with private noise at each check. Cleaner-past design = next full
  council round if vpriv also fails. History guidance (blocks off on the uncond row) = informational side probe.
