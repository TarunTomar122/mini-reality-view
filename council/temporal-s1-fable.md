# Temporal consistency and memory: plan (Fable, round 1)

## 1. Recommendation

**Why it flickers.** The cameras do not pin down what a half-seen car at the picture edge looks like, so each frame
guesses afresh. The fix is to show the model its previous answer: option 2, also the only cheap route to memory.

**Order.**

1. **Free probes first (no training)**: independent frames, SDEdit warm start 0.5 / 0.7 (already running), plus
   previous-frame attention sharing (self-attention also reads the previous frame's keys/values, as in
   Text2Video-Zero, arXiv 2303.13439). They set the bar and validate the metrics. Expected: steadier, no memory, lag.
2. **Previous-frame conditioning (main bet).** Closest published recipe is GameNGen (arXiv 2408.14837): SD 1.4, past
   frames VAE-encoded and concatenated on the latent channels, noise added to them in training (level uniform up to
   0.7, level given to the model). Their ablation: without the noise, quality collapses after 10-20 frames. They used
   teacher forcing only and unfroze the whole UNet; we use LoRA and far less data, and our own DiT needed self_prev, so
   **noise-only vs noise + own frames is the real open question**. Training on own rollouts is what Self Forcing
   (arXiv 2506.08009) argues for.
   - Fit: `conv_in` 4 -> 9 channels (4 noisy latent + 4 previous latent noised to level s + 1 plane holding s; plane 1
     and zeros when there is no previous frame). Old weights copied, new ones zero, as in InstructPix2Pix (arXiv
     2211.09800). `conv_in` trained in fp32 and saved with the checkpoint. At step 0 the model equals the E-winner.
   - Guidance stays on the cameras only: uncond = null camera tokens **with** the previous frame kept, so CFG 2 pushes
     towards what the cameras say and against copying. (GameNGen guides on past frames instead; our truth is the
     cameras.)
   - Cost: same speed as now for teacher forcing; roughly 2.5-3x slower per step with own rollouts (estimate, measure).
     Inference unchanged (one extra conv input).
3. **Deferred.** Option 3 (past camera tokens / BEVFormer): 2-3x more context tokens, and BEVFormer's train-only
   checkpoint will give better features on train drives than on val. Only if rule R4 fires. Option 4 (AnimateDiff,
   arXiv 2307.04725): denoises 16 frames jointly, so it does not stream frame by frame, has no memory past its window,
   and is furthest from Tesla's description. Fallback only.

## 2. Metrics

All on 10 Hz val sequences, render R_j vs key K_j (512x256, own-car mask excluded), Mask2Former as in judge_pair.py.
Key tracks: key car masks linked between consecutive ticks by Hungarian IoU >= 0.5 (eyeball two clips).

- **F1 Presence flips (primary).** Over all (track, j) where the key track exists at j-1 and j: share where the
  render's matched status (IoU >= 0.3) differs between j-1 and j.
- **F2 Car identity change (primary; the owner's complaint).** For tracks matched at j-1 and j: crop the render to the
  key car's box (+10%), resize 96x96, DINOv2-small CLS; AC = mean(1 - cosine). Report AC_render - AC_key. Split picture
  edge thirds vs centre.
- **F3 Warp error (secondary).** RAFT (torchvision) flow on the key pair (K_j, K_j-1); valid pixels = forward-backward
  consistent and key warp error < 10/255. WE = mean |R_j - warp(R_j-1)| on valid pixels; report WE_render - WE_key.
  RAFT on flat game graphics is unverified.
- **F4 Ghosts.** Unmatched render cars linked over time (IoU >= 0.5): count per frame and mean lifetime.
- **Drift over 20 s.** For m in {LPIPS, car F1 and precision per 1 s bin, road IoU}: D_m = mean over 17-20 s minus mean
  over 1-4 s. Scenes change over 20 s, so report **net drift** = D_m(sequential) - D_m(same checkpoint, frames
  independent), paired scene bootstrap. Plot the per-second curves.
- **Placement.** Existing judge on every 2 Hz keyframe inside the rollouts from 2 s on (~1,100 frames on 30 drives),
  plus the single-frame mode (no previous frame). Both vs the E-winner.
- **Lag / copy test.** At 5 s swap the camera tokens to another drive: frames until car F1 against the new drive's
  keys reaches 80% of its steady value.
- **Memory (descriptive, small n).** Recall on key cars whose nuScenes visibility is level 1 at a keyframe but was
  >= 3 earlier in the drive.

**Validate before trusting:** the key video and VAE(key) give the floor; independent frames with a *new* noise seed
per frame must score clearly worst on F1-F3; fixed noise in between. If that ordering fails, fix the metric first.

## 3. Data

- **Train sequences:** E3's 150 drives (tmp/tpv_quarter.txt), every sim tick, whole drive: ~29k frames. New
  `sim_render.py --ticks` flag with `--keys_only`; estimate 3-6 min per drive, ~12 GPU-h of 2080 Ti over 8 shards.
  Tokens ~19 GB RAM (fp16), fine.
- **Diversity guard:** keep the 9,587 keyframe singles from all drives in every batch as the "no previous frame" rows
  (4 of 16). They double as previous-frame dropout.
- **Val:** the 30 judge drives, whole drive at 10 Hz (~6k frames). Dev subset of 10 (incl. 0915 / 0563 / 0632) for
  sweeps; all 30 only for final numbers.
- **Pairing:** define tick time through the keyframes, not `t0 + 0.1 j`: tick 5k+i <-> T_k + (i/5)(T_k+1 - T_k)
  (keyframes are not exactly 0.5 s apart; **check ScenarioNet's converter interpolates this way**, not confirmed
  online). Per camera take the sweep nearest in time; drop the tick if any camera is > 45 ms off (< 0.45 m at 10 m/s,
  below the judge's 1-2 m resolution). Store `(scene, tick, time, 6 files)`.
- Previous frame = tick-1 (80%) or tick-2 (20%), covering 12 Hz run time and dropped frames.

## 4. Experiments

All start from the E1/E2/E3 winner, bs 16 = 12 sequence rows + 4 singles, lr 1e-4, style 2, s ~ U[0, 0.7] in training.

| Arm | What | Budget |
|---|---|---|
| **P** | No training: independent; SDEdit 0.5 / 0.7; previous-frame attention sharing | ~2 GPU-h rendering + scoring |
| **A** | Previous latent on channels, teacher-forced (true previous key + noise) | 3 h, ~11k steps; save 2k / 5k |
| **B** | A's 2k checkpoint, then 6 of the 12 sequence rows use the model's **own** previous frame: 8-step DDIM, CFG 2, no grad, from cameras at tick-1 and the frame at tick-2. Each rollout is cached per (drive, tick) and reused as the tick-2 input half the time, so chains deepen for free | 3 h, maybe ~4-5k steps |
| **C** (only if R4) | B + camera tokens from 0.5 s and 1.0 s ago with a time embedding | 3 h |

Eval per arm: 20 s rollouts on dev drives at inference s in {0, 0.1, 0.3}, same fixed sampling noise each frame; best
s on all 30. Compare A and B at equal wall-clock and at equal steps (A@5k).

**Decision rules** (thresholds are my judgement; adjust once the floors are known):

- **R1 Placement gate (all arms):** car F1 difference vs E-winner has a 95% CI lower bound above -0.03 and road IoU
  falls by < 0.01, in rollouts and in single-frame mode.
- **R2 Adopt a free probe** as the default if F1 flips and F2 fall >= 30% vs independent, net LPIPS drift <= +0.01,
  lag <= 5 frames, R1 holds. It then becomes the bar for A / B.
- **R3 Adopt previous-frame conditioning** if the best of A / B cuts F1 flips and F2 by >= 50% vs independent and by
  >= 25% vs the best probe, net drift over 20 s is <= +0.02 LPIPS and >= -0.03 car F1, ghosts do not grow over time,
  lag <= 5 frames, R1 holds. Choose **A** if it passes alone (simpler, ~2.5x cheaper; agrees with GameNGen). Choose
  **B** if A fails drift or ghosts and B passes, or if B's net drift is lower with a CI excluding 0. Winner goes into
  the long run.
- **R4 Run C** only if the winner passes R3 but memory recall is no better than independent frames.

## 5. Risks and stop conditions

- **Copying the previous frame.** In teacher forcing the previous key is always right, so "trust it over the cameras"
  is a cheap solution: lag, and ghost cars that never leave. Lag test, F4 and R1 catch it. One fix attempt
  (s ~ U[0.3, 0.9], singles 4 -> 8 rows); if lag stays > 10 frames, stop.
- **Placement loss from 150 drives.** If E1 clearly beats E3, render 300 drives (10 s windows) before arm A.
- **Too few steps in B** (~4k). If B is still improving and ahead on drift, extend it rather than conclude.
- **Training rollouts are short** (1-2 frames deep, 8 steps); run time is 240 frames at 25 steps. The gap shrinks but
  does not close.
- **Metrics rest on a Cityscapes segmenter and RAFT on game graphics.** If the validation ordering fails, fall back to
  side-by-side clips plus drift curves.
- **Sim interpolation** is linear between 2 Hz labels, so turning cars slide slightly in the keys. Accepted.
- **Memory may stay short**: one previous frame carries only what the last picture showed.
- **Call it off** if neither A nor B beats the best free probe by the R3 margins after the one fix attempt: ship the
  probe and spend the time on placement.

## Sources (existence checked)

GameNGen arXiv 2408.14837 (method read in the paper) · Self Forcing arXiv 2506.08009 · Text2Video-Zero arXiv
2303.13439 · InstructPix2Pix arXiv 2211.09800 · AnimateDiff arXiv 2307.04725 · Lai et al., Learning Blind Video
Temporal Consistency, arXiv 1808.00449 (warp error). Not confirmed: ScenarioNet's exact interpolation.
