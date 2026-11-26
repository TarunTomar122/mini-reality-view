# Temporal consistency and memory: round 2 (Fable)

Written without reading Astra's round 2. Speeds and costs are estimates until timed.

## 1. Where I agree with Astra (and what I change)

- **Same bet and order:** frozen warm start, then a previous-picture input, then training on the model's own frames.
  Input memory (past camera tokens, BEVFormer) and AnimateDiff wait for a later round. I drop my arm C.
- **A control arm, which I lacked.** Continuing on the dense targets without history separates the gain from history
  from the gain of more updates. Adopted as arm A.
- **Separate zero-initialised conv beside a frozen `conv_in`.** Same function as my 4 -> 9 widening; base UNet
  untouched. Adopted.
- **Cached 25-step rollouts for self-training.** Eight-tick chains reach depth 7 with the run-time sampler; my online
  8-step version reached depth 1-2. Adopted.
- **Equal updates, identical target draws, two paired seeds, whole-drive bootstrap.** Adopted.
- **Development drives from the train split**, so nothing is tuned on val. Better than my dev subset of val.
- Also adopted: drift as a late gap to the same model run cold; the memory-event definition; the pairing audit.
- Dropped from my plan: the attention-sharing probe and the tick-1 / tick-2 mix.

## 2. Where I disagree, and why

1. **Pixel-only flicker score.** Astra's car warp residual has two blind spots. A car absent in both frames scores a
   perfect zero, so erasing edge traffic is rewarded. And this project found that "pixel scores reward blur" (README);
   a sedan turning into an SUV of the same colour moves few pixels, and that was the owner's complaint. I keep my
   identity-change score as co-primary.
2. **L1 as the drift gate.** Same lesson. Gate on LPIPS and car F1; report L1 to compare with the old DiT (+1.8).
3. **One fixed history-noise level (0.3).** Our DiT: 0.1 drifts, 0.3 helps. GameNGen: a range up to 0.7 with the level
   given to the model. A fixed level makes a wrong guess cost a whole arm; a range plus one input plane lets us sweep
   it at run time for free. I adopt Astra's corruption formula.
4. **4,000 updates at 5e-5 for a zero-initialised layer.** If B looks useless we cannot tell it from undertraining.
   6,000 fit the cap, and the 4,000 checkpoint is scored, so Astra's setting is nested.
5. **Dense-only batches from ~120 drives.** Placement was learnt from 9,587 keyframes; I keep 4 keyframe rows from all
   train drives in every batch, in every arm.
6. **No automatic copy test.** One camera-swap rollout per method gives a number instead of an impression.

## 3. The plan I would sign

**Data (2080 Ti, extra budget).** From E3's 150 drives (tmp/tpv_quarter.txt): 6 development drives chosen by seed
across traffic density and turning, 144 training drives. Val = the 30 judge drives. Whole drives (~20 s) at 10 Hz,
keys only, ~36k keys, judge's 32-row crop kept; lists frozen first. Pairing: tick time linear between real sample
timestamps; nearest sweep per camera; reject > 50 ms; break sequences at rejects and repeated bundles; manifest of
`(scene, tick, time, 6 files, offsets)`; pairs sampled by manifest.

**Frozen methods:** independent with fixed noise; warm start 0.5 and 0.7 (13 / 17 steps); independent with a new seed
every frame (metric check only). Pick the warm strength on development drives, lock it, run val once.

**Three trained arms** from the E-winner: 6,000 updates, checkpoints at 2k / 4k / 6k, batch 16 = 12 dense rows + 4
keyframe singles (all train drives except development), LR 5e-5 for LoRA and adapter (500 warm-up steps), 1e-4 for the
new conv, rank 64, epsilon loss, 10% camera dropout, style 2. Cap 180 slice-min including caches; time 100 steps and
one cache shard first; cut all arms equally if needed.

| Arm | What |
|---|---|
| **A** | Dense targets, no history (control) |
| **B** | History conv, 5 inputs: previous key latent corrupted as sqrt(1 - s^2) z + s eps, plus a plane holding s; s ~ U[0.1, 0.5]. No history (singles and 10% of dense rows) = zeros with plane 1. Both guidance branches get the same history |
| **C** | Fork B at 1,000. At 1,000 / 2,500 / 4,000 cache detached DDIM-25 rollouts of 128 random eight-tick train clips; half of the history rows use the cached own frame paired with its true next key |

Evaluation: DDIM 25, CFG 2, fixed sampling and history noise per drive, first tick without history. Sweep run-time s
in {0.1, 0.3, 0.5} on development drives, lock, then val. Second seed only for the methods that decide. About 1.5
GPU-h per method per seed on 30 drives.

**Metrics** (own car excluded, drives weighted equally):

- **W, car warp residual (co-primary):** Astra's definition unchanged, plus the outer 20% each side and the valid
  car-pixel share.
- **ID, identity change (co-primary):** key car tracks linked across ticks (Hungarian, IoU >= 0.5). For tracks the
  render matches (IoU >= 0.3) at both ticks: crop the render to the key box + 10%, 96x96, DINOv2-small CLS,
  mean(1 - cosine), minus the same number on the keys.
- **Flips (secondary):** share of consecutive track ticks where matched status changes. **Ghosts:** unmatched render
  cars per frame, and their lifetime.
- **Drift:** late gap = last 2 s of the rollout minus the same frames rendered cold, for LPIPS, car F1 and L1; early
  gap (2 s from tick 5) alongside, to tell a constant offset from growth. Clips >= 19 s only.
- **Placement:** judge_pair.py on every fifth tick of the rollout, and with no history.
- **Lag:** swap cameras to another drive at 5 s; ticks until car F1 against the new keys reaches 80% of its steady
  value. Development drives.
- **Memory:** recall on Astra's pre-marked events vs the same tick cold. Descriptive; no claim if events are few.

**Decision rules** (paired bootstrap over drives, 2,000 draws, seeds kept together):

- **Gates, every method:** car F1 lower bound >= -0.02 and road IoU >= -0.01 vs frozen independent; late gap upper
  bound <= +0.02 LPIPS and car F1 lower bound >= -0.03; lag <= 5 ticks; ghosts not growing.
- **Adopt warm start** if W and ID both fall >= 15% vs frozen independent, intervals excluding zero, gates pass.
- **Adopt B or C** if the same holds against both arm A and the best qualifying frozen method.
- **B** if both qualify; **C over B** only for >= 0.01 less late LPIPS gap or >= 10% less W or ID, interval excluding
  zero, placement kept. **Keep A** only for a confirmed placement gain with flicker no worse.
- The owner's side-by-side clips on 0915 / 0563 / 0632 can veto a pass, not grant one.

**Stop rules:**

- Metric check fails (key video and VAE round trip must score best, new-seed-per-frame worst; hand check of ~20 key
  masks; flow keeps too few edge-car pixels): repair the measurement before ranking.
- Dense rendering costs over twice the estimate: cut to 100 training drives, keep clip length.
- Lag over 10 ticks: one fix (s ~ U[0.3, 0.7], history dropout 40%), then stop that arm.
- Neither B nor C clears the rules: ship the warm start if it qualified, else stay independent; the next round goes to
  input memory.
- Wide intervals leave the choice open, and we say so.

## 4. Remaining disagreements

The six points in section 2. Thresholds are judgement; I took Astra's where we differed.
