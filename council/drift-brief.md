# Council brief: the memory model drifts late in long drives (27 Nov 12:10)

Owner question (verbatim): "oh shit why the gap increasing can we add something to fix this gap lol, can u run a research round with council on it"

Read first: PROGRESS.md (top entries from 27 Nov), council/PLAN-temporal.md, scripts/train_temporal.py, scripts/temporal_rollout.py
(history method, `--method hist`), scripts/temporal_metrics.py (signed 764dd3ec95db, do not edit). Owner rules that bind any plan:
the goal is a SMOOTH video (no cars jumping between frames); camera-only at run time; one candidate trained long + a control,
cheap 5k dev checks, one exam at the end; "spend more compute testing one good thing than 3 okay things"; "if something is
proven in research and doesn't work for us then maybe we are not doing it correctly instead of jumping to a new thing";
fake cars are measured but do not block now; account limit 20 cluster jobs (1 A6000 for training, ~18 x 2080 Ti for rollouts).

## Model
SD 1.5 + LoRA r64 + CameraAdapter (DINOv2 tokens of 6 cameras + rays) renders a 512x256 chase-view frame (latent 4x32x64),
DDIM 25, CFG 2. Memory ("history") = a zero-init 3x3 conv beside conv_in reading 5 channels: the previous rendered latent,
corrupted sqrt(1-s^2) z + s eps, plus a plane holding s. Training s ~ U[0, 0.7]; inference s = 0.3 (locked earlier). Each
tick restarts from the SAME per-(drive, seed) noise tensor z0 (shared noise across ticks), so consecutive frames are already
correlated; the history conv is the only recurrent path. Restart cold after hard breaks.

## Arms
A: no history (control). C: B (history trained on previous KEY latents) forked at 1k, then trained with 6 of 16 rows/batch
using the model's OWN rendered previous frame as history (own-rollout cache), 6 rows true previous key, 4 keyframe singles.
Long bet: C forked from C@4k (with optimiser), pool = whole-drive own rollouts of 48 seeded train drives (12 each rendered at
s 0 / 0.1 / 0.3 / 0.5), cache refreshed from C@8k/12k/16k ("training never waits"; stages resume on the newest COMPLETE cache).
IMPORTANT: stages 4k->8k->12k->16k all trained on own_4k (history frames drawn by C@4k); own_8k finished ~11:00, own_12k
rendering now; stage 16k->20k will use own_12k (or own_8k). A trained 4k->20k on identical rows/noise.
Note: the cached own frames are single-step: each history frame is the cached C@4k output for tick i-1 of a whole-drive
rollout; the target is the KEY at tick i (epsilon loss). No multi-step unrolled loss.

## Numbers (6 dev drives x 2 seeds, ~185 ticks each at 10 Hz; signed scorer)
| check | C mem ID / A ID | C mem W / A W | car F1 C / A | late LPIPS gap C (vs own cold) | lingering ghosts >=5 ticks C / A |
|---|---|---|---|---|---|
| 4k | 0.088 / 0.109 | 29.0 / 32.0 | 0.448 / 0.463 | small | 8.8 / 4.3 |
| 5k | 0.093 / 0.104 | 29.6 / 30.8 | 0.441 / 0.448 | +0.006 | 9.25 / 4.25 |
| 10k | 0.085 / 0.096 | 26.9 / 28.8 | 0.383 / 0.454 | +0.059 (CI 0.039..0.076) | 15.25 / 4.67 |
C memory vs C memory-off (same weights): 5k F1 0.441 vs 0.429; 10k F1 0.383 vs 0.421, late gap +0.059, ghosts>=5 15.25 vs 4.9.
C memory-off at 10k F1 0.421 vs A 0.454 (weights slightly worse too). Recall C 0.39 vs A 0.435 at 10k.
Stop rule (signed): car F1 < -0.03 or late drift > +0.03 at two checks running -> stop. 10k is strike one; 15k check
results ~12:30 (C@15k still trained on own_4k). ID = edge-car appearance change between ticks; W = warp residual.
Owner watched the 5k videos: "memory on is clearly better" (smoother).

## Question
Why does the memory rollout drift more as training goes on (late gap x10 from 5k to 10k, lingering ghosts x1.6), and what
should we ADD or FIX to close it while keeping the smoothness? Ground every proposal in published work (e.g. exposure bias /
scheduled sampling, Diffusion Forcing, Self Forcing / rolling-out-in-training, history noise augmentation as in GameNGen /
DIAMOND / Oasis, history guidance, anchoring / periodic re-keying, stale-cache effects in DAgger-style training, etc.), say how
the method maps to OUR setup, and why our current recipe might be doing a proven method incorrectly (owner's rule). Include:
1. Diagnosis: the most likely cause(s), and the cheapest test that tells them apart (with expected numbers) using what exists
   (checkpoints 5k/8k/10k/12k/15k, own_4k/own_8k caches, A checkpoints, dev rollouts, signed scorer). E.g. does C@10k drift
   less with a higher inference s? with own_8k-trained weights? is it the stale cache?
2. The fix: ONE recommended change to the running long bet (or a restart point), concrete enough to implement today:
   code changes, hyper-parameters, which checkpoint to fork, steps, cost on 1 A6000 (~110 updates/min; each stage start
   costs ~30 min data loading) + 2080 Ti rollouts (~10 min per 48-drive own-cache refresh shard set with batching).
3. What to measure at the next 5k check and the rule that says it worked.
Keep it under ~900 words. Write your answer to council/drift-s1-<your name>.md and reply with a 5-line summary.
