# Temporal consistency and memory: merged plan (v3: Astra's and Fable's sign-off objections applied)

Sources: council/temporal-brief.md; round 1 temporal-s1-astra.md, temporal-s1-fable.md; round 2 temporal-s2-astra.md,
temporal-s2-fable.md. Where round 2 still differed, the choice below and the reason are given in [brackets].

## Order
1. Frozen methods (no training): independent fixed noise; warm start (SDEdit from the previous output) at 0.5 / 0.7;
   new noise every frame (metric check only).
2. Three trained arms from the E1/E2/E3 winner (A control, B previous key, C own rollouts).
3. Input memory (past camera tokens / BEVFormer) and video modules only in a later round, if a hidden-car deficit
   survives this test.

## Data (2080 Ti, keys only, existing TPV camera and 32-row crop, fresh folder data/sim/keys_tpv10)
- Train: 144 drives from E3's list (tmp/tpv_quarter.txt). Development: 6 more drives from that list, chosen by seed
  across traffic density and turning. Lists and seed-0 draws frozen before fitting.
- Decision set: the 30 judge drives (val). Final confirmation: 24 further official val drives never viewed or used,
  scored once after settings are locked. [Astra: the 3 preview drives already informed visual judgments.]
- Whole drives (~20 s) at every 10 Hz sim tick. Pilot 2 drives first: check tick 5k against existing keyframe keys and
  ScenarioNet's interpolation against the converter timestamps before using intermediate ticks.
- Pairing: tick time interpolated linearly between the two real sample timestamps; per camera the nearest same-scene
  sweep; reject offsets > 45 ms [Astra's tighter bound]; break sequences at rejects / repeated rig bundles; manifest
  (scene, tick, time, 6 files, offsets); report rejects and usable durations.
- Cache scaled VAE latents of keys and the 6 calibrated DINO token grids per (scene, tick).

## Trained arms (all from the E-winner; equal updates; identical target draws across arms)
- 6,000 updates, checkpoints 2k / 4k / 6k, 4k also scored [Fable: a zero-initialised branch at 5e-5 may need it;
  Astra's 4k is nested]. Batch 16 = 12 dense rows + 4 keyframe singles from all train drives (both agree). LoRA r=64,
  epsilon loss, LR 5e-5 for LoRA + adapter (500 warm-up), 1e-4 for the new conv, 10% camera dropout, style 2, bf16.
  The 6 development drives are excluded from the keyframe singles too.
  Cap 180 slice-minutes per arm including caches; time 100 updates and one cache shard first; cut all arms equally.
- **A, control:** dense targets, current cameras only.
- **B, previous key:** a parallel zero-initialised conv beside the frozen conv_in takes 5 channels: the previous tick's
  key latent corrupted as sqrt(1 - s^2) z + s eps, plus a constant plane holding s; s ~ U[0, 0.7] [Astra's range,
  covers Fable's U[0.1, 0.5]; GameNGen used a range with the level given]. No history (singles and 10% of dense rows)
  = zeros with plane 1. Both guidance passes get the same history. Saved / reloaded explicitly.
- **C, own rollouts:** fork B at 1,000 (with optimiser state). At 1,000 / 2,500 / 4,000 freeze a snapshot and cache
  detached DDIM-25 / CFG-2 rollouts of 128 eight-tick train clips (s = 0.3); for half of the history rows use the
  cached own frame with its true next key, corrupted as in B. A and B see the same preselected targets.
- Inference for B/C: feed back the last output; first tick without history; run-time s swept over {0, 0.1, 0.3, 0.5}
  on development drives, then locked.
- **Noise control (all methods):** one sampling-noise tensor per (drive, seed), reused across ticks, methods and cold
  re-renders; history corruption drawn from a separate seeded stream. Fresh noise every frame only in the metric check.

## Metrics (own car excluded; drives weighted equally; cold = same model, same frames, no history)
- **ID, edge-car appearance change (co-primary):** link key car masks across adjacent ticks (Hungarian IoU >= 0.5);
  for tracks centred in the outer thirds, crop both frames to the key box + 10%, 96x96, DINOv2-small CLS. Per track
  transition: |cosine distance between the two ticks in the render - the same in the keys| (absolute, so a frozen
  car is penalised as much as a flickering one), then averaged, over transitions where the render matches the car
  (IoU >= 0.3) at both ticks [Fable: a crop of a missed car compares road with road and looks steady]. Misses are
  not ignored: they enter through edge-car recall (gate below) and flips [Astra's concern]. Percentage reductions
  only where the reference score is positive.
- **Edge-car recall:** share of eligible key edge-car track ticks matched by the render (IoU >= 0.3).
- **W, car warp residual (co-primary):** pixel residual on car pixels after warping with the keys' motion (round-1
  definition), outer 20% each side, with the valid car-pixel share reported. Only used if the flow check passes.
  [Fable: pixel-only scores reward erased or blurred cars; Astra: identity alone misses jitter. Both kept.]
- **Flips / ghosts (secondary):** share of track ticks whose matched status changes; unmatched render cars per frame
  and their lifetime.
- **Placement:** judge_pair.py (car F1, road IoU) on every fifth tick, in rollout and cold.
- **Drift:** late gap = LPIPS of the last 2 s minus the same frames cold, and the same for car F1; early gap (2 s from
  tick 5) alongside. Clips >= 19 s only. No numerical comparison with the old DiT's +1.8 (a different quantity:
  last-five minus first-five keyframe rollout L1 at 2 Hz) unless that cadence, window and scale are reproduced and the
  dataset difference is labelled.
- **Lag:** at 5 s swap to another drive's cameras; ticks until car F1 vs the new keys reaches 80% of its cold value
  (five-tick window, nonzero reference).
- **Memory:** pre-marked events (car visible in a rig photo, hidden in all six >= 0.5 s, still identifiable in the
  chase key; all six photos inspected); rollout vs cold recall on the last hidden tick. Claim a gain only with >= 20
  events over >= 5 drives and a positive lower bound; otherwise descriptive.

## Decision rules (paired bootstrap over drives, 2,000 draws, seeds 0/1 kept together)
- **Gates for every method:** car F1 and road IoU point differences >= 0 AND lower bounds >= -0.02 (car F1) and
  >= -0.01 (road IoU) vs frozen independent; for B/C the same checks also against arm A and against the frozen
  comparator used for adoption; edge-car recall drop <= 0.02 (point difference) vs the same comparators; late LPIPS gap
  upper bound <= +0.02 and late car F1 gap lower bound >= -0.03; median lag <= 5 ticks; ghosts not growing.
- **Warm start adopted** if ID and W both fall >= 15% vs frozen independent (intervals excluding zero) and gates pass.
- **B or C adopted** if the same holds against arm A, arm A + warm start (evaluation only, same locked strength), and
  the best qualifying frozen method [Fable: history must beat extra training and warm start together].
- **B preferred**; C only for >= 0.01 less late LPIPS gap or >= 10% less ID or W (interval excluding zero), placement
  kept. A kept only for a confirmed placement gain with flicker no worse.
- Development drives pick settings; the decision set ranks; the 24 untouched drives confirm once. **Confirmed** =
  both flicker scores (ID, W) improve vs the same comparators with intervals excluding zero and every gate passes on
  those 24 drives; otherwise nothing is adopted and the result is reported as unconfirmed.
- Trained arms are compared at equal update counts (2k / 4k / 6k; C's updates include B's first 1,000). Each arm is a
  single training run, and the write-up says so. The owner's
  side-by-side clips can veto a pass, not grant one.

## Stop rules
- Metric check fails (key video and VAE round trip must score best, new-noise-every-frame worst; hand check of ~20
  key masks and two turning clips; flow keeps too few edge-car pixels): fix the measurement before ranking.
- Dense rendering costs over twice the estimate: cut to 100 training drives, keep clip length.
- Lag over 10 ticks: one fix (s ~ U[0.3, 0.7], history dropout 40%), then stop that arm. An exited car that visibly
  persists >= 0.5 s in two or more drives rejects the method.
- Fewer than 12 qualifying final drives, or unreliable masks / tracks: pause ranking.
- Neither B nor C passes: keep the warm start if it qualified, else independent; the next round goes to input memory.
- Wide intervals leave the choice open, and we say so.
