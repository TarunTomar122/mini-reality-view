# Temporal council · Astra · round 2

**I would run three matched continuations of the TPV winner.** Fable persuaded me to vary history noise, tell the model its level, retain four single-frame examples per batch, and measure changes in edge-car appearance directly. GameNGen sustained long rollouts using conditioning augmentation. Our earlier DiT gained from training on its own outputs (+1.8 RGB L1 drift over 20 s). Both recipes deserve a test on SD 1.5. [GameNGen](https://arxiv.org/abs/2408.14837) · [Local evidence](temporal-brief.md)

**Where I differ.** Fable’s trained arms all receive history. The TPV winner saw sparse keys, so extra updates on dense targets could explain a gain. I would spend the third arm on an independent continuation and defer attention sharing, camera-token memory, BEVFormer and video modules. I would also reserve untouched drives for final scoring. The three named preview drives have already informed visual judgments. Finally, nuScenes visibility level 1 permits partial visibility. A memory claim needs inspection of all six photos. [TPV plan](PLAN-tpv.md) · [Preview observations](../PROGRESS.md)

## Paired data and budget

Render whole drives at 10 Hz, about 20 s each. Use E3’s 150 training candidates, ten existing judge drives for development, and 24 official validation drives drawn outside the 30 judge drives for final scoring. Freeze eligible lists and seed-0 draws before fitting. Render keys only, in a fresh directory with the existing TPV geometry and 512×256 crop. Pilot two drives on a 2080 Ti. Check tick 5k against existing keyed images and verify replay interpolation against converter timestamps before using intermediate ticks.

Within each five-tick interval interpolate the two actual sample timestamps. Take the nearest same-scene exposure per camera, reject any offset above 45 ms, and log signed offsets and latest exposure latency. Break sequences at gaps or repeated rig bundles. Cache scaled VAE latents and six calibrated DINO token grids by scene/tick. Use the same accepted ticks across arms. Report rejects, usable durations and 2 Hz anchor scores separately.

## Arms

Frozen reference: independent fixed noise and warm strengths 0.5/0.7. Roll each sequence cold from its first frame. Use style 2, CFG 2, DDIM25 schedule and paired seeds 0/1 throughout evaluation.

All trained arms start from the selected TPV checkpoint. Use 4,000 updates, effective batch 16 (12 dense targets, four singles from all 9,587 training keys), identical target draws, rank 64, bf16, full-key epsilon MSE, AdamW at 5e−5, zero weight decay, 500-step linear warmup and 10% camera dropout.

| Arm | History during training |
|---|---|
| A · Independent | Current cameras only throughout. |
| B · Previous key | Previous tick’s scaled latent, corrupted as √(1−s²)z + sε, s uniform [0,0.7]. Add a parallel zero-initialized convolution of four history channels and one constant s plane to frozen conv_in. Singles use zeros and s=1. Save/reload the branch explicitly. Both CFG passes receive identical history. |
| C · Previous prediction | Fork B at update 1,000 including optimizer state. At 1,000 and 2,500, freeze a snapshot and roll 128 eight-tick training clips from cold using DDIM25, CFG 2 and s=0.3. Cache detached outputs. For six dense rows per batch use cached predecessors of the exact target; preselect the same targets for A/B. Apply B’s training corruption. |

At inference B/C feed back their last output, using fixed history noise per drive independent of sampling noise. Pick s from {0, 0.1, 0.3} on development. Time 100 updates and a rollout shard first. At 70 updates/min, gradient work costs ~57 min per arm. Charge startup, cache generation and the shared prefix against each applicable 180-minute slice cap. Reduce cache clips equally across target manifests before fitting if needed. Time and report 2080 Ti rendering, token extraction and A6000 evaluation separately.

## Measurement and decision

- **Primary flicker** is the edge-car appearance change. Link adjacent key masks by Hungarian IoU ≥0.5. For tracks centred in either outer third, crop both frames to their respective key boxes plus 10%, resize to 96×96 and compute DINOv2-small CLS cosine distance in predictions and keys. Average the absolute difference between those distances, including missed predictions, then average drives equally. Report track counts, matched-only scores and the fraction of persistent key tracks whose predicted match status flips.
- **Placement** uses the existing ego exclusion, ≥8 px car height, Hungarian IoU ≥0.3, pooled car F1 and mean road IoU on every fifth tick, both cold and in rollout. **Drift** is (last two seconds minus first two seconds) LPIPS to key in rollout, minus that change in identical cold renders. Require contiguous ≥19 s and report absolute late LPIPS. At 5 s switch to another drive’s cameras and count ticks to regain 80% of its cold car F1 over a five-tick window with nonzero reference F1.
- **Memory** is paired rollout versus cold recall on the last hidden tick of cars previously visible in a rig photo, hidden in all six for ≥0.5 s, and still identifiable in the chase key. Mark events before viewing predictions. Report counts and drives. Require ≥20 events across ≥5 drives and a positive lower paired recall-difference bound to claim a memory gain.
- **Adoption.** On development select the lowest-flicker setting passing the gates. Lock settings before final scoring. Bootstrap 2,000 paired drive samples keeping seeds together. Require ≥15% less primary flicker with a 95% interval excluding zero improvement, nonincreasing flips, nonnegative placement point differences with lower bounds ≥−0.02 F1 and ≥−0.01 road IoU, upper drift bound ≤+0.02 LPIPS, and median swap recovery ≤5 ticks. Compare warm to frozen independent, then B/C to both A and the best qualifying frozen reference. Prefer B when both pass unless C further lowers drift with a paired interval excluding zero.
- **Stop.** Audit 20 varied key frames and two turning clips for missed edge cars or broken tracks. Check key and VAE round trips and inspect exits blind. Reject a candidate if an exited car visibly persists ≥0.5 s in two or more drives. Pause ranking for unreliable masks/tracks or fewer than 12 qualifying final drives. If neither history arm passes, retain the qualifying frozen reference (independent by default). Consider pooled past-camera tokens only after a credible hidden-car deficit survives this test.
