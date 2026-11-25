# Sim data, round 2: critique of Astra and the plan I would sign (seat: fable)

## 1. Critique of Astra's plan

**Right, and adopted below.**
- The control. A fresh real-only run with the same minibatches, timesteps and noise as the sim arm is better than my
  forks from `sd_run1`, whose data changes at 10k. Paired scene-bootstrap intervals beat my "twice the seed spread".
- Shuffled-camera negative control, "equal real steps" and "equal total updates" as two separate comparisons, and
  pilot scenes kept apart from the final 30.
- Rig QA: full camera-to-ego poses (our test script is yaw-only at 70 degrees), landmarks within 2 px, at least 90%
  of logged cars present. A six-camera visibility mask saved next to each sim key.
- Three style rows and the shared null token. My fourth row and "keep the style token when dropping" were unnecessary:
  a shared null only pushes guidance slightly away from the sim look, and the control stays identical to today.
- Car scoring must count extra cars as well as missed ones. My "hit rate" could be gamed by smearing car pixels.
- `train_sd.py` does evaluate style 1 on splat scenes only (line 208), so most val scenes are skipped today.

**Wrong or unsupported.**
- *The pilot cannot show what it gates on.* It scores car placement after 1k sim + 3k real steps from scratch. Our
  300-step smoke test ignored the cameras and at 5k `sd_run1` still has "positions not precise". Both arms may sit at
  the floor at 3k. With 10 scenes x 3 frames (maybe 100 cars), a 0.05 F1 gate is inside the noise.
- *It can only detect a head start.* Round-1 evidence says pretraining buys speed once real data is plentiful (Zoph et
  al. 2020; Tremblay et al. 2018) and that mixing beats separate phases (Wei et al. 2025). Astra's only mixing arm is
  optional, runs after the full pretraining, and also changes which parts train.
- *Adapter-only sim phase.* With LoRA at zero the stock UNet must place cars from tokens alone. That is my reasoning,
  not a citation, but no evidence supports the opposite either. It needs a test before the arm depends on it.
- *The input gap is never measured.* The adapter reads DINOv2-Small tokens of flat-shaded renders. Depth Anything V2
  (2024) found small DINOv2 encoders transfer worst from synthetic images. Without a token comparison a null result
  cannot be explained.
- *Hand marking as the only metric.* It cannot be rerun per checkpoint, and one person marking both arms is hard to keep
  blind. A fixed 12 px threshold is about 0.4 m at 15 m from the drone and 1.1 m at 40 m. Two image rows is thin for kerbs.
- *Weather and time-of-day matching* does not apply to ScenarioNet's single plain look; it belongs to the CARLA stage.

**Missing.** Twin keys (real cameras paired with a rendered complete key of the same moment), the one sim use with no
input gap. The rear-axle ego origin check. The published near-misses for this exact task (GCD 2024; 6Img-to-3D 2024).

## 2. Converged plan

**Data (one render serves every arm).** ScenarioNet replays of 200 train scenes from both cities, all keyframes (about
8k), plus 8 train scenes held out for sim QA. Per-scene real intrinsics and poses (about 65 degrees, back camera
about 89). Ego hidden in all seven shots. One plain look. Per frame: six sim cameras twice (clean; rig jitter 0.3 m /
5 degrees / FOV 15% with exact rays plus the phone augmentation), the drone key, the semantic drone image, the visibility mask.
- *Sim rows:* sim cameras -> sim key, full weight on every pixel.
- *Twin rows:* real cameras -> sim key, weight 1 on road, markings, walkway and objects, 0.05 on terrain and sky.

**Model changes.** `nn.Embedding(3)`: 0 lidar, 1 splat, 2 sim. Trainer gains a sim key set, per-source row quotas, a
choice of trainable parts, and batches, timesteps and noise seeded by step so arms are paired. Nothing else changes:
shared null tokens, 10% drop, lr 1e-4, 500-step warm-up, 0.05 hole floor, render style 1, DDIM 25, CFG 2.

**Arms, all from the same initial weights, sharing the same real batches (11 lidar + 5 splat rows).**

| Run | Recipe | Steps | GPU-h | What it tells us |
|---|---|---|---:|---|
| E0 | Twin overlay on lidar keys (0.3 m), landmarks (2 px), cars present, token cosine, sim cameras into `sd_run1` | none | 0.5 | Is the sim aligned; how big is the input gap |
| E1 | Metric validation (below), `sd_run1` at 5k and 10k | none | 1.0 | Is the metric valid; how many real steps the pilot needs |
| R | Render and encode | none | 1.0 | Data |
| S | Sim rows only, three variants: adapter / adapter + cross-attention LoRA / all. Winner continues to 2k | 3 x 1k + 1k | 1.2 | How little must train to learn sim layout |
| A | Real only | 4k | 1.2 | Control |
| B | S winner, fresh AdamW, then A's batches, all parts train (owner's idea) | 4k | 1.2 | Pretrain, then fine-tune |
| C | A's batches + 4 sim-style rows per step (2 sim, 2 twin), all parts train | 4k | 1.5 | Keep sim in the mix |
| Evals | 30 val scenes outside the final 30, 4 frames each, 2 seeds, matched and shuffled cameras | none | 0.8 | Scores below |

Pilot total about 8.4 GPU-h. S winner = fewest trained parts whose matched-minus-shuffled LPIPS gap on the 8 QA scenes
is at least 80% of the "all" variant's gap. Real steps are 4k unless E1 says otherwise (disagreement 3).

**Geometry metric.** Labels come from nuScenes, never from a key: vehicle 3D boxes within 40 m projected into the drone
camera (hull, at least 8 px tall), and the map's drivable area drawn on the ground plane. Scored only where the lidar
key has coverage or a box lies. Before any render is looked at, label overlays are checked on the real keys and
misaligned frames dropped (Astra's rule). Predictions: SegFormer-B5 road and vehicle masks on the style-1 render.
- *Car F1.* Recall = boxes at least half covered by vehicle pixels. Precision = vehicle pixels inside a box hull
  grown by 4 px. Misses and extras both count; merged parked cars do not break it.
- *Kerb error.* Symmetric mean distance between label and predicted road edges, in ground metres, capped at 2 m.
- *Road IoU*, and masked LPIPS as the look guardrail (style 0 vs lidar keys; style 1 vs splat keys where they exist).

*Validation, all required before any arm is scored.* (a) Projected labels match the semantic drone image of val
replays at IoU 0.9 or more, which tests the projection without the segmenter. (b) Splat keys and `sd_run1`@10k score
at least twice the shuffled-camera and mean-image values on car F1. (c) Two seeds give the noise floor. (d) Astra's
blind hand marking on 30 frames ranks arms the same way and correlates per frame at 0.7 or more. If (b) or (d) fails,
hand marking with a threshold of half the box width becomes the metric. Also report scenes far from any train pose
(Lilja et al. 2024), because twin keys and labels share a source.

**Go rule, fixed in advance.** An arm goes to a full run if, against A at equal real steps: car F1 rises by 0.05 or
kerb error falls by 10%, with the paired 95% interval excluding zero; the other of the two is not worse; LPIPS is
worse by at most 0.01; shuffling cameras drops its score to the floor; and a blind picture sheet shows no sim look. A
passing arm is then compared with A extended to equal total updates (+0.6 GPU-h). If C passes, one extra run splits sim
rows from twin rows (1.5). If nothing passes, stop and write it up.

**Full run (about 26 GPU-h).** Winner on all 700 train scenes against a fresh real-only control, both on the
post-10k `sd_run1` recipe, 40k real steps each, scored at 5/10/20/40k on the final 30. A gain present at 5k and gone at
40k is reported as "faster, not better". CARLA follows only a positive result or a diagnosed content gap.

**Order.** Fix rig and ego origin -> E0 -> E1 -> R -> S -> A, B, C in parallel -> evals -> decision.

## 3. Remaining disagreements

1. **Pretrain or mix.** I expect C to beat B; Astra's plan is B. Settled by B vs C on identical real batches.
2. **What trains in the sim phase.** Astra: adapter only. I doubt it can place cars. Settled by run S.
3. **Pilot size.** If E1 shows `sd_run1`-style training clears the metric floor by 3x the seed spread at 3k real
   steps, I accept Astra's 1k + 3k pilot and the budget falls to about 7 GPU-h. If not, 4k is the minimum.
4. **Metric.** Automatic first (mine) or hand-marked first (Astra). Settled by validation step (d).
5. **Twin keys.** Unpublished and mine alone. Settled by the E0 overlay and, if C passes, the split run.
6. **After a failure.** Astra goes to CARLA if replays fail QA. I would go only if S learns sim layout while the
   real and sim tokens differ. Settled by the E0 token cosine and the S scores.

*Caveats.* No new citations; all are from round 1, where some numbers came from summaries. GPU-hours assume 1.8 min
per 100 steps at batch 16 and linear scaling to 20 rows, which is untested on the 18 GB slice.
