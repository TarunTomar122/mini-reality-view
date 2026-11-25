# Agreed plan: simulator data (council Astra + Fable, rounds 1-4, 25 Nov 2026)

Sources: s1-astra.md, s1-fable.md (research + citations), s2-*.md (critiques), s3-*.md, s4-astra.md (final points).

1. **Verify the replay first.** ScenarioNet replays of official nuScenes train scenes in MetaDrive. Per-log calibrated
   intrinsics and poses for the 6 cameras at 448x252 (~65 deg front/side, ~89 deg back), ego origin (nuScenes rear
   axle vs simulator centre; length/width), our own car hidden in the drone view. Overlay ego footprint, moving boxes
   and map corners on lidar; require median car-centre error <= 0.3 m within 25 m and landmark projections within
   2 px. Stop and fix on systematic offsets.
2. **Data.** ~2,000 ego-hidden replay pairs (official train scenes, stratified by city/turns/traffic), all 7 views at
   one tick, drone key 512x288 -> crop top 32 rows, full coverage. Inputs: clean and phone-augmented (matching rays).
3. **Arms** (identical SD 1.5 + rank-64 LoRA + camera adapter init; DINO, VAE, base UNet frozen; style rows 0 lidar,
   1 splat, 2 sim; shared learned null, 10% condition dropout; AdamW 1e-4, 500-step warmup per phase; paired real
   batches of 16 = 11 lidar + 5 splat with existing trust weights and 0.05 hole floor):
   - **A** real only, 7k updates (scored at 3k, 5k; 7k for the equal-update comparison with B).
   - **B** 2k sim-only updates (adapter + LoRA), then 5k real (scored at 3k and 5k real).
   - **C** real + 2 sim-camera rows + 2 real-camera rows targeting the replay key per step, 5k (scored at 3k, 5k).
4. **Geometry judge.** Frozen Mask2Former Swin-L Cityscapes panoptic: car instances (conf >= 0.5) matched one-to-one
   (IoU >= 0.3) to projected nuScenes vehicle hulls >= 8 px tall within 30 m in visible regions -> pooled F1; road
   edges vs projected drivable-area boundaries, symmetric nearest-edge distance in ground metres, capped at 2 m.
   4 frames x 30 val scenes outside the final 30; style 1, DDIM 25, CFG 2, 2 paired seeds. Second judge
   SegFormer-B5 must agree on the sign of each arm's difference from A.
   **Validation before any verdict:** graded ground-plane label shifts of 0.5 / 1 / 2 m must lower F1 at each step
   (>= 0.15 drop at 1 m) and raise kerb error; shuffled cameras at fixed noise must score worse.
5. **Go rule.** >= 0.05 F1 gain or >= 10% kerb-error reduction over A, paired scene-bootstrap 95% CI excluding zero,
   no regression on the other metric. (Appearance guardrails dropped by the owner: LPIPS and 'sim look' are reported only.)
   On a pass: scale to ~8,000 pairs, 5k sim + 40k real vs A at 40k/45k.
6. **Budget** ~8 GPU-hours (training ~6, prep + scoring ~2). CARLA only after a positive pilot and a Vulkan check.

## Decision rules agreed with the owner (25 Nov, so work continues without waiting)
- Alignment check fails -> up to 3 genuinely different fixes; still failing -> drop sim, continue real-only.
- Judge fails its validation (shift/shuffle) -> backup judge SegFormer-B5; also fails -> sim result inconclusive ->
  skip sim unless the pictures are obviously better.
- Verdict:
  - C passes the go rule (B does not) -> at sd_run1's next checkpoint pause it and add sim rows (style 2, ~20% of each
    batch) to its mix; continue (no restart).
  - B passes (C does not) -> render ~8,000 pairs; new main run: 5k sim then real with sd_run1's settings (phone tokens,
    trust keys); replaces sd_run1 only if it beats it at equal real steps (~1 extra day).
  - Both pass -> take the larger gain; if about equal, take C (cheaper, no restart).
  - Neither passes -> drop sim; sd_run1 is the final model; README "What did not work".
  - Owner (25 Nov): a game-like look is acceptable if placement is accurate. Appearance (LPIPS, 'sim look') is reported
    but is NOT a go/no-go guardrail; the decision rests on car-placement F1 and kerb error only.
- Then in all cases: finish + evaluate the final model (30 real val scenes, pictures, vs older models), optional Difix
  if it visibly helps, distil to 1-4 steps, finalise try_my_photos.py + README how-to (tested on phone-like nuScenes
  photos), final README + results page. New design questions -> council again.
