# Car map round 2: the signed X1 / X2 design does not use the map. Redesign brief (26 Nov ~21:45)

Read first: council/PLAN-carmap.md (signed v3), council/carmap-brief.md (round 1 context), PROGRESS.md (entries from
18:40 on), council/drift-20261126-2055*.md (hourly review incl. the bugs found). Code: mrv/carmap.py (rasterizer,
CarMapBranch), scripts/train_carmap.py, scripts/sd_render.py, scripts/carmap_diag.py, scripts/bev_degrade.py.

## What was run (as signed, after the reviewers' fixes)
- E3 10k frozen (SD 1.5 + LoRA r64 + CameraAdapter, DINOv2 tokens of the 6 cameras + rays, style 2, latents 4x32x64).
  Branch = 10-channel chase-view box picture (cars / heavy / people / obstacles coverage x score, inverse depth, edges,
  heading sin/cos, length, availability) area-pooled 256x512 -> 32x64, stem 2 conv (64 ch), 3 stride-2 downs,
  zero-init 1x1 projections to 320/640/1280/1280, passed as down_intrablock_additional_residuals (T2I-Adapter style).
  X2 adds the BEVFormer BEV grid sampled at z = 0 / 1 m per latent cell (512 -> 32 by 1x1) + valid planes.
- Training: keyframe keys of the 144 train drives (2,259 rows), boxes = camera-only BEVFormer, degraded to val error
  rates (P3 leak), 15% box dropout, 10% camera dropout, AdamW 1e-4 (500 warm-up), batch 16, epsilon loss, 4k updates
  (~1 s/update on an H200 1g slice). Loss X1 0.12 -> ~0.05-0.07 (noisy), grad norm 0.003-0.01 throughout.
- Render bug found by Fable and fixed (diffusers pops the residual list; the map acted on DDIM step 1 only). Fixed
  renders differ from E3 on the same noise by 3.9-32 /255 per pixel (bug: 0.3-2.7).

## Results at 2k (fixed renders; 30 decision drives x 4 keyframes x seeds 0/1; Mask2Former; paired drive bootstrap)
| | car F1 | vs E3 0.318 | map-only swap F1 | correct - swapped | road IoU vs E3 0.848 |
|---|---|---|---|---|---|
| X1 boxes | 0.315 | -0.002 [-0.030, +0.025] | 0.323 | -0.007 [-0.024, +0.009] | 0.856 (+0.008 [+0.003, +0.013]) |
| X2 boxes + grid | 0.322 | +0.004 [-0.024, +0.035] | 0.334 | -0.013 [-0.034, +0.009] | 0.855 (+0.007 [+0.002, +0.013]) |
SegFormer: X1 0.307 vs 0.302, X2 0.302 vs 0.302. Stop rule (map-only swap moves car F1 < 0.02 at 2k) fires for both,
validly this time. 4k checkpoints exist, unscored. Picture: pulled/cm2k_sheet_fixed.jpg (local; columns key | detector
map overlay | E3 | X1 right map | X1 wrong map): the branch changes road edges / car shapes, but right and wrong maps
give near-identical changes; the detector boxes do sit on the key's cars.

## Single-step diagnostic (decision keyframes, for information only; reviewers asked for dev drives next time)
scripts/carmap_diag.py, 120 keyframes, fixed noise, eps-MSE right map / swapped map / branch off:
X1 4k t=300 0.10658 / 0.10669 / 0.10627; t=700 0.01900 / 0.01904 / 0.01891; on map-covered cells right vs off
0.21911 vs 0.22037. X2 2k t=700 0.01880 / 0.01882 / 0.01891. Branch output changes eps by 1-4% RMS, nearly the same for
right and swapped maps; residual RMS per scale ~[0.07, 0.12, 0.29, 1.07] at X1 4k. -> the branch learned a
map-independent shift (it improves road IoU slightly) and essentially no map-dependent signal.

## Context
- Placement is the bottleneck: renderer car F1 0.318 vs VAE ceiling 0.763; projected BEVFormer hulls 0.417 (camera-only
  0.444), GT hulls 0.459 against the game keys. So the boxes carry placement information E3 lacks.
- E3 already draws plausible cars from the camera tokens; the epsilon loss at 32x64 may give little gradient toward
  using an extra, partly redundant input (cars are small: many are a few latent cells).
- Open review items from the hourly check: (Astra) camera-only BEVFormer: CAN bus zeroed but projection matrices still
  use per-camera timestamped ego poses; P2 / hull gate may need repeating with fixed calibration. P3 hull F1 omitted.
  Rasterizer orders hulls by mean depth (not per-face depth). The decision drives have now been looked at twice for
  car-map scoring.
- Budget: H200 1g.18gb slices currently unavailable (node draining); A6000 (48 GB) and 2080 Ti available; <= 12 jobs.
  Temporal track runs in parallel (arms A/B/C, ~1 h per 6k on A6000).

## Questions
1. Why does the branch ignore the map? Rank the likely causes (injection point/strength, zero-init at 1e-4 for 4k,
   redundancy with camera tokens, loss signal on small cars, map encoding, degraded boxes, something else).
2. Propose ONE redesigned experiment (max two arms) with a cheap early check that would show map use within ~1k updates
   (e.g. map-only swap on DEV drives, or a single-step diagnostic), the training recipe, and how it fits the signed
   gates (which stay: car F1 >= 0.382, gain >= +0.05 with lower bound > 0, road not down, full swap <= 0.05, map-only
   swap lower bound > 0, train-vs-held-out <= 0.05, oracle test). Options to weigh (not exhaustive): input-level
   conditioning (concat to the latent via a zero-init conv beside conv_in, like the temporal history conv) or a
   ControlNet copy instead of T2I residuals; heavier camera dropout or a map-only curriculum so the map must be used;
   car-pixel loss weighting; oracle boxes first then detector boxes; a richer map (rendered game-style car silhouettes);
   unfreezing LoRA jointly.
3. Should the decision drives still be used for the redesign's decisions, or should early checks move to dev / a
   fresh val subset?
4. Anything about the camera-only issue that must be fixed before any new training.
Write your answer to council/carmap-r2-<you>.md (under ~600 words).
