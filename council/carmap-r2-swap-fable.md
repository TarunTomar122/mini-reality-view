# Car map round 2 swap: Fable

Read: swap note, both answers, brief, `mrv/carmap.py`, `train_carmap.py`. Nothing run.

## Positions
1. **Accept Astra's injection** (one bias-free zero-init 3x3 conv after `conv_in`, availability plane dropped). We both rank
   injection last; bias-free removes the map-independent shortcut, and one hook is harder to get wrong than a residual list.
   I withdraw "UNet fp32": under autocast it changes nothing. Keep the conv's own weights fp32.
2. **Accept Astra's loss.** True-box hulls only reach 0.459 against the keys, so the key's own car mask is the better
   target. One change: normalise the car term over the batch (sum of masked error / sum of mask cells), not per image.
3. **Counter: keep t = (1 - u^3) x 1000.** Uniform t is the X1 recipe; eps-MSE is 0.107 at t=300 vs 0.019 at t=700, so
   the gradient comes from where the map is not needed. The base is frozen, so nothing can be damaged.
4. **Counter: 50% through 2k, then 25%.** 500 updates (100 of them warm-up) is too short to rely on.
5. **Accept one arm, frozen E3.** I drop R2. Named fallback, not run now: add a bias-free 4-scale branch.
6. **Accept:** no extra box dropout; degradation re-calibrated after the yaw and pose fixes.
7. **Merge.** 500 and 1k: single-step check at t=700/900, cameras on, key car masks (hard at 1k: right < swapped on
   >= 75% of keyframes, mean gap >= 2%); rendered dev recorded only. 2k: Astra's rendered rule is the hard gate. Six
   drives give about +/-0.04 on a rendered difference, too noisy to stop on at 1k.
8. **Counter, softened.** Astra is right that those drives are the 96 calibration drives. Use 12 as dev only if keys
   cost under ~30 min; otherwise 8 keyframes per dev drive at the 2k gate. Not blocking.
9. **Hold: yes**, on current caches with yaw fixed, labelled non-adoptable, never scored on decision drives.

## Yaw direction
From the mmdet3d source (yaw_nus = -yaw - pi/2) the heading should be **-yaw - pi**; plain negation fixes the hulls but
leaves `head` reversed. This is from reading, not measured: pin it with the mod-360 check before any training.

## Recipe I would sign
- **Before:** negate yaw, pin heading on the full circle, re-run hulls / P2 / P3. In parallel, fixed-calibration
  lidar2img, re-extract, 0.397 hull gate, P3 with hull F1, re-degrade.
- **Arm:** frozen E3 (LoRA and CameraAdapter included); bias-free zero-init 3x3 conv, nine planes -> 320 channels after
  `conv_in`, every step, both CFG passes. 144 flat-key drives, batch 16, AdamW 1e-4, 100 warm-up, one seed, 4k on A6000.
- **Training:** cubic t; 50/50 whole-image MSE + batch-normalised key-car-mask MSE (ego excluded; empty -> whole-image);
  camera dropout 50% to 2k then 25%; calibrated degradation only.
- **First:** 100-update timing; same-noise branch-off = E3.
- **Checks (dev only):** as point 7. A failing arm stops and is never scored on decision drives.
- **Adoption:** signed gates unchanged, comparator E3, one locked 4k scoring on the 30 decision drives, oracle test,
  final 24 untouched. Only a run on the fixed-calibration caches can be adopted.
