# Car map round 2, swap (26 Nov ~22:05)

Both answers are in: council/carmap-r2-astra.md and council/carmap-r2-fable.md. Read the other reviewer's answer.

## New measurement (Fable's yaw point, checked)
tmp/yawcheck.py on 250 decision frames, 1,272 predicted cars (score >= 0.3, <= 40 m) matched to nuScenes GT cars
(centre <= 1.5 m), corners from mrv/carmap.box_corners_ego for both (GT via gt_frame):
| yaw conversion | footprint IoU | heading err mod 180, median | share < 15 deg |
|---|---|---|---|
| as coded (yaw) | 0.620 | 3.8 deg | 0.78 |
| -yaw | 0.667 | 2.2 deg | 0.97 |
| -yaw - pi | 0.667 | 2.2 deg | 0.97 |
| yaw + pi/2 | 0.290 | 86 deg | 0.03 |
So the detector yaw is mirrored in carmap.py / bev_box_judge.py (Fable is right). Fix = negate the predicted yaw
(the 180-degree direction still to be pinned by a full-circle check against GT before the heading channel is used).
The 0.417 / 0.444 hull numbers, P2 and P3 were computed with mirrored hulls and must be re-run.

## Where you agree (as I read it)
- Main causes: the loss gives little signal where the map matters (car cells; Fable: timestep sampling too) and the
  cameras make the map redundant -> heavy camera dropout early in training.
- All early checks on dev drives; the 30 decision drives only for one locked scoring per surviving arm; final 24 kept.
- Camera-only fix before any adoptable training: lidar2img from fixed calibration only, re-extract, repeat the 0.397
  hull gate (P2, with and without previous BEV) and P3 with hull F1, re-degrade.
- Single-step right vs swapped measured directly (car-cell MSE and RMS(eps_right - eps_swapped)).

## Where you differ - please converge
1. Injection: Astra = one bias-free zero-init 3x3 conv from the map planes onto the 320 channels after conv_in (all
   steps, both CFG passes); Fable = keep the 4-scale T2I branch, wider (128, 4-conv stem), UNet weights fp32.
2. Loss: Astra = 50/50 whole-image MSE + MSE on pooled target-KEY car masks (ego excluded); Fable = x8 weight on cells
   under TRUE-box hulls (dilated 1 cell), normalised.
3. Timesteps: Fable = t = (1 - u^3) x 1000 (bias to high noise, T2I-Adapter); Astra = uniform.
4. Camera dropout: Astra = 50% until step 500 then 10%; Fable = 50% throughout.
5. Arms: Astra = one arm, frozen E3; Fable = R1 frozen E3 + R2 with LoRA unfrozen (swap must remove >= half the gain).
6. Box dropout / degradation: Astra drops the extra 15% box dropout; Fable keeps degradation, questions stacking.
7. Early check: Astra = rendered dev check at 1k and 2k (correct beats E3 and correct - swapped >= 0.02); Fable =
   single-step check at 500 / 1k (right < swapped on >= 75% of keyframes, mean gap >= 2%), then rendered dev swap >= 0.02.
8. Second dev set: Fable suggests ~12 unused official-val drives from the P3 pool (needs game keys rendered).
9. Whether the 1k mechanism probe may start on the current (pose-using) caches, labelled non-adoptable (Fable yes).

Reply with your final position on each numbered point (accept / counter, one line each) and the recipe you would sign.
Write to council/carmap-r2-swap-<you>.md (under ~400 words).
