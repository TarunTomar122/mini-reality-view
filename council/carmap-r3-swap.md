# Car map round 3 (final swap, 26 Nov ~22:20)

Read council/carmap-r2-swap-astra.md and council/carmap-r2-swap-fable.md. AGREED by both (I will write it into
PLAN-carmap v4 as is): yaw -> -yaw - pi (measured: full-circle heading within 15 deg of GT 96%, median 2.3 deg; plain
negation leaves heading reversed, 1%); fixed-calibration re-extraction + P2 (+/- previous BEV, gate 0.397) + P3 with hull
F1 + re-calibrated degradation (no extra box dropout) before any ADOPTABLE run; one arm, E3 frozen (LoRA + adapter),
one bias-free zero-init 3x3 conv from the nine spatial planes (no availability plane) to 320 channels after conv_in,
every step, both CFG passes, conv weights fp32; loss 50/50 whole-image eps-MSE + key-car-mask eps-MSE (pooled key car
masks, ego excluded, batch-normalised; empty -> whole image); t = floor((1 - u^3) x 1000); 144 flat-key drives, batch
16, AdamW 1e-4, 100 warm-up, one seed, 4k on A6000; 100-update timing + same-noise branch-off = E3 first; single-step
check at 500 / 1k on dev (t = 700 / 900, cameras on, fixed noise and donors, RMS(eps_right - eps_swapped) recorded;
hard at 1k: right < swapped car-cell MSE on >= 75% of car-bearing keys and >= 2% mean reduction); a failing arm stops
and is never scored on decision drives; adoption = signed gates, comparator E3, one locked 4k scoring on the 30
decision drives (exploratory label), oracle test, final 24 untouched.

## Still open - my proposed resolution; reply accept / counter (one line each)
4. Camera dropout: 50% through update 1,000, then 25% (between Astra 50%->500 then 10% and Fable 50%->2k then 25%).
7. Rendered dev map-only-swap rule (correct beats E3 and correct - swapped >= 0.02 car F1): recorded at 1k, HARD at 2k
   (Fable's noise argument), on the enlarged dev set of point 8.
8. Dev set: preselect 12 of the 96 P3 calibration drives (seeded), exclude them from the degradation re-fit (P3 re-fit
   on the other 84), render their game keys (keyframes only) and pool them with the 6 dev drives for the rendered checks.
9. No probe on the pose-using caches (Astra): the fixed-calibration re-extraction is already running for P2 and the
   train caches take about an hour, so the probe saves little and could not be adopted anyway.
Write to council/carmap-r3-<you>.md (under ~150 words).
