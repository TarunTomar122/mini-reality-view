# Experiment T: real cameras in, game-style third-person view out (merged plan)

Inputs: council/tpv-brief.md, council/tpv-s1-fable.md (Fable). Astra unavailable this session; second view = mine.
Adopted as Fable proposed, with the changes below.

## Arms (10k steps each, bs 16, style 2, seed 0, no phone copies, hole weight 0; slurm/tpv_experiments.sh)
- E1: init = real-key renderer sd_run1 40k (adapter + LoRA; style rows 2 -> 3, new row copied from row 1); all drives.
- E2: plain SD 1.5 + fresh adapter; all drives.
- E3: as E1 on a fixed random quarter of the training drives (150 of 600 candidates, tmp/tpv_quarter.txt).
- C (trained adapter + fresh LoRA) dropped, following Fable: adapter and LoRA were trained as a pair.

## Data
- Keys: MetaDrive replays, third-person camera 7.5 m behind the car centre, 2.4 m up, pitch -8 deg, 60 deg HFOV, own car
  shown; every 2nd keyframe; ~500 train drives (data/scn_8k) + held-out val drives (data/scn_val, official val split).
- Inputs: real DINOv2 tokens + calibrated rays (data/enc_full).

## Scoring (slurm/tpv_judge.sh, scripts/judge_pair.py) at 5k and 10k
- 30 held-out val drives x 4 keyed frames x 2 seeds, DDIM 25, CFG 2, style 2.
- Change vs Fable: instead of rendering instance masks from MetaDrive (unverified headless), the same segmenter is run
  on the render and on the exact key; cars are matched one-to-one (IoU >= 0.3, own car masked by its projected hull) ->
  pooled car F1; road IoU; L1 / LPIPS vs key. The keys are exact, so this measures agreement with the true layout as far
  as the segmenter sees it.
- Ceiling: the keys through the VAE round trip, scored the same way (on E1's 10k frame list). If the segmenter finds
  few cars in game keys, switch to MetaDrive's semantic/instance camera or a small segmenter trained on keys.
- Paired scene bootstrap (judge_pair.py --compare).

## Decision rules (from Fable)
- Init: E1 beats E2 by >= 0.03 car F1 (CI excludes 0) -> use E1's init; else plain SD (if E2 draws almost no cars at
  10k, E1 wins outright).
- Data: E1 - E3 >= 0.05 F1 (CI excludes 0) -> data-limited: render the remaining train drives + every keyframe before
  the long run; < 0.02 -> enough; in between -> add the remaining drives only.
- Go: long run (40k, winning init, phone copies added) if the best arm reaches >= half the ceiling car F1 at 10k and
  road IoU is not falling 5k -> 10k; below a quarter of the ceiling -> back to the council.
