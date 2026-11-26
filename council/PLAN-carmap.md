# Car map: merged plan (draft for sign-off)

Sources: council/carmap-brief.md; rounds 1-2 carmap-s1/s2-astra.md, carmap-s1/s2-fable.md. Coordinator's choices on
points still open after round 2 are marked [choice: ...]. Supersedes council/PLAN-bev.md for the third-person target.

## Form (agreed)
Detector boxes painted into the chase view (same projection as the judge, ego occluder, far-to-near, depth-tested
faces), fed through a zero-initialised pixel-aligned side branch with residuals at four UNet encoder resolutions
(both round-2 plans); same map in both CFG passes; DINO tokens and rays stay. Map picture 256x512 -> 32x64: coverage x
score for cars, heavy vehicles, people/bikes, obstacles; inverse depth; instance boundary; heading sin/cos; box length
(MetaDrive picks car models from length; verify against the installed assets before crediting appearance to it);
availability gate ("unavailable" distinct from "empty"). Causal feed only; scores decay with age, expire at 0.5 s.
15% random box dropout in training (separate random stream).

## Preflights (2080 Ti / A6000, no training; any reject -> no training)
- P1 cache: bev_extract.py boxes (all classes, score >= 0.2) + grids for the 144 train + 6 dev drives, the 30 decision
  drives, other official-val drives for calibration, and the final 24 (stored, unviewed). Verify token coverage
  (grid-file skip can omit boxes); record checkpoint, motion source, calibration, timestamps, sweep IDs.
- P2 telemetry: BEVFormer aligns its previous grid with can_bus ego motion (from the ego pose). Rerun the decision-drive
  hull test with can_bus zeroed (with and without the previous grid). If hull F1 >= 0.397 (0.417 - 0.02) that mode is
  used for every cache. Otherwise the standard mode is used, results are labelled "cameras + ego motion", and the car
  map may enter the final model only after a camera-only motion source (causal six-camera visual odometry, Astra)
  passes the same 0.397 test [choice: owner's camera-only rule for the final model; both members' ideas kept].
- P3 detector train leak (before training, Fable; metrics from Astra): predicted vs true boxes on 30 seeded train drives
  vs 30+ unused official-val drives (not decision / final): hull F1, recall, precision, centre error by distance.
  If recall or precision differs by > 0.05 or centre error by > 25% (or predicted train-val hull F1 gap exceeds the GT
  gap by > 0.05), training maps are degraded to the val error rates (drops, jitter, phantoms by distance).
- P4 10 Hz feed (dev drives; needed for X3 only): hold the last 2 Hz picture vs roll boxes forward (greedy same-class
  association, 2 m gate, birth at >= 0.5 or two hits >= 0.3, constant velocity in a common frame) vs detection on
  sweeps; cheapest path whose in-between-tick hull F1 is within 0.03 of refresh ticks; inspect 20 overlays incl. turns
  and exits.
- P5 timing: 100 updates, peak memory and one scored drive; branch switched off must reproduce E3 on the same noise.

## Experiments (H200 1g.18gb, <= 180 slice-minutes each; rendering / scoring counted separately)
- **X1, boxes:** E3 10k frozen, branch only [choice: Fable's frozen base, so the control is E3 itself on the flat keys
  where 0.318 / 0.763 were measured]. Flat keyframe keys, 144 train drives, style 2, batch 16, epsilon loss, AdamW 1e-4
  for the branch, 4,000 updates, checkpoint 2k, decision at 4k [Fable; Astra's cap rule: if 4k does not fit, the
  largest common affordable checkpoint is locked for X1 and X2 before scoring].
- **X2, boxes + grid:** as X1 with identical draws, plus the BEV grid sampled where each latent cell's ray meets z = 0
  and z = 1 m (invalid masked, 256 -> 32 channels) [choice: Fable's X2; it is the only arm aimed at the road gap
  (0.848 vs ceiling 0.94); Astra's oracle-trained arm replaced by an oracle TEST (below) that answers the same
  "branch vs detector" question without a training run].
- **X3, map + memory:** the X1/X2 winner's branch moved onto the locked temporal winner; 3,000 updates on lit dense keys
  + lit singles (temporal batch mix), 10 Hz maps from P4. Base frozen, unless X1 lowered precision vs E3 (upper bound
  < 0): then LoRA + adapter also train at 5e-5 and arm A's 4k -> 6k change is quoted as the extra-training effect.
  B/C keep their history conv and corruption; C's own rollouts regenerated with the map. If a training-free method won
  the temporal round, X3 tests the map at 10 Hz on E3.

## Decision rules (30 decision drives x 4 keyframes x seeds 0/1, DDIM 25, CFG 2, all 240 pairs; 2,000 paired drive
## bootstraps, seeds together, pooled F1 recomputed)
- X1 / X2 adopted: Mask2Former car F1 >= 0.382; gain over E3 >= +0.05 with lower bound > 0; SegFormer gain positive;
  road IoU difference >= 0 with lower bound >= -0.01.
- Controls: full swap (cameras, rays and map together) <= 0.05 vs the original key, correct - swapped lower bound > 0;
  map-only swap: correct - map-only-swapped lower bound > 0 [Astra; the cameras alone already place cars, so a fixed
  "half the gain" is not required]; the fraction of the gain removed is reported (Fable).
- Oracle test (true boxes at test time, no training on them): reported; < 0.45 means the branch, not the detector, is
  the limit.
- Train vs held-out: same judge on 30 fixed train drives (with precision and recall); train gain exceeding held-out
  gain by > 0.05 -> adoption withheld.
- Grid kept only for >= +0.02 car F1 or >= +0.01 road IoU over X1 (lower bound > 0), the other metric not down.
  Otherwise X1 is the winner; within 0.02 car F1 prefer X1.
- X3: lit VAE ceiling re-measured; car F1 >= max(0.382, half the lit ceiling); positive paired gain over memory alone;
  every signed temporal gate (PLAN-temporal.md) incl. flicker gains kept; median street-switch lag <= 5 ticks against
  the donor's cold F1; map checks: boxes shifted 2 m mid-rollout followed within 2 ticks, a deleted box gone within 5;
  an exited car persisting >= 0.5 s in two or more drives vetoes. Confirmed once on the final 24 (inconclusive ->
  unconfirmed).

## Stop rules
- Any preflight reject: no training. Map switched off does not reproduce E3: fix before training.
- At 2k, map-only swap moves car F1 by < 0.02: stop that arm, back to council.
- No map passes: no X3; back to council.
- Over time: X1 and X2 cut equally; the full scoring panel is kept. One training seed per arm, stated in the write-up.
