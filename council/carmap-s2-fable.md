# Car map, round 2 (Fable)

## 1. Agreements
- Form: detector boxes painted into the chase view, fed through a zero-initialised, pixel-aligned side branch. The
  six DINO streams stay. No attention tokens. The BEV grid is an add-on for the road.
- Start from E3 10k. Train on detector output. Map first, memory after; every tick gets its current map.
- Scoring: 30 drives x 4 frames x seeds 0/1, DDIM-25, CFG 2, 2,000 paired drive bootstraps (seeds together).
  Pass = car F1 >= 0.382 (half of 0.763) and >= +0.05 over the control with lower bound > 0; road IoU lower bound
  >= -0.01; full swap <= 0.05; train-vs-held-out flag at 0.05; final 24 drives sealed; one training seed per arm.

## 2. Where Astra is right (I converge)
- **Freeze E3 for the map test.** The control is then E3 itself, on the flat keys where 0.318 and 0.763 were
  measured, independent of temporal arm A and the lit re-render. With the base frozen a switched-off map is exactly
  E3, so my 15% whole-map dropout trains nothing. Dropped.
- **Telemetry audit.** BEVFormer shifts its previous grid by ego motion from `can_bus`, which the nuScenes loader
  fills from the ego pose (lidar localisation). My round 1 missed it; it also affects my roll-forward.
- **Experiment 3 = map + memory merge**, not my label-trained arm. No training on true boxes.
- Grid on top of boxes, with Astra's rule. Depth-tested box faces with the ego occluder; "unavailable" vs "empty"
  as separate states; causal feed only, score decaying with age, expiry at 0.5 s.

## 3. Disagreements, with evidence
1. **Injection depth.** Astra: one zero conv beside `conv_in`. With LoRA and adapter frozen this is the weakest
   setting: our LoRA does not touch the ResNet convs, and T2I-Adapter (frozen UNet) found injection at all four
   encoder scales best (arXiv 2302.08453). I keep four zero-init residuals.
2. **Length of the runs.** Astra: 2,000 updates inside 180 aggregate minutes. The owner's budget is 3 h on the
   slice with caching extra. A new branch at 1e-4 is slow to wake (the temporal plan scores 4k for this reason).
   I want 4,000 with a 2k checkpoint; the decision step stays prespecified (Astra's point).
3. **Leak check before, not after.** Astra flags a train-vs-held-out gap after training; finding it then costs a
   run. Measure the detector's own gap first and correct the training maps.
4. **Oracle evaluation.** One test-time render with true boxes (no training on them). Without it a failed run
   cannot be assigned to the branch or to the detector.
5. **10 Hz default.** Astra prefers direct detection on sweeps. BEVFormer's history was trained on keyframes from
   the past 2 s; its single-frame variant is clearly weaker (NDS 44.8 vs 51.7, velocity error about double; from
   memory, re-check in arXiv 2203.17270); sweep infos do not exist; cost is 5-6x. Settled by measurement (P4).
6. **Unfreezing at the merge.** A frozen base cannot stop drawing its own misplaced cars; the branch can only add
   on top. If experiment 1 shows doubled cars, the merge must train LoRA + adapter.

## 4. Plan I would sign

**Preflight (2080 Ti / A6000, no training)**
- P1 cache: `bev_extract.py --boxes_out` (all classes, score >= 0.2 as coded) plus grids, on 144 train + 6
  development drives, the 30 decision drives, the ~96 other val drives, and the final 24 (stored, unviewed). Check
  token coverage (the grid-file skip can omit boxes). Reject flipped projections and missing pairs.
- P2 telemetry: rerun the decision-drive box test with `can_bus` zeroed, with and without the previous grid. If
  hull F1 >= 0.397 (0.417 - 0.02), that mode is used for every cache. Otherwise the standard mode is used and every
  result is labelled "cameras + ego motion"; camera-only motion becomes a named follow-up.
- P3 train leak: predicted vs true car boxes (centre within 2 m, in view, <= 50 m): recall, precision, centre
  error, plus hull F1, on 30 train drives vs the 96 val drives (decision drives reported only). If recall or
  precision differs by > 0.05 or centre error by > 25%, training maps are degraded to the val rates (drops by
  distance, jitter, pasted phantoms), fitted on the 96 drives.
- P4 10 Hz feed (development drives, needed only for X3): hold last picture vs roll boxes forward by predicted
  velocity vs detect on sweeps. Take the cheapest path whose in-between-tick hull F1 is within 0.03 of refresh ticks.
- P5: 100 updates and one scored drive timed; branch off must reproduce E3 on the same noise.

**Picture and branch.** 256x512: car, large-vehicle and person/bike coverage x score; inverse depth; instance
boundary; heading sin/cos; box length (MetaDrive picks the car model from length); availability gate. Conv stem to
32x64 plus zero-init projections at the three coarser scales. 15% random box dropout. Same map in both guidance passes.

**X1, boxes.** E3 10k frozen; branch only. Flat keyframe keys, 144 train drives, style 2, batch 16, epsilon loss,
AdamW 1e-4, 4,000 updates, checkpoint 2k, decision at 4k.

**X2, boxes + grid.** As X1 with identical draws, plus the grid sampled where each latent cell's ray meets z = 0
and z = 1 m (invalid masked, 256 -> 32 channels). Runs alongside X1.

**X3, map + memory.** Branch of the X1/X2 winner moved onto the locked temporal winner; 3,000 updates on lit dense
keys + lit singles in the temporal batch mix, 10 Hz maps from P4. Base frozen, unless X1 lowered precision vs E3
(upper bound < 0): then LoRA + adapter train at 5e-5 and arm A's 4k -> 6k change is quoted as the extra-training
effect. B/C keep their history conv and corruption schedule; C's rollouts are regenerated with the map. If a
training-free method won, the base is E3 and X3 tests the map at 10 Hz.

**Decision rules**
- X1 adopted: car F1 >= 0.382; X1 - E3 >= +0.05, lower bound > 0, both judges agree in sign; road IoU difference
  >= 0 with lower bound >= -0.01.
- Controls: full swap (cameras, rays, map) <= 0.05 with correct - swapped lower bound > 0. Map-only swap must
  remove at least half of the gain over E3, else the gain is not from the map.
- Oracle (true boxes at test): reported; < 0.45 means the branch, not the detector, is the limit.
- Train vs held-out: same judge on 30 fixed train drives, with precision and recall. Train gain exceeding held-out
  gain by > 0.05: adoption withheld.
- Grid kept only for >= +0.02 car F1 or >= +0.01 road IoU over X1, lower bound > 0, other metric not down.
- X3: lit ceiling re-measured; car F1 >= half of it; positive paired gain over memory alone; every signed temporal
  gate; flicker gains kept. Map checks: boxes shifted 2 m mid-rollout are followed within 2 ticks; a deleted box is
  gone within 5; at a street switch car F1 reaches 80% of cold within 5 ticks. Persistent ghosts reject.

**Stop rules**
- Any preflight reject: no training.
- At 2k, map-only swap moves car F1 by < 0.02: stop that arm, back to council.
- No map passes: no X3; back to council.
- Over time: X1 and X2 cut equally; the full panel is kept.
