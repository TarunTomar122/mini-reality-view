# Car map, round 1 (Fable)

**Recommendation in one line:** draw BEVFormer's tracked 3D boxes into the chase view as a small multi-channel
picture and feed it through a zero-initialised, pixel-aligned side branch; start from E3 10k; keep BEVFormer at 2 Hz,
track its boxes and roll them forward to 10 Hz; train memory afterwards with the map switched on.

## (a) Input form: decoded boxes drawn in the chase view

Why boxes, not the feature grid or tokens:
- **It removes the step the renderer fails at.** The boxes already land on the game cars (0.417 vs 0.459 for GT
  boxes). The chase camera is fixed to the car, so box -> pixels is exact geometry with nothing to learn. Tokens
  (MagicDrive puts boxes in cross-attention [2]) make the network learn the projection again, which is what our
  DINO tokens + rays fail at. Panacea feeds BEV layouts through a ControlNet [3][4] (exact channels not checked).
- **The game target is a function of the boxes.** MetaDrive picks each car's 3D model from the logged length alone
  (<= 4 m small, <= 5.2 medium, <= 6.2 large, else XL; no random call) [7]. A length channel tells the renderer
  which model to draw: that targets "edge cars change model between frames" without memory. (Rescaling to the
  logged size and colours: not confirmed.)
- **10 Hz and other detectors come free.** Boxes interpolate; a held feature grid leaves a 15 m/s car up to 6 m
  stale. The picture is detector-agnostic; the grid ties us to a torch 1.10 env that cannot run on H200.
- **Road layout: the grid is weaker than PLAN-bev assumed.** The checkpoint is trained on detection only, so kerbs
  and lanes are in its features only incidentally. Road IoU is 0.85 of 0.94; not the target here. Later: a
  camera-only map network drawn into the same picture, noting that nuScenes val overlaps train geographically [8].

The picture (256x512, painted far to near, reusing `bev_box_judge.paint`): occupancy x detector score for four
groups (car; truck/bus/trailer; person/bike; barrier/cone), inverse depth, heading sin/cos, box length, instance
boundary, and a "map present" plane. Score as intensity avoids a hard threshold.

Branch: a conv stem (stride 8 -> 32x64) added after `conv_in`, plus zero-init 1x1 projections at the three coarser
encoder scales (T2I-Adapter: all four scales work best with a frozen UNet [5]; diffusers has a hook for it [6]).
Our LoRA does not touch the ResNet convs, so input-only injection is a risk. A full ControlNet copy [4] is too
heavy for 3 h.

Detector errors:
- Missed cars: 15% whole-map dropout plus random box dropout, so DINO tokens stay a second source.
- Phantoms: train on real detector output so the renderer learns how far to trust it.
- **Main risk: BEVFormer was trained on our training drives**, so its boxes are cleaner there than on held-out
  streets, and the renderer will over-trust the map. Unmeasured; measure first (preflight).
- Phone path: map = null. Any later monocular 3D detector can draw the same picture.

## (b) Init: E3 10k

E1 - E3 = -0.030 [-0.077, +0.013]: a tie leaning to E3. E3 is the temporal round's init and train list, which
gives a free equal-update control (temporal arm A) and a simple final merge. Keyframe singles already come from all
train drives, so E1's extra drives are still seen. Not worth one of three experiments.

## (c) 2 Hz detector, 10 Hz renderer

BEVFormer has memory, not tracking: it fuses the previous BEV grid and reports better velocity and low-visibility
recall [1], but outputs no track IDs, and each frame is decoded independently, so boxes near the score threshold
still blink. Val errors: mATE 0.673 m, mAVE 0.394 m/s [1].

Plan: keep it on keyframes (in distribution). Add a small tracker (CenterPoint-style greedy nearest-centre matching
with predicted velocity [9]): IDs, birth after 2 hits or score >= 0.5, death after 3 misses, smoothed size and
heading (stable length = stable car model). For each 10 Hz tick, move every track at constant velocity in the world
frame from the latest *past* keyframe, then re-project with the current ego pose; the same causal rule in training
and at run time. Extra error over 0.4 s is ~0.15-0.25 m, below the judge's 1-2 m resolution. Cost: a visible 2 Hz
correction step; tracker smoothing is the knob.

Sweeps are the fallback: 0.08 s gaps are outside its training spacing (previous frames drawn from the past 2 s
[1]), sweep infos do not exist, and 10 Hz noise needs filtering anyway. Try on the dev drives only if the
preflight fails.

## (d) With memory

- **Roles.** Map = fresh placement every tick, from the cameras. History = look and identity only.
- **Order.** Map first (this round). Then the winning memory recipe is retrained ~3k updates from the map winner
  with the map on (C's rollouts regenerated with the map). Do not add LoRA weights from two separate runs. If the
  training-free warm start wins: no retrain, re-check its gates with the map on.
- **Memory in the map.** Tracker coasting (keep an occluded track a few ticks, score decaying) is object permanence
  that is explicit and forgets on schedule. Test on the temporal plan's hidden-car events.
- **Failure checks** (added to the temporal gates): (1) shift all map boxes 2 m sideways mid-rollout: drawn cars
  follow within 2 ticks; (2) delete a track: car gone within 5 ticks; (3) history from a wrong-map rollout +
  correct current map: car F1 within 0.02 of cold; (4) rollout car F1 >= cold - 0.02 with the map on.

## (e) Three experiments (6,000 updates each, <= 180 slice-minutes, checkpoints 2k / 4k / 6k)

All inside `train_temporal.py` with arm A's exact draws (new draws appended, so A's stream is unchanged): **arm A is
the control at equal updates, at no cost.** Init E3 10k, LR 5e-5 LoRA + adapter, 1e-4 branch.

- **M1, detector-trained boxes:** maps = tracked BEVFormer boxes in training and test. The camera-only candidate.
- **M2, label-trained boxes:** training maps = true boxes from the ScenarioNet tracks (what MetaDrive replays).
  Tested with true boxes (ceiling of the mechanism) and with BEVFormer boxes (does it survive real errors?). M1 is
  also tested with true boxes: a 2x2 table separating mechanism, detector and over-trust.
- **M3, BEV grid:** PLAN-bev W2 retargeted (grid sampled where each latent cell's ray meets z = 0 and z = 1 m;
  in-between ticks = last keyframe grid shifted by ego motion), same branch.

Scoring: decision drives, cold, keyframe ticks, 4 frames x 2 seeds, `judge_pair.py`, both judges; paired scene
bootstrap, 2,000 draws, seeds together. Re-measure the 0.763 ceiling on the lit keys first.

Decision rules (camera-only test maps):
- **Adopt the map** if best arm - A >= +0.05 car F1, interval excluding 0, judges agree in sign, AND car F1 >= 0.38
  (half the ceiling). My guess is 0.45-0.60, no more than a guess.
- **Road IoU:** lower bound of the difference vs A >= -0.01.
- **Controls:** full swap_scene <= 0.05. Map-only swap (right cameras, another drive's map) must remove at least
  half of (normal - full swap), else the map is being ignored. Null map: report vs A; flag a drop > 0.03.
- **Train vs held-out:** same judge on 30 E3 training drives. Gap exceeding A's by > 0.05 = over-trust: prefer the
  other box arm, or add the ~96 unused val drives as honest-map rows.
- **M1 vs M2:** higher held-out car F1; within 0.02 -> smaller train gap.
- **Grid:** replaces boxes only at >= +0.03 car F1 over the best box arm (interval excluding 0); added on top only
  for >= +0.02 road IoU. Otherwise dropped.

Preflight (no training, 2080 Ti / A6000, sharded; my estimate ~4-6 GPU-hours at ~2 frames/s, untimed):
1. `bev_extract.py --boxes_out` (all classes, score >= 0.1) on all train drives, the ~96 val drives outside
   decision / final, and the final 24 (stored, not looked at). `bev_box_judge.py` on train vs held-out frames = the
   leakage gap. If train - val hull F1 > 0.05, M1 degrades its train maps (drops, jitter, phantoms) to the rates
   measured on the 96 drives.
2. Tracker + roll-forward on the dev drives: hull F1 at in-between ticks within 0.03 of keyframe ticks, else sweeps.
3. Hand check of 20 overlays; confirm the length -> model rule in our MetaDrive version.

Code: `scripts/carmap_track.py`; `mrv/carmap.py` (one rasteriser for training, rendering, judging); `MapBranch` in
`mrv/sd.py`; `--map` in `train_temporal.py`, `sd_render.py`, `video_render.py`; grid features for M3 only (~8-16 GB).

Stop rules:
- At 2k, map-only swap changes car F1 by less than a quarter of (normal - full swap): one fix (branch LR x3), then
  stop that arm.
- M2 with true boxes < 0.45 at 6k: the mechanism is the limit, not the detector. Back to council (ControlNet copy).
- No arm reaches +0.05 over A: stop; "the renderer loses placement" needs re-examining (far cars at 1 latent cell,
  judge matching).
- Timing: 100 updates timed first; all arms cut equally.

## Unsure
- Size of the detector's train-vs-val gap (decides M1 vs M2).
- Whether 6k updates is enough for a new branch; the 2k / 4k / 6k trend will say.
- Hull F1 saturates at 0.46, so how much of the 0.318 -> 0.763 gap the map closes is unknown.
- This round depends on temporal arm A and its dense caches.

## Sources (all found today)
1. BEVFormer, arXiv 2203.17270 (val NDS 51.7, mAP 41.6, mATE 0.673, mAVE 0.394); repo README: no tracking output.
2. MagicDrive, arXiv 2310.02601: boxes via cross-attention, road map via an additive encoder.
3. Panacea, arXiv 2311.16813. 4. ControlNet, arXiv 2302.05543. 5. T2I-Adapter, arXiv 2302.08453.
6. diffusers `unet_2d_condition.py`, `down_intrablock_additional_residuals` (seen in a summary, not read verbatim).
7. MetaDrive `manager/scenario_traffic_manager.py`, `get_vehicle_type` (main branch; our version may differ).
8. Lilja et al., arXiv 2312.06420 (>80% of val samples within 5 m of a train sample); StreamMapNet, 2308.12570.
9. CenterPoint, arXiv 2006.11275.
