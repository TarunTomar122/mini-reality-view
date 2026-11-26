# Council brief: car map (spatial conditioning from a camera-only 3D model) for the third-person renderer

Read first: README.md, PROGRESS.md (newest first, especially 26 Nov 08:30-15:35), council/PLAN-bev.md (our earlier
converged plan for the OLD drone target), council/PLAN-tpv.md, council/PLAN-temporal.md (FINAL + amendments),
mrv/sd.py, scripts/train_sd.py, scripts/train_temporal.py, scripts/bev_extract.py, scripts/bev_box_judge.py,
scripts/judge_pair.py, scripts/sim_render.py.

## Where we are
- Goal: six real nuScenes camera photos in -> a game-style third-person chase view (own car visible, camera 7.5 m
  behind the ego origin, 2.4 m up, pitch -8 deg, 60 deg HFOV, 512x288, top 32 rows cropped -> SD latents 4x32x64).
  CAMERA-ONLY at run time.
- Renderer: Stable Diffusion 1.5 (frozen) + LoRA r=64 + CameraAdapter (DINOv2-small tokens 6x144x384 + viewing rays
  + camera embedding + style -> 865 context tokens replacing the CLIP prompt).
- Targets: exact MetaDrive game renders of the same moments (every labelled vehicle / pedestrian at its true pose).
- Experiment T (10k steps, 30 held-out drives, judge = Mask2Former car F1 vs the exact game key, IoU >= 0.3):
  E1 (init our old renderer, all 9.6k frames) 0.288; E3 (same init, quarter of the drives) 0.318; E2 (plain SD) 0.044
  at 5k. VAE ceiling (true keys through the SD VAE) 0.763. Wrong-street control 0.019. Road IoU ~0.85 (ceiling 0.94).
  Placement climbs slowly (E3 0.294 at 5k -> 0.318 at 10k). E1 - E3 = -0.030 [-0.077, +0.013].
- Placement diagnostic (26 Nov 15:35): BEVFormer-base (camera-only, train-only checkpoint, temporal BEV, 2 Hz
  keyframes) run on the 30 decision drives; its car boxes (score >= 0.3) projected as hulls into the chase view and
  matched to the game-key car masks: 0.417; nuScenes GT boxes with the same hull method: 0.459 (hulls are fatter
  than silhouettes, so 0.46 is that method's ceiling); our renderer 0.318 = 42% of its 0.763 ceiling. Overlays show
  the projection lines up. Conclusion: the cameras contain the placement; the renderer loses it.
- BEVFormer env (envs/bev: torch 1.10.2+cu113, mmdet3d 0.17.1, detectron2 v0.6, pillow 9.5) works on 2080 Ti / A6000
  (not H200). scripts/bev_extract.py can cache the 200x200x256 BEV grid resampled to a 48x32 ego grid, and/or boxes.
  nuScenes temporal infos exist for train and val. It runs on 2 Hz keyframes (and could be run on sweeps).
- In parallel: the temporal / memory round (council/PLAN-temporal.md) picks the memory method on E3 10k. The final
  model = car map + the winning memory method, with a short retrain.

## Questions
(a) Input form for the car map (must stay camera-only at run time): BEV feature grid (dense, warped into the chase
    view or as tokens) vs decoded detections rendered into the chase view (projected car masks / boxes / an instance
    or layout image through a ControlNet-style or zero-conv branch) vs extra cross-attention tokens. Which, and why?
    Consider: detector errors (missed / phantom cars), the game look (generic models), occlusion, the road layout
    (kerbs, lanes: road IoU ceiling 0.94, we are at 0.85), and the phone-photo path (later; uncalibrated).
(b) Init from E3 10k vs E1 10k.
(c) BEVFormer is temporal but runs at 2 Hz keyframes. Does its tracking help flicker? How to feed the renderer at 10 Hz
    (run on sweeps, interpolate boxes / grids, hold the last keyframe)?
(d) Interaction with memory: memory must not lock in misplaced cars. How to combine the car map with the winning memory
    method (order of training, what conditions what, failure checks).
(e) Exactly 3 experiments of ~3 GPU-hours each (H200 1g.18gb slice for training; 2080 Ti / A6000 for BEVFormer
    caching and scoring), with decision rules on car F1 vs the 0.763 ceiling (paired scene bootstrap over the 30
    decision drives, seeds kept together), the swap_scene control, road IoU, and a train-vs-held-out check. Also: what
    caching / code is needed, and stop rules.
Be practical, cite sources you rely on (verify they exist), honest about uncertainty; under ~1500 words.
