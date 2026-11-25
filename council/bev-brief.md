# Council brief: a "world model" encoder (3D-aware camera features) for the renderer

## Where we are
Mini Reality View: 6 nuScenes cameras -> camera tokens -> diffusion renderer -> chase-cam ("drone") picture,
9 m behind / 4.5 m above the ego car. CAMERA-ONLY at run time (no lidar input). Read README.md, PROGRESS.md (newest at
top), mrv/sd.py (CameraAdapter), scripts/train_sd.py, scripts/encode.py, council/PLAN-sim.md (how the geometry judge
works) for detail.

- Current tokens: DINOv2-small, frozen, per camera, per frame: 448x252 -> 32x18 patches, 2x2 pooled -> 6 x 144 x 384,
  plus a Plücker ray per patch from calibration (default nuScenes rig for phone photos).
- Main renderer: Stable Diffusion 1.5 + LoRA r=64 + CameraAdapter (tokens + rays + camera emb + 2D pos + style ->
  2-layer transformer -> 865 context tokens replacing the CLIP prompt). Trained on nuScenes train (700 scenes,
  ~28k keyframes at 2 Hz, + phone-style augmented copies). Answer keys: lidar splats and Gaussian-splat renders in the
  drone view (noisy). At 10k steps the pictures look like streets but car/kerb placement is vague.
- Geometry judge (Mask2Former / SegFormer on renders vs projected nuScenes boxes and map): car F1 + kerb error.
- Compute: Slurm, Teaching partition, max 8 CPUs/job, GPUs: 2080 Ti 11 GB, RTX A6000 48 GB, H200 slices 18/71 GB,
  jobs <= 2 days. Main env: Python 3.11, torch 2.15 (cu132), diffusers, transformers. Separate conda envs are fine
  (we built one for drivestudio with gcc 11 / CUDA 11.8 and one for MetaDrive). Compute nodes can't fetch big
  downloads reliably; the login node can.

## The gap
Tesla's renderer sits on their "Universal Vision Encoder": features trained end-to-end for driving, which already
encode 3D layout, other agents, occlusion and memory over time. Our DINOv2 tokens are generic, per-camera, per-frame,
with no 3D and no memory, so the renderer must infer the 3D layout itself from ~4 h of driving.

## Question
Should we replace or augment DINOv2 tokens with features from a PUBLIC, PRETRAINED, CAMERA-ONLY 3D perception model
trained on nuScenes (e.g. BEVFormer, BEVDet/BEVDepth, StreamPETR, SparseBEV, an occupancy network such as
SparseOcc/FlashOcc/BEVDet-Occ, or anything newer and better), and if so exactly how?

Please answer concretely:
1. Which model and checkpoint (URL), and why. Weigh: camera-only, quality on nuScenes val, temporal memory, how hard
   it is to install today (mmcv/mmdet3d version pain vs plain PyTorch), speed, licence.
2. Which features to take (BEV grid? occupancy? object queries? multi-level?), at what resolution, and how to feed
   them to the SD CameraAdapter (tokens count, positional encoding in ego metres, alongside or instead of DINOv2).
   Note the drone camera is fixed relative to the ego, so a BEV grid in ego coordinates maps to the drone view
   by a fixed geometry.
3. Train/val leakage: the perception model was trained on nuScenes train with labels; our renderer trains on the
   same train scenes and is judged on val. What does that do to the comparison and how should we handle it?
4. Phone photos (a goal): users give 6 uncalibrated phone photos; we assume the default nuScenes rig. How badly would
   a BEV model break, and how do we keep that path working (fallback, augmentation)?
5. Experiment design that fits next to the running work: feature extraction as an offline cache (like our DINOv2
   tokens), then a comparison arm vs the current adapter at equal steps, judged with the geometry judge (car F1, kerb
   error) plus LPIPS. Rough GPU-hours and calendar time; what can run on 2080 Ti / A6000 while the H200 slices are busy.
6. Risks and what would make you call it off early.

Keep it practical: something we can build and run in 1-3 days on this cluster.
