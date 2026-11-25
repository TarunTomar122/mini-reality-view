# Experiment W: 3D-aware encoder (merged plan)

Inputs: council/bev-brief.md, council/bev-s1-fable.md (Fable, round 1). Astra was unavailable in this session; the
second view is mine (the orchestrating agent), written after reading Fable's plan and the code. One round: we agree on
the core, the changes below are refinements.

## Agreed (from Fable's plan)
- Add, do not replace: DINOv2 tokens + rays stay; BEV features are an extra input with dropout.
- First pick BEVFormer-base (dense BEV 200x200x256, temporal memory, road layout + cars), checkpoint
  bevformer_r101_dcn_24ep.pth (train-only). Fallback StreamPETR (cars only), then SparseBEV (torch 2).
- Offline cache: BEV after the encoder, run recurrently per scene at 2 Hz from scene start, converted lidar -> ego
  frame, cropped x -8..+41 m, y +-16.4 m, 2x2 pooled -> 48x32 cells x 256, fp16 (~27 GB).
- Sanity gate: official val test on a subset, mAP within 2 points of 41.6. BEV check picture (PCA + GT boxes) before
  any training (catches flips).
- Phone rows (augmented token copies) always get the BEV null, so the DINO-only path stays trained; try_my_photos.py
  defaults to it.
- Leakage: valid on val (checkpoint train-only), but features are cleaner on train scenes than on val -> 15% BEV
  dropout + block dropout + small noise; report a train-scene vs val-scene gap. README says plainly that W uses
  extra supervision (nuScenes box labels inside BEVFormer).
- Go rule as in PLAN-sim: >= 0.05 car F1 or >= 10% lower kerb error vs A, paired scene bootstrap 95% CI excluding
  zero, no regression on the other, both judges agree on sign.

## Changes (second view)
1. **W2 first, W1 only if cheap.** The pixel-aligned warp (ground-plane lookup of the BEV grid for each 32x64 latent
   cell, zero-init 1x1 conv added after conv_in) is the stronger and cheaper design; cross-attention tokens (W1) add
   1,536 context tokens and slow every step. Run W2 against A first; W1 only if an H200 slice is idle.
   Add to W2 a second, height-aware sample: also sample the grid at the ray's crossing of z = 1 m (car-body height),
   concatenated (512 channels in), so tall objects above the ground point are not misplaced.
2. **Judge ceiling and judge resolution from the keys themselves.** Before W or sim verdicts, score the answer keys
   (lidar keys and trusted splat keys of the judge frames) with the judge at shifts 0/0.5/1/2 m. That is both the
   ceiling Fable asks for and a cleaner resolution test than model renders (keys have exact placement), which also
   addresses the judge-validation problem found on 25 Nov (F1 0.26 on SD-10k renders drops only 0.04 at 1 m).
3. **Install on a GPU node, download on the login node.** torch 1.9.1+cu111 cannot run on H200 (sm_90): extraction
   runs on A6000 (sm_86) / 2080 Ti (sm_75). mmdet3d 0.17.1 ops are compiled with conda-forge gcc and CUDA toolkit
   11.x (same major as cu111), only for sm_75/sm_86, MAX_JOBS low (nvdiffrast lesson). Half a day budget, then fall
   back.
4. **Control arm = sim-pilot arm A**, same sets/seed/steps (7k), as Fable says. The W arm adds the BEV path only;
   everything else identical, including phone rows.
5. **Later arm (not now): Depth Anything 3** metric depth per DINO patch -> 3D point per token. No nuScenes labels
   (no leakage) and it works on uncalibrated phone photos, which BEVFormer cannot. Candidate if W wins on val but
   breaks the phone path, or if W fails to install.

## Build order
1. env `bevformer` (conda-forge python 3.8, torch 1.9.1+cu111 wheels, mmcv-full 1.4.0 prebuilt cu111/torch1.9 wheel,
   mmdet 2.14.0, mmseg 0.14.1, mmdet3d 0.17.1 from source, detectron2) + checkpoint + can_bus.zip (login node
   downloads, GPU-node build).
2. create_data temporal infos (CPU job) -> sanity gate (A6000).
3. scripts/bev_extract.py (4 shards, A6000/2080 Ti) -> data/bev; scripts/bev_check.py picture.
4. mrv/sd.py BevWarp + conv_in hook; train_sd.py --bev --bev_mode warp --bev_drop 0.15; sd_render.py --bev.
5. Train W2 (7k, seed 7, same sets as A) when an H200 slice frees; judge vs A at 3k/5k/7k + train-scene gap.
