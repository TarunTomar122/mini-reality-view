# Experiment W, round 1 (Fable): a 3D-aware encoder for the renderer

**Recommendation: yes, as an add-on, not a replacement.** Add BEVFormer-base's BEV grid to the existing DINOv2
tokens and test two ways of feeding it in. Fallback if it will not install in half a day: StreamPETR object queries.

Checked on 25 Nov 2026: the checkpoint URLs below answer HTTP 200 with the stated sizes, and the install pins are
copied from each repo's own install doc. Nothing was installed or run, so every speed and memory figure is an estimate.

## 1. Model and checkpoint

| | BEVFormer-base (first pick) | StreamPETR V2-99 (fallback) |
|---|---|---|
| Output | dense BEV grid 200x200x256, ±51.2 m, 0.512 m/cell | 900 object queries x 256 + 3D boxes |
| nuScenes val | NDS 51.7 / mAP 41.6 | NDS 57.1 / mAP 48.2 |
| Memory over time | recurrent: previous BEV, aligned by ego motion | queue of past object queries |
| Road layout in the features | yes (implicit) | no, objects only |
| Install pins | py3.8, torch 1.9.1+cu111, mmcv-full 1.4.0, mmdet 2.14.0, mmseg 0.14.1, **mmdet3d v0.17.1 built from source**, detectron2 | py3.8, torch 1.9.0+cu111, mmcv-full 1.6.0 (wheel), mmdet 2.28.2, mmseg 0.30.0, mmdet3d v1.0.0rc6 (`pip install -e`), flash-attn optional |
| Checkpoint | 277 MB, GitHub release | 338 MB, GitHub release |
| Licence | Apache-2.0 | Apache-2.0 |

- BEVFormer: https://github.com/fundamentalvision/BEVFormer, config `projects/configs/bevformer/bevformer_base.py`,
  weights https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth
- StreamPETR: https://github.com/exiawsh/StreamPETR, config `stream_petr_vov_flash_800_bs2_seq_24e.py`,
  weights https://github.com/exiawsh/storage/releases/download/v1.0/stream_petr_vov_flash_800_bs2_seq_24e.pth
  (lighter option: `stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth`, 155 MB, NDS 54.6, 31.7 FPS on a 3090).

**Why BEVFormer first.** Our two judge numbers are car F1 and kerb error. A dense BEV grid is the only candidate that
carries road layout as well as cars, and it is already a ground-plane grid, which maps to the fixed drone camera by
fixed geometry. Sparse detectors (StreamPETR, SparseBEV) score higher on detection but say nothing about kerbs.
BEVFormer-small (150x150, NDS 47.9) and tiny (50x50, 2 m cells, NDS 35.4) are too weak or too coarse.

**Why it might fail.** It has the oldest stack of all candidates. mmdet3d 0.17.1 compiles its own CUDA ops, and
torch 1.9+cu111 against our CUDA 11.8 / gcc 11 toolchain may not build. I found no maintained plain-PyTorch port
with weights, and no confirmed mmcv 2.x / mmdet3d 1.x port.

**Considered and not picked:**
- FlashOcc / BEVDet-Occ (https://github.com/Yzichen/FlashOCC, Apache-2.0, mIoU 32-43.5): explicit occupancy would be
  attractive, but weights are on Google Drive / Baidu, it needs mmdet3d 1.0.0rc4 plus BEVDet's compiled pooling op,
  and 0.4 m voxels at ~32 mIoU are blobby. Second-round candidate.
- SparseBEV (https://github.com/MCG-NJU/SparseBEV, MIT, NDS 55.6 R50): torch 2.0 / cu118 and a pure-PyTorch sampling
  option, so it is the easiest install; weights on Google Drive. Use it if StreamPETR also fails. Do not use its
  `trainval` checkpoints (trained on val).
- Outside the brief but worth a later arm: Depth Anything 3 (https://github.com/ByteDance-Seed/Depth-Anything-3,
  `pip`, torch >= 2, metric model Apache-2.0). Depth per DINO patch turns each camera token's ray into a 3D point.
  Not trained on nuScenes labels, so no leakage, and it works on uncalibrated phone photos.

## 2. Which features, and how they enter SD

**Take:** the BEV embedding after the 6-layer encoder (what the model keeps as `prev_bev`), 200x200x256, run
recurrently through each scene in keyframe order (2 Hz, as it was trained), reset at scene start.

**Cache:** crop to 49 m forward x 33 m wide around the drone's field of view (x -8..+41 m, y ±16.4 m in the ego
frame), average-pool 2x2 -> **48x32 cells of 1.02 m, 256-d, fp16**, about 0.8 MB/frame, ~27 GB for 34k keyframes
(the DINO cache is 0.66 MB/frame).

**Frame gotcha:** BEVFormer's grid is in the lidar frame (x right, y forward, sensor ~0.9 m ahead of the rear axle),
not our ego frame. Convert with the per-scene lidar-to-ego calibration and verify by plotting ground-truth box
centres on a PCA picture of the grid.

Two arms; keep DINOv2 tokens in both:

- **W1, extra prompt tokens (smallest change).** `LayerNorm -> Linear(256, 768)` + Fourier features of the cell's
  (x, y) in metres + Fourier features of the drone pixel (u, v) where that ground point lands + a learned "BEV" type
  embedding. Concatenate with the 864 camera tokens before the 2-layer mixer: 864 + 1536 + 1 = 2401 context tokens.
  The null set grows to match.
- **W2, pixel-aligned map (I expect this one to win).** For each of the 32x64 latent cells of the drone picture,
  intersect its ray with the ground (z = 0) and bilinearly sample the BEV grid there; cells above the horizon or
  outside the grid get zero. Feed the 256x32x64 map through a zero-initialised 1x1 conv and add it to the UNet's
  `conv_in` output. About 30 lines; the warp is a constant lookup table. Cross-attention alone is a weak way to tell
  SD *where* things go, and a spatially aligned input is the standard fix (ControlNet, T2I-Adapter).

For the StreamPETR fallback: top 128 queries by score, each `MLP(256-d embedding)` + Fourier(x, y, z) + size, yaw,
velocity, score -> 128 tokens (W1 style), and optionally predicted footprints painted into the drone view (W2 style).

## 3. Train/val leakage

- The val comparison is valid: this checkpoint was trained on nuScenes train only, so val scenes are unseen by the
  perception model, the renderer and the judge truth alike.
- The real problem is a **train/val feature gap**. On train scenes BEVFormer has seen the labels, so its features
  are cleaner than on val (val mAP is only 41.6). The renderer learns to trust them and gets worse inputs at test time.
- Handling:
  1. Regularise: drop the whole BEV input on 15% of rows, drop random 4x4 cell blocks, add small Gaussian noise.
  2. Measure it: score both arms on 30 train scenes as well as the 30 val judge scenes. If W's train-minus-val gap is
     much larger than A's, the renderer is over-trusting.
  3. The clean cure, K-fold retraining of the perception model, costs days on 8 GPUs. Not worth it now.
- Say plainly in the README that arm W uses extra supervision (nuScenes 3D box labels) that arm A does not.

## 4. Phone photos

Expect it to break badly. BEVFormer projects each BEV cell into the images with exact calibration, at
1600x900 nuScenes intrinsics, and its memory needs ego motion from the CAN bus. With an assumed rig, unknown phone
FOV and a single frame, features land in the wrong cells. Even on real nuScenes, removing the temporal memory
costs several NDS points. I have no measured number for wrong-rig input.

Keep the path working by construction:
- DINOv2 tokens stay. Phone-copy training rows always get the BEV input replaced by a learned null, so the renderer
  keeps a trained DINO-only mode, and `try_my_photos.py` uses that mode by default.
- Later, if W wins: run BEVFormer on the phone-augmented copies with the default rig and no history, and measure
  whether BEV-on beats BEV-off on phone-like val photos before enabling it.

## 5. Experiment and build list

Control = sim-pilot **arm A** as already trained (real 11 lidar + 5 splat, 7k steps, seed 7, snapshots 3k/5k/7k).
W1 and W2 use the same sets, seed and steps, so no new baseline is needed.

1. **Env `bevformer`** (login node, ~2 h): install per `docs/install.md`. Download the checkpoint and nuScenes
   `can_bus.zip` on the login node. Run `tools/create_data.py ... --canbus` (CPU job) to get
   `nuscenes_infos_temporal_{train,val}.pkl`.
2. **Sanity gate** (A6000, 20 min): official test script on ~600 val samples. Must give mAP within 2 points of 41.6.
3. **`scripts/bev_extract.py`** (in the BEVFormer env): loop scenes in order, grab `prev_bev` after each frame,
   lidar->ego, crop, pool, save `data/bev/<scene>.pt` with `sample_tokens`. Sharded like `encode.py`.
   850 scenes, ~34k frames at an assumed 2-3 FPS: **4-5 GPU-h**, four A6000 or 2080 Ti jobs, ~1.5 h wall.
   (11 GB should be enough for inference; unverified.)
4. **`scripts/bev_check.py`**: PCA of the grid with GT boxes and map edges drawn on it, 5 scenes. Catches flips.
5. **`mrv/sd.py`**: `CameraAdapter(bev=...)` for W1; `BevWarp` module plus `conv_in` hook for W2.
   **`train_sd.py`**: `--bev data/bev --bev_mode tokens|warp --bev_drop 0.15`. `sd_render.py`: same flags.
6. **Train W1 and W2**, 7k steps each: ~2.5 GPU-h each on an 18 GB H200 slice, ~4 h on an A6000 (W1's longer
   context may need batch 8 x 2 accumulation there). A 2080 Ti is too small for training.
7. **Judge**: existing set (val scenes 30-59, 4 frames x 2 seeds, style 1, DDIM 25, CFG 2), Mask2Former + SegFormer,
   `--compare` against A at 3k/5k/7k; LPIPS reported only. Plus the 30-train-scene gap check from section 3.

Total about **12-15 GPU-hours**. Calendar: day 1 env, gate, extraction; day 2 code, both arms, judging. Everything
except the two training runs fits on 2080 Ti / A6000 while the H200 slices are busy; the old torch build probably
cannot use the H200s at all.

**Go rule** (same as PLAN-sim): >= 0.05 car F1 or >= 10% lower kerb error vs A at 7k, paired scene-bootstrap 95% CI
excluding zero, no regression on the other metric, both judges agreeing on sign.

## 6. Risks and call-off points

- **Install** (most likely failure). No passing sanity gate after half a day -> switch to StreamPETR (no compiled
  ops beyond the mmcv wheel); if that fails too, SparseBEV on torch 2.0.
- **Frame or flip bug.** The BEV check fails -> fix before any training; a mirrored grid would silently look like
  "BEV does not help".
- **Judge not validated.** If the shift/shuffle validation of the judge fails, W is as inconclusive as sim.
- **Over-trust.** W's train-val gap far above A's and no val gain -> raise dropout once, then stop.
- **No signal.** Neither W1 nor W2 meets the go rule at 7k -> call it off, write it up under "What did not work",
  and consider the Depth Anything 3 arm instead.
- **Noisy keys cap the gain.** If lidar/splat keys misplace cars themselves, better inputs cannot fix it; the judge
  on the keys themselves gives that ceiling.
- **StreamPETR fallback** can only help car F1, not kerbs.
