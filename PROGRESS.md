# Mini Reality View: progress (newest at top)

## 2026-11-25 12:50
- Aug test OK: augmented tokens align with clean ones (same sample order), mean cosine to clean tokens 0.81. Launched 2 full augmented copies of all 850 scenes (8 jobs) -> data/enc_aug_{0,1}.
- train_sd.py: --tokens_aug (phone-like copies as extra training rows, train scenes only, own rays) and --init. CameraAdapter + LORA_TARGETS moved to mrv/sd.py.
- scripts/try_my_photos.py: 6 photos -> EXIF-rotate, centre-crop 16:9, 448x252 -> DINOv2 -> pooled tokens; default rig rays assets/default_rays.pt (mean over 850 scenes); renders with the SD checkpoint (CFG) or a DiT checkpoint; CUDA / MPS / CPU.
- mix_high 25.4k/30k; c-selfprev 14.8k/20k; low lidar targets 824/850.

## 2026-11-25 12:20
- mix_high 20k/30k (loss 0.16); c-selfprev 10.6k/20k; low lidar targets 777/850 (then enc-low -> low_lidar / low_mix_* and lat512-low -> sd_run1).
- P (phone photos) started: scripts/encode_aug.py — per camera independently: zoom-crop z in [1,1.3] at a random offset (narrower FOV + small yaw/pitch jitter), brightness/contrast/saturation x0.75-1.25, hue +-0.03, gamma 0.8-1.25, blur (50%, sigma <= 1.2), JPEG q 35-95; rays recomputed for the patches inside the crop. Small test job 3750677 (2 scenes) before the full copies.

## 2026-11-25 11:50
- mix_high 14.6k/30k (loss 0.19); c-selfprev 6.6k/20k; low lidar targets 613/850; low splat keys encoded with/without trust (enc_splat_tv_low, enc_splat_tv_low_trust).
- Queued after enc-low (all 45M + rays, 40k, bs 64): low_lidar (3750659, step-2 model comparison vs A_rays at the high drone), low_mix_nomask (3750660) vs low_mix_trust (3750661) (step-1 trust-mask comparison; splat mix 0.3).

## 2026-11-25 11:50 — Step 3 started early (user asked): SD 1.5 renderer
- scripts/train_sd.py: frozen SD 1.5 UNet (base weights bf16/fp16) + LoRA r=64 on attention/proj/ff + CameraAdapter (DINOv2 tokens 384->768, + ray MLP, per-camera + 2D positions, style token, 2-layer transformer -> 865 tokens replacing the CLIP text tokens) + learned null tokens (10% drop) for CFG. SD's own DDPM schedule, epsilon prediction, coverage/trust-weighted loss (hole_w 0.05), 512x256 keys (latent 4x32x64). Render: DDIM 25 steps, CFG 2.0. 42.9M trainable.
- Smoke test (300 steps, 70 trainval splat keys low+trust, H200 1g.18gb, bf16): runs, ~1.7 min / 100 steps at bs 16. Pictures (pulled/sd_smoke.jpg): photo-like street textures already (trees, road, buildings), not yet following the cameras (as expected at 300 steps). 2 val scenes LPIPS 0.428.
- Queued sd_run1: keys = low-drone lidar (style 0, all 700 train scenes) + low-drone splat with trust masks (style 1, 70 scenes), mix 0.3, 40k steps (~11 h), after 4 encode_latents shards that wait on the low lidar targets (608/850 done).

## 2026-11-25 11:20
- mix_high 9k/30k (loss 0.22). c-selfprev 2.4k/20k (its "min" column is wrong: the self-prev loop reused the timer variable t0; renamed to ta/tb for future runs, training itself unaffected).
- Trust masks for the 70 trainval low splats done. Encoding low splat keys without / with trust masks (tokens reused from enc_full): 3750618 / 3750619 -> data/enc_splat_tv_low{,_trust}.
- Low lidar targets 430/850; enc-low (4 shards) waits on them.
- One rsync dropped mid-transfer (unexpected EOF); retried fine.

## 2026-11-25 10:45
- Video drift fix 1+2 combined (s_prev 0.3 + reset every 8): L1 26.76, LPIPS 0.376, first5->last5 27.5->30.4 (~same as s_prev 0.3 alone). Single-frame full_b stays flat (29.4->29.6), so the remaining +3 is real drift.
- Fix 3 started: train.py --self_prev 0.5 — for half the samples the previous frame is drawn by the model itself (EMA, 2 steps, single-frame mode, from the previous keyframe's camera tokens) instead of the perfect key; fine-tune full_c for 20k steps, lr 1e-4 (c-selfprev 3750591, H200 18 GB slice).
- mix_high running (3.6k/30k). Low lidar targets 266/850. Trainval low splats 70/70 rendered; trust masks running (16 done).

## 2026-11-25 10:20
- Experiment A on 30 val scenes, 4 steps (45M, 700 scenes, 40k): no rays L1 26.22 / LPIPS 0.384; rays 26.27 / 0.377. Small gain for rays; keep them (also needed for phone photos). Pictures: pulled/rays_ablation.jpg (kerbs/road edges slightly better placed).
  Note: these 45M runs (L1 26.2) beat full_b (110M, 27.8) on L1 and are close on LPIPS (0.377-0.384 vs 0.374).
- Video drift fixes on full_c (30 val scenes, 4 steps): s_prev 0.1: LPIPS 0.389, L1 first5->last5 27.4->35.2 | fix 1 s_prev 0.3: 0.374, 27.4->30.8 | s_prev 0.5: 0.408, 28.5->34.2 (beyond the 0-0.3 training range) | fix 2 reset every 8: 0.385, 27.3->30.1. Best: s_prev 0.3. Drift reduced by more than half, not gone. Next: s0.3 + reset 8 (3750559); fix 3 (train on own rollouts) if needed.
- mix_high OOM'd on an 11 GB 2080 Ti (110M, bs 64) -> resubmitted on gpu:h200_1g.18gb (3750558). Teaching partition gres names: nvidia_geforce_rtx_2080_ti, nvidia_rtx_a6000, h200, h200_3g.71gb, h200_1g.18gb.
- Full low-drone lidar targets: 74/850 scenes so far (8 shards running).

## 2026-11-25 10:00
- ssh master had gone stale ("Connection closed by UNKNOWN port 65535"): ssh -O exit mlp, reconnect -> fine.
- Stage 5 video model full_c (512x16, 80k) on 30 val scenes, whole-scene rollouts, 4 steps, s_prev 0.1: masked L1 29.95, LPIPS 0.389 (single-frame full_b 27.8 / 0.374). DRIFT: L1 first 5 frames 27.4 -> last 5 frames 35.2. Stage counts as failed; rescues:
  fix 1: more noise on the previous frame at run time (s_prev 0.3, 0.5: 3750530/31); fix 2: restart from pure noise every 8 frames (3750532); fix 3 if needed: retrain on the model's own rollouts.
- Experiment A finished (40k each): evals 3750528 (no rays) / 3750529 (rays). evaluate.py now handles rays (+ --reset_every).
- enc_splat_tv done (70 scenes). Style-mix run mix_high: init full_b, lidar keys (style 0) + trainval splat keys high drone (style 1), mix 0.5, lr 1e-4, 30k steps (3750533).
- Low-drone lidar keys for all 850 scenes: 8 target shards -> 4 encode shards with --tokens_from data/enc_full (no DINOv2 rerun) -> data/enc_full_low. Trust masks for the 70 trainval low splats queued after the last re-render (4 shards).

## 2026-11-25 09:15
- Trust masks v2 (opposite views count half): mean trust per scene low 0.38-0.54, high 0.30-0.63. The dark band at the bottom of the low-drone masks is unchanged: it is the ground under/right around our own car (bottom of the low-drone frame ~ the ego position), which no camera sees -> correctly untrusted (the final view draws the car there). v1 diagnosis was wrong; the angle change is kept as reasonable.
- Experiment A smoothed train loss (no rays / rays): 10-15k 0.841/0.826, 20-25k 0.746/0.753, 28-34k 0.681/0.667. Small edge for rays so far; single-batch losses were misleading.
- full-train-c (video, 512x16, 80k) DONE; in-training rollout masked L1 on 6 val scenes 24.7-34.4. Full eval with drift: eval-full-c 3750514 (30 val scenes, 4 steps).
- enc_splat_tv 67/70; trainval low re-renders 44/70.

## 2026-11-25 08:45
- Trust masks v1 done on mini (low drone mean trust 0.30-0.51 per scene, high drone 0.13-0.50 -> low drone keys more trustworthy, matches the pictures). Pictures (pulled/trust_masks.jpg): street, cars and facades trusted, sky not; but the road right behind the car was untrusted although the back camera sees it — the angle weight clip(cos,0,1) gave 0 when the real and drone views point in opposite directions. Fix 1: weight 0.5 + 0.5*clip(cos) (flat ground looks the same from both sides). Rerun 3750477/78.
- Experiment A at 24k/40k: train loss with rays 0.658 vs 0.738 without (clear gap).
- enc_splat_tv 28/70 encoded; low re-renders of the 70 trainval splats running (7 jobs); full-train-c 75.8k/80k; mini scene 9 splat finally running on the A6000.

## 2026-11-25 08:15
- All 70 trainval splat scenes done -> encoding them (enc-splat-tv 3750446); low-drone re-renders of the 70 queued (7 jobs x 10 scenes, 3750447-53).
- Step 2 keys: mini splat re-render at the LOW drone done. Pictures (pulled/splat_high_vs_low.jpg): far fewer floaters and colour junk, sharper cars/facades/excavator, sky now visible; remaining smears at the bottom corners (right behind the car, unseen by any camera) and one sky glitch (scene-0655). Low drone is clearly the better key; model comparison at both heights still to do.
- Experiment A at ~14.5k/40k: loss with rays 0.779 vs 0.790 without (gap narrowing). full-train-c 70k/80k.

## 2026-11-25 07:45
- Stage 7 distillation (1-step student of full_b's 4-step output, 15k steps, final loss 0.011): val masked L1 27.4 (vs 28.9 plain 1-step) but LPIPS 0.394 (vs 0.381 plain 1-step, 0.374 plain 4-step). Pictures (pulled/distill_sheet.jpg): very close to the 4-step teacher, a bit smoother. Verdict: no gain over plain 1-step sampling; keep plain few-step sampling for now and revisit with a perceptual/adversarial loss for the SD1.5 model.
- Experiment A at 5.2k/40k: train loss with rays 0.894 vs 0.952 without (early).
- Trainval splats 67/70; mini re-renders running (12 scene dirs written so far); full-train-c 65k/80k.

## 2026-11-25 07:20
- calib_rays.pt done (850 scenes, 6x144 Plücker rays each). Lidar keys at the low drone done for mini (coverage 0.21-0.81, about the same as high).
- Experiment A launched: 45M DiT, all 700 train scenes, 40k steps, bs 64, identical except --rays: runs/A_norays vs runs/A_rays.

## 2026-11-25 07:15
- Stage 7 step baselines, full_b on 30 val scenes: 1 step L1 28.9 / LPIPS 0.381; 2 steps 27.5 / 0.383; 4 steps 27.8 / 0.374. One step is already within 0.007 LPIPS of four -> flow-matching paths are nearly straight here; distillation (10.6k/15k, loss 0.031) can only close a small gap.
- full-train-c 59.8k/80k. Trainval splats 59/70. Mini low/high re-renders running; trust masks queued behind them.

## 2026-11-25 06:50
- Trainval splats 51/70; full-train-c 54k/80k; distillation + 1/2-step baselines running; SD1.5 UNet/VAE + Difix + peft downloaded (DL_DONE). CPU jobs (calib rays, lidar-low) and the mini low/high re-renders still queued.
- Step 1 code: scripts/trust_masks.py — lift each drone pixel with the splat depth, project into the 6 real cameras of that keyframe, visible = in image, in front, < 60 m, not behind a nearer lidar surface (per-camera lidar depth map at 1/4 res, 5x5 min-spread); weight by cos(angle between drone and real view) x exp(-dist/60); trust = opacity x sqrt(best). Written as key alpha. Queued after the mini re-renders (3750396 low, 3750397 high).

## 2026-11-25 06:50 — camera-only, and a phone-photo goal (user, side chat)
- The model stays camera-only (no lidar input at run time). Experiment A (calibration rays) stays.
- New goal: work from just 6 phone photos taken in a street (front, front-left, front-right, back-left, back, back-right) and draw the drone view.
  - When no calibration is given, assume the standard nuScenes rig directions and FOV (mean rays over all scenes).
  - Training robustness so phone photos are in distribution: random crop/scale (FOV ±15%), centre-crop to 16:9, colour/exposure jitter, small per-camera yaw jitter, JPEG/blur. Ray inputs must follow the same crop/scale/yaw so geometry stays consistent.
  - Deliver scripts/try_my_photos.py --front a.jpg --front_left b.jpg ... --out drone.png (cluster GPU, or CPU/MPS on the Mac for a small model) and a README how-to for taking the 6 photos.

## 2026-11-25 06:40
- Public repo created (user request): https://github.com/TarunTomar122/mini-reality-view — README in the style of the user's spiking-vsr write-up, figures in docs/figs, explainer moved to docs/explainer.html, Slurm log paths made relative (no usernames in the repo). Kept updated as results come in.
- Experiment A code: scripts/calib.py (per-scene K + cam->ego -> Plücker rays per pooled patch, data/calib_rays.pt, job 3750368), Renderer(rays=True) adds an MLP ray embedding in context(), train.py --rays. SD1.5 UNet/VAE downloaded; Difix downloading.

## 2026-11-25 06:30 — experiment A approved (user, side chat)
A) Camera calibration as input (camera-only, Tesla-like): add a per-patch ray embedding (ego-frame direction from K + cam->ego, plus camera origin; Plücker or dir+origin -> MLP) to the DINOv2 tokens in Renderer.context(); store K and cam->ego (or the 16x9 ray grid) at encode time. Compare with vs without rays.
Placement: finish runs -> 2 (lower drone) -> 1 (trust masks) -> A (quick ablation on the from-scratch DiT) -> 3 (SD1.5 with the best camera-only inputs, incl. rays) -> distil -> final eval -> write-up. Rays go into the SD1.5 camera adapter too. Scheduled task prompt updated.

## 2026-11-25 06:20
- Queue busy: lidar-low, mini splat re-renders (low/high), distillation, step baselines all pending. Trainval splats 39/70; full-train-c 49k/80k.
- Step 3 research: Difix is public (code nv-tlabs/difix3d, weights HF nvidia/difix, research/non-commercial licence; SD-Turbo based, single step). SD 1.5 weights via HF mirror sd-legacy/stable-diffusion-v1-5 (CreativeML OpenRAIL-M). Full fine-tune of the 860M UNet won't fit 11 GB (AdamW states ~14 GB) -> plan: frozen SD1.5 UNet + LoRA + a trainable camera-token adapter replacing the CLIP text tokens; 512x256 (latent 32x64, divisible by 8).
  Downloading SD1.5 UNet/VAE, Difix, sd-turbo licence + installing peft on the login node (logs/dl_sd.log). scripts/encode_latents.py: 512x256 latents of any key set (crop top 32 rows; alpha = weight).

## 2026-11-25 06:10
- Step 2 (lower drone) started: mrv/nusc.py CHASES {"high": 13 m back / 8.5 m up / look 12 m ahead, "low": 9 m back / 4.5 m up / look 15 m ahead at 0.5 m}; make_targets.py and ds_render_chase.py take --chase. ds_render_chase now also saves depth + camera (…_depth.npz) for trust masks.
  slurm/ds_render.sh re-renders trained splats with no retraining. Jobs: mini splats low (3750359) and high-with-depth (new id) -> data/targets_splat_mini_{low,high}; lidar keys low on mini (3750360) -> data/targets_lidar_low.

## 2026-11-25 06:00 — NEW PLAN (user, via side chat; project is now multi-day)
Why: even splat keys aren't clean (floaters, edge smears); the model can only chase its answer key. Finish the running work (full_c video, 70 trainval splats, distillation + step baselines, style-mix run), then:
1) Trust masks: per-pixel confidence for splat keys (opacity x seen-by-a-real-camera via depth reprojection into the 6 cameras at nearby frames / depth consistency), stored as target alpha so the loss weights it; low weight elsewhere. Compare with vs without on held-out scenes.
2) Lower drone: second chase camera ~4-5 m up, ~8-10 m back (configurable in mrv/nusc.py, ds_render_chase.py, make_targets.py); re-render splat + lidar keys from existing splat checkpoints; compare keys and models at both heights; pick one or train on both with a pose input.
3) Pretrained generator: fine-tune an open pretrained diffusion model (start: SD1.5 UNet with camera-token cross-attention, same SD VAE latents; also SD-Turbo/SDXS, PixArt/Sana; Difix/Difix3D+ to clean splat keys; Cosmos/Vista/OpenDWM as possible teachers). Check weights, licence, GPU fit. Then distil to 1-4 steps.
Order: finish runs -> 2 -> 1 -> 3 -> distil -> final eval -> README + plain HTML results page. Scheduled 30-min task prompt updated with this plan.

## 2026-11-25 05:45
- Trainval splats 28/70; full-train-c 43.8k/80k (loss 0.45). Distillation + step baselines still queued (GPU queue busy); mini scene 9 waiting for the A6000.

## 2026-11-25 05:20
- Trainval splats 19/70 rendered, 19 running, 32 queued, no failures. full-train-c 38.4k/80k.
- Stage 7: scripts/distill.py — 1-step student copies full_b's 4-step output from the same noise (latent MSE), 15k steps (3750324) -> eval at 1 step (3750327).
  Baselines for full_b at 1 and 2 steps on 30 val scenes (3750325/6); 4 steps already measured (LPIPS 0.374).

## 2026-11-25 04:50
- Trainval splats: 2/70 rendered, 19 running (expect ~4 h for all). Mini splats 9/10 (scene 9 waiting for the A6000). full-train-c 33k/80k.
- Prepared stage-6 training: Renderer(n_styles=2) adds a learned "target style" embedding (0 = lidar target, 1 = splat target);
  train.py --enc2 <splat enc> --mix 0.5 --init <ckpt> trains on both sets (lidar teaches layout from ~700 scenes, splat teaches complete appearance from ~57), renders with --render_style 1.
  evaluate.py --truth_enc <splat enc> --style 1 scores against splat targets. Smoke-tested on CPU (old checkpoints load with only style.weight missing).

## 2026-11-25 04:15
- Stage 8 eval, 30 val scenes, 4 steps:
  | run | data | model | masked L1 | LPIPS |
  | mean image | - | - | 30.6 | 0.655 |
  | full_a | 340 scenes | 45M, 40k | 28.1 | 0.412 |
  | full_b | 700 train scenes | 110M (512x16), 80k | 27.8 | 0.374 |
  Pictures (pulled/full_b_sheet.jpg): sharper and more detailed than A; road shape follows the truth (curves match). It also copies the lidar-ring stripes and holes of the targets -> splat targets should fix that.
- Sky masks for 70 trainval scenes done; 19 splat jobs running, 51 queued. full-train-c (video) 27.8k/80k. Mini splat scene 9 queued for the A6000.

## 2026-11-25 03:45
- full-train-b DONE (512x16, 80k steps, final loss ~0.24). In-training val render (25 steps) masked L1: 0003 30.6, 0012 26.4, 0013 27.2, 0014 29.5, 0015 28.4, 0016 29.6 (~same as run A on L1). Proper eval (30 val scenes, LPIPS) job 3750251.
- full-train-c (video) at ~20k/80k.
- Trainval per-scene preprocessing: all 64 finished; sky masks 22 done / 13 running / 35 queued; 70 splat jobs queued behind them.
- Mini splats 9/10: scene 9 (scene-1100) had OOM'd on an 11 GB 2080 Ti (4.5 GB reserved-unallocated, fragmentation) -> rerun on A6000 with expandable_segments (3750252).

## 2026-11-25 03:15
- Cleaned up: the 70 old ds-tv jobs (depending on the cancelled bulk sky jobs, DependencyNeverSatisfied) were still queued -> cancelled. Now 70 per-scene chains (18 preprocessing running, 14 queued).
- full-train-b 67.2k/80k (loss 0.24) -> done in ~20 min; full-train-c 17k/80k (loss 0.71). Mini splats 9/10.

## 2026-11-25 02:50
- full-train-b step 46.6k/80k (loss 0.32); full-train-c (video) 11.8k (loss 0.76). Splat mini: 8/10 done, scene 7 running fine after the clamp fix.
- Trainval drivestudio preprocessing was thrashing: 2 jobs x 8 workers each holding the full nuScenes index (MaxRSS 67 GB at a 64 GB limit, 32 min CPU in 2 h), only ~6/70 scenes complete after 2 h.
  -> cancelled; new slurm/ds_tv_chain.sh submits per scene: preprocess (1 worker, 2 CPU, 24 GB) -> sky masks -> splat 15k -> chase render. Scenes 006/032/034/042/067/069 were complete and skip preprocessing.

## 2026-11-25 02:15
- Scene 7 fix worked: no NaN, 15k/30k iters.
- full-train-b step 25.8k/80k after 37 min (fast GPU), loss 0.52; full-train-c (video) step 6.4k (slower slice), loss 0.82.
- Trainval drivestudio preprocessing running (8 scenes in parallel per job, ~600-1200 images written per scene so far); sky masks + 70 splat jobs queued behind it.

## 2026-11-25 01:50
- Encode DONE: 850/850 scenes in data/enc_full. full-train-b (512x16 single-frame) and full-train-c (512x16 video --prev) running: ~6 min/1k steps -> 80k in ~8 h (ETA ~09:45).
- Splat targets: 8/10 mini scenes done (scene 4 finished). Scene 7: rescue 2 (sharp_shape_reg w=0) also NaN — the NaN is 0/0 in the scale ratio: the car is parked, the lidar has exact duplicate points, kNN distance 0 -> log(0) = -inf initial scale. Rescue 3 (root cause): patched drivestudio models/gaussians/vanilla.py + models/nodes/rigid.py to clamp the initial kNN distance at 1e-4 m; rerun 3749884.
- Queued trainval splat pipeline: sky masks x4 (3749885-88, after preprocessing) -> 70 scenes (idx 0-69) at 15k iters on 2080 Ti, project tv -> data/targets_splat_tv.

## 2026-11-25 01:15
- Splat scene 7 (night, scene-1077): rescue 1 (seed=1) also NaN at step 0 -> rescue 2: sharp_shape_reg off for Background + RigidNodes (3749868).
- Encode 762/850; trainval drivestudio preprocessing (2 jobs) running; splat scene 4 still training.

## 2026-11-25 00:55
- Stage 8 eval of full-train-a (30 val scenes, 4 steps): masked L1 28.1 (mean-image baseline 30.6), LPIPS 0.41 (baseline 0.65). No drift for a single-frame model (first-5 vs last-5 frames 29.7 / 29.7).
- Stage 6: 30k-iter Street-Gaussian targets for 7 mini scenes are far better than lidar targets (pulled/splat_vs_lidar.jpg): complete scenes, parked cars, excavator, lane paint, facades, no holes; streaks at the frame edges, night scene noisier.
  scene 4 still training; scene 7 (scene-1077, night) NaN at step 0 in Background sharp_shape_reg -> rescue 1: seed=1 (3749832). ds_scene.sh now takes extra overrides.
- Splats cost ~2 GPU-h/scene at 30k, so not for all 850. Plan: splat ~70 trainval scenes (idx 0-69, incl. some val scenes) at fewer iterations, then fine-tune the full lidar-trained model on splat targets. drivestudio trainval preprocessing jobs 3749841/3749842 (idx 0-34, 35-69). ds_scene.sh takes DSROOT/NUSC_ROOT/NUSC_VER env vars for trainval.
- Encode 585/850; full-train-b/c queued after it.

## 2026-11-25 00:20
- Targets 850/850 done. Encode round 2 running (400/850 encoded). full-train-b/c queued after it.
- 8 splat scenes training (~1 h in).
- Stage 8 script: scripts/evaluate.py (val scenes; masked L1, LPIPS on covered pixels, error per frame for drift, mean-image baseline; video models roll out the whole scene). LPIPS weights cached. eval-full-a job 3749819 (30 val scenes, 4 steps).

## 2026-11-24 23:45
- Targets 844/850 (7/8 shards done); encode round 2 waits on the last shard; full-train-b/c wait on encode.
- 9/10 splat scenes training (30k iters), 1 done or queued; no real errors in logs.

## 2026-11-24 23:15
- full-train-a DONE (340 scenes, 45M, 40k steps, final loss ~0.3). Held-out val scenes masked L1: 0003 28.7, 0012 26.0, 0013 27.3, 0014 28.0, 0015 28.5, 0016 29.3.
  Pictures (pulled/full_a_sheet.jpg): real road structure on unseen streets (lane lines, kerbs, trees roughly right), still blurry. Big step up from mini (which drew a generic blur on unseen scenes).
- Stage 6 test (scene 0, 2k iters, ~6 min train + ~18 min load/render on 2080 Ti): renders work and show things the lidar targets miss (trucks, excavator), but with colourful floaters (sky, near camera) — expected at 2k iters and with a viewpoint 7 m above the training cameras (pulled/splat_test_sheet.jpg).
  Launched all 10 mini scenes at 30k iters (3749741-50).
- Targets 757/850 scenes. Queued after encode round 2: full-train-b (512x16, ~110M, 80k steps) and full-train-c (same + --prev video) -> runs/full_b, runs/full_c.

## 2026-11-24 22:45
- Download COMPLETE (21/21). Targets: 570 scenes so far (round 2 running); encode round 2 pending on them.
- full-train-a at 38k/40k (loss ~0.29); rendering of val scenes next.
- Stage 6: sky masks done (mini, 10 scenes); ds-test0 (2k iters, scene 0, 2080 Ti) started.

## 2026-11-24 22:15
- full-train-a step 28.4k/40k (loss 0.34), ETA ~22:50 + render.
- Download 20/21 (only trainval10 camera left, streaming). full_round.sh round 2 submitted for parts 04-09 (targets skip the 340 done scenes; encode skips existing).
- Stage 6: drivestudio preprocessing of mini DONE (images, lidar, calib, masks, objects, 10 Hz). Sky masks running (~4-5 min/scene, 1206 images each). Then the 2k-iter test on scene 0.

## 2026-11-24 21:50
- full-train-a step 18.4k/40k (loss 0.46), ETA ~23:15. Download 18/21. Kerberos renewed (valid to 07:37).
- Stage 6 code: scripts/ds_render_chase.py (loads a drivestudio checkpoint, builds our chase camera from the CAM_FRONT pose + nuScenes cam->ego calibration, renders RGB + opacity at every keyframe -> data/targets_splat_<proj>/<scene>/<k>_<token>.png); slurm/ds_scene.sh (train streetgs.yaml on one mini scene, load_smpl False, then render).
- Chain: preprocessing 3749415 (was stuck on QOSMaxCpuPerJobLimit with 16 CPUs -> scontrol to 8) -> sky masks 3749506 -> 2k-iteration splat test on scene 0 (3749507, 2080 Ti) before launching all 10 scenes at 30k.

## 2026-11-24 21:15
- full-train-a: step 8.6k/40k after 27 min (loss ~0.8, not memorising like mini's 0.03); ETA ~23:15.
- Download 15/21.
- drivestudio env DONE (DS_ENV_OK, torch 2.1.2 cu118, gsplat 1.3.0). nvdiffrast only built for sm 7.5/8.6 (sm_90 ptxas crash) -> splat jobs must request --gres=gpu:nvidia_geforce_rtx_2080_ti:1 or gpu:nvidia_rtx_a6000:1, not H200 slices.
- drivestudio mini preprocessing 3749415 still queued (16-CPU job).

## 2026-11-24 20:50
- full-train-a running (since 20:41, H200 MIG slice, bf16) on 340 encoded scenes.
- Download 12/21.
- drivestudio env: gsplat 1.3.0 + pytorch3d built with conda gcc 11. nvdiffrast: ptxas segfault with 8 parallel jobs -> slurm/ds_finish.sh (MAX_JOBS=1, falls back to sm 7.5/8.6 only) job 3749414. drivestudio imports nvdiffrast in models/modules.py, so it's needed.
- drivestudio preprocessing of mini (10 Hz, interpolate_N 4) job 3749415 (slurm/ds_preprocess.sh). Plan: sky masks via scripts/sky_masks.py, then train configs/paper_legacy/streetgs.yaml with dataset=nuscenes/6cams, load_smpl=False, one job per scene; then write a chase-cam renderer over the trained splats.

## 2026-11-24 20:15
- Download 10/21 archives (parts 01-05 partly); targets done for 340 scenes; encode 4 shards running (144/340 after 23 min).
- drivestudio env: gsplat compile failed (system gcc 13 > CUDA 11.8 limit 11) -> conda-forge gcc/gxx 11 in the env, CC/CXX set; rebuild job 3749366.
- Queued full-train-a (3749367, after encode): data/enc_full (pooled tokens), official val held out, 45M DiT, 40k steps, bs 64 -> runs/full_a. First scaling point vs mini.

## 2026-11-24 19:45
- Stage 7 baseline (mini stage-4 model, H200 MIG 1g slice, batch 1 incl. VAE decode), masked L1 / fps:
  train scene-0061: 1 step 15.9 | 2 steps 1.5 / 20 fps | 4: 1.5 / 11 | 8: 2.0 / 5.8 | 16: 2.8 / 2.9 | 32: 3.4 / 1.5
  held-out scene-1077: ~20 at every step count (1 step 19.5 / 33 fps).
  -> 2 steps already matches or beats 32 here; ~20 fps at 2 steps before any distillation.
- Full targets: 8 shards running ~45 s/scene, ~290 scenes done (parts 01-03), coverage 0.5-0.83. Encode shards wait on them.
- Download 7/21 archives done (parts 01-03 + meta); 04/05 streaming.
- drivestudio env build still installing pip deps (30 min in).
- Next: when encode lands, train on what's there (val held out) for a first scaling point; full train when download completes.

## 2026-11-24 19:15
- Download: meta + parts 01, 02 (camera + lidar) done (~170 scenes); 03/04 streaming.
- full_round.sh submitted: 8 target shards (3749275-82) -> 4 encode shards (3749283-86, after targets). Rerun after more parts land.
- Fixes: node /tmp full (grep write error in env build; also earlier /tmp failure) -> TMPDIR=~/mini-reality-view/tmp in py.sbatch + setup_drivestudio.sh; env build resubmitted 3749273.
  steps_speed OOM (decoded a whole scene's GT at once on a shared 16 GB slice) -> chunked; resubmitted 3749274.
- Queue is busy (other project's jobs), all ours pending on Priority.

## 2026-11-24 18:46
- Download: no part finished since 18:23 (3 parallel streams still running). drivestudio env job 3749202 running (conda done, pip/compiles next).
- GPU was idle -> stage 7 baseline submitted: scripts/steps_speed.py on the mini stage-4 model (steps 1..32, masked L1 + fps at batch 1) job 3749229.

## 2026-11-24 18:40
- Training-scene render check: masked L1 3.2 (scene-0061) / 3.1 (scene-0916) vs 20-56 on held-out -> model uses the cameras; it just needs more data.
- slurm/full_round.sh: 8 CPU target shards + 4 GPU encode shards (pool 2) on data/nuscenes_full -> data/targets_full, data/enc_full. Run it when download parts land (idempotent).
- Stage 6 prep: drivestudio env build (slurm/setup_drivestudio.sh, conda-forge + nvidia channels only; torch 2.1.2 cu118, gsplat 1.3.0, pytorch3d, nvdiffrast) job 3749202. First try failed on Anaconda ToS (not accepted on the user's behalf).
  scripts/sky_masks.py replaces mmcv SegFormer with transformers SegFormer-B5 Cityscapes (weights cached in hf/).
- Queue: another project's ~12 GPU jobs are queued/running on Teaching, so ours may wait.
- Download: 3 parallel streams since 18:23, part01 camera ~6.7k front images in.

## 2026-11-24 18:30
- Stages 4+5 on mini (8 train scenes / 322 frames, 15k steps, 45M DiT, ~25 min each on an H200 MIG slice): train loss 1.8 -> 0.03 (memorised).
  Held-out masked L1 (0-255): scene-1077 20.4 (s4) / 19.8 (s5); scene-0553 53.4 / 56.3. Pictures (pulled/stage45_sheet.jpg): 1077 gets the rough layout (dark road, lit centre), 0553 a generic grey blur. Both test scenes are night scenes. Expected with 8 scenes; fix = full data.
- Sanity check submitted: render training scenes 0061/0916 (job 3749183) to confirm the model uses the cameras.
- Full download switched to streaming curl|tar, 3 parts in parallel (unpacking was the bottleneck). Done so far: meta, trainval01 lidar.
- Code for full data: make_targets skips undownloaded/finished scenes; encode --pool 2 stores 16x9 tokens (0.66 MB/frame); train --test_scenes val uses the official val split.

## 2026-11-24 18:00
- Stage 1 done: runs/stage1/cameras.jpg + bev.jpg look right (6 cams, lidar, camera cones).
- Stage 2 v1 (±3 keyframes, 0.08 m points): sparse, roughly 40% covered. v2 (static points from the whole scene, moving objects from current frame only, 0.12 m points): coverage 0.49-0.82 per scene (night/parked scene-0553 only 0.18). v2 is clearly better (pulled/compare_v1_v2.jpg) -> make_targets defaults now K=100, radius 0.12.
- Cancelled v1 training; re-encode v2 -> data/enc_v2 (job 3749135), then stage4 3749136 + stage5 3749137 train on it.
- Full download: meta done, trainval01 lidar in progress.

## 2026-11-24 17:45
- Env OK (torch 2.15 cu132 got pulled in by pip; works). Account = Teaching partition only; GPU jobs land on 2080 Ti / A6000 / H200 MIG slices. train.py uses fp16+GradScaler when no bf16.
- GPU check: H200 MIG 1g.18gb, CUDA OK. stage1_look first run crashed (np.Inf in old matplotlib) -> patched, rerun 3749120.
- Stage 2 targets job 3749113 running (~40 s/scene). Chained: encode 3749121 -> train stage4 3749122 + stage5 (--prev) 3749123, 15k steps each.
- Full trainval download started on login node (slurm/download_full.sh, logs/download-full.log, resumable via data/full_done.txt). No disk quota on home.

## 2026-11-24 ~17:15
- Code written: mrv/nusc.py, mrv/model.py (45M DiT), scripts/stage1_look.py, make_targets.py, encode.py, train.py; slurm/setup_env.sh, py.sbatch.
- Cluster ~/mini-reality-view: nuScenes-mini downloading/unpacking to data/nuscenes; .venv installing (logs/setup_env.log).
- Next: stage 1 look (runs/stage1), stage 2 targets on mini (data/targets_lidar), stage 3 encode (GPU), stage 4 train (test scenes scene-0553, scene-1077), stage 5 --prev.
