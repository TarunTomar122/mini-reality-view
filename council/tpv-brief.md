# Council brief: real cameras in, game-style third-person view out

## Context
Mini Reality View (read README.md, PROGRESS.md top entries, mrv/sd.py, scripts/train_sd.py, scripts/sim_render.py,
council/PLAN-sim.md for detail). Goal: 6 nuScenes camera photos -> a picture of the car from behind (third-person
chase camera, own car visible, like a racing game). CAMERA-ONLY at run time.

What we learned so far:
- Our main renderer (Stable Diffusion 1.5 + LoRA r=64 on the UNet + a "CameraAdapter": DINOv2-small tokens per
  camera (6 x 144 x 384) + per-token viewing rays + camera embedding + style token -> 2-layer transformer -> 865
  context tokens replacing the CLIP prompt; DDPM epsilon loss) was trained 40k steps (bs 16) on REAL answer keys made
  from lidar points and Gaussian-splat reconstructions rendered from a drone camera. The keys are messy (gaps,
  floaters, smears) and the model copies that mess. Car placement plateaued (judge car F1 ~0.29).
- New idea from the owner: keep real photos as input, but make the answer key a GAME render of the same moment.
  We replay each nuScenes drive in the MetaDrive simulator (ScenarioNet): road layout from the HD map, every labelled
  vehicle / pedestrian at its true pose, own car shown, third-person camera ~7.5 m behind, ~2.4 m up, 60 deg HFOV,
  512x288 (top 32 rows cropped -> 512x256 -> SD latents 4x32x64). The key is exact and clean, but game-looking:
  generic car models, grass instead of buildings, only labelled objects.
- Data: ~7,900 pairs from ~508 training drives (every 2nd keyframe at 2 Hz). Real tokens also exist for
  phone-style augmented copies of every training frame. Could cheaply grow: every keyframe (~2x), all 700 train
  drives (~1.4x), or non-keyframes (cameras ~12 Hz, sim 10 Hz; positions between labels interpolated) (~10x).
- In an earlier pilot, training on sim keys for 2k steps then switching to real keys gave a head start; style
  changes looked quick to learn, placement slow.
- Compute: SD training ~70 steps/min at bs 16 on an H200 1g.18gb slice; a few slices; 2080 Ti / A6000 for rendering.

## Questions
1. Initialisation: (A) plain SD 1.5 + fresh adapter, (B) our trained renderer as-is (adapter + LoRA), (C) our trained
   camera adapter + a fresh LoRA, or something else (e.g. a different base model better suited to clean synthetic
   images, SD 2.1 / SDXL-turbo / a pixel-space or smaller model, or a non-diffusion decoder since the target is a
   near-deterministic render). What does the literature (sim-to-real / real-to-sim translation, BEV/layout-to-image
   like MagicDrive, BEVControl, DriveDreamer, ControlNet-style conditioning, feed-forward novel-view models) suggest?
2. Is ~7,900 pairs enough? How would you test that (learning curve)? Which data growth option is worth it first?
3. Evaluation: since the key is exact, what should we measure? (pixel L1 / LPIPS vs the game key; car placement F1
   with our Mask2Former judge; we could also render exact game segmentation / instance masks for each key; held-out
   val drives rendered the same way.)
4. Propose exactly 3 experiments we can run in parallel within ~3 GPU-hours each (H200 1g.18gb slices), with what each
   tests, settings (init, steps, batch, which data), and a decision rule for picking the setup for the full run.

Keep it practical, cite sources with URLs where you rely on them, and be honest about uncertainty.
