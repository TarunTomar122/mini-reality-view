# Council brief: temporal consistency and memory for the chase-view renderer

## Where we are (read README.md, PROGRESS.md newest first, council/PLAN-tpv.md, mrv/sd.py, scripts/train_sd.py,
## scripts/sim_render.py, scripts/video_render.py, scripts/judge_pair.py)
- Goal: six real nuScenes camera photos in -> a game-style third-person chase view (own car visible, camera 7.5 m
  behind the car centre, 2.4 m up) out. CAMERA-ONLY at run time.
- Renderer: Stable Diffusion 1.5 (frozen) + LoRA r=64 + CameraAdapter (DINOv2-small tokens 6x144x384 per frame +
  per-token viewing rays + camera embedding + style token -> 865 context tokens replacing the CLIP prompt); epsilon
  loss, DDIM 25, CFG 2; 512x256 output (SD latents 4x32x64).
- Answer keys: MetaDrive replays of each nuScenes drive (map geometry, every labelled object at its pose) rendered from
  the chase camera: exact, clean, game-looking. 12,111 keyframe pairs now (9,587 train / 2,524 held-out val).
- Running now (council/PLAN-tpv.md): E1 init from our earlier real-key renderer, E2 plain SD, E3 E1 on a quarter of
  the drives, 10k steps each. At 2.5k steps E1/E3 already match the game keys closely on held-out drives; E2 invents
  cones. The winner of E1/E2/E3 sets the starting point for this experiment.
- Each frame is drawn independently. The owner watched a 10 s clip (real camera sweeps at ~12 Hz through E1 at 2.5k,
  same fixed noise every frame): mostly stable, but cars at the edges change model or shape between frames, and
  nothing is remembered once it leaves the cameras' view.
- Earlier evidence: our from-scratch DiT video variant (previous latent + noise s concatenated on channels) drifted
  (pixel error 27.4 -> 35.2 over 20 s at s=0.1); s=0.3 helped; training half the time on its OWN previous frames
  ("self_prev") cut drift to +1.8 L1 over 20 s and gave the best LPIPS. Tesla's description of Reality View: the
  previous Reality View frame plus noise is denoised into the next one.
- Being tried now as a free baseline: SDEdit-style warm start at inference (previous output latent re-noised to
  strength 0.5 / 0.7, same fixed noise), no retraining, on the same two clips.

## Data available
- Real cameras: nuScenes sweeps at ~12 Hz per camera (non-keyframes have images but no labels); keyframes at 2 Hz.
- Simulator: replays run at 10 Hz; object poses between 2 Hz labels are interpolated by ScenarioNet. We can render
  chase-view keys at any sim tick (sim_render.py, ~1-2 min per drive on a 2080 Ti; two cluster nodes hang MetaDrive
  and are excluded). Camera and sim ticks differ by up to ~40 ms.
- Compute: H200 1g.18gb slices (~70 SD training steps/min at bs 16), A6000 / 2080 Ti for rendering and scoring.
  BEVFormer-base (temporal BEV, train-only checkpoint) is installed in a separate env (council/PLAN-bev.md); its
  features are not cached yet.

## Options to evaluate (not exhaustive; propose others if better)
1. No-retrain trick: warm start / SDEdit from the previous output plus noise at inference.
2. Tesla-style previous-frame conditioning: previous output latent (+ noise) as extra conv_in channels on SD 1.5
   (zero-initialised), trained partly on the model's own previous frames (self-rollout / self_prev).
3. Temporal input memory: past camera tokens (k frames, with ego-motion), or BEVFormer's temporal BEV.
4. Video diffusion / temporal attention modules (AnimateDiff-style motion modules, multi-frame denoising).

## What we need from you
1. Recommendation(s) with reasoning and sources (check that cited papers/repos exist): which options, in what order,
   how they fit SD 1.5 + our adapter, and what it costs.
2. Metrics (define them precisely, computable with what we have):
   - a flicker / temporal stability score (e.g. warped-frame difference using the game keys' own motion, or per-object
     track consistency against the game keys over a sequence),
   - drift over 20 s,
   - placement accuracy must not get worse (we already score car F1 / road IoU vs exact game keys per frame).
3. Data plan: which sequences to render (rate, length, train/val), how to pair real camera sweeps with sim ticks.
4. An experiment plan: 2-4 arms that fit ~3 GPU-hours each on an H200 1g.18gb slice, with settings, the eval, and
   decision rules (what result would make us adopt each option).
5. Risks, and what would make you call it off.
Keep it practical and honest about uncertainty; under ~1500 words.
