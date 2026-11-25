# Mini Reality View: a drone's-eye view of the car, drawn by a diffusion model from its six cameras

We are rebuilding, at home and in the open, the idea behind Tesla's FSD v15 "Reality View": a live picture of the car
and its surroundings seen from a virtual camera floating behind and above it. Six ordinary car cameras go in; a small
image generator draws the chase view. No game engine, no 3D car models.

**Status: work in progress.** This page is updated as results come in. Lab notes with every number and job: `PROGRESS.md`.

## The problem

A car's cameras sit about 1.5 m off the ground and look outwards. Nobody films the car from 8 m above and behind, so
there is no "correct" drone picture to learn from. To train a model to draw that view, we first have to make the
answer key ourselves, and the model can only ever be as good as that key.

## The idea

Tesla's description (Elon Musk on X, 22 Nov 2026): the same vision features that drive the car are fed to a separate
~1B-parameter diffusion model, which takes the previous Reality View frame plus a little noise and denoises it into
the next one. We copy that shape at a small scale:

```
6 car photos ─► DINOv2 (frozen) ─► camera tokens ─┐
                                                   ├─► diffusion renderer ─► drone-view frame
previous frame + a little noise ───────────────────┘          │
                         ▲                                     │
                         └─────────── fed back ────────────────┘
```

- **Camera tokens:** each photo goes through a frozen DINOv2-small; every 14×14 patch becomes a list of 384 numbers.
- **Renderer:** a diffusion transformer (DiT) trained with flow matching, working in Stable Diffusion's compressed
  image space (a 256×144 frame becomes 4×18×32 numbers).
- **Video:** the previous output, with noise added, is a second input, so frames stay consistent and small mistakes
  don't snowball.

[`docs/explainer.html`](docs/explainer.html) walks through every stage with pictures (open it in a browser).

## Making the answer key

We tried three kinds of key, each better than the last.

**1. Lidar points, one moment.** Colour each lidar point with the camera pixel it lands on, then look at the coloured
points from the virtual drone (13 m behind, 8.5 m up). About 40% of the picture gets covered; the rest is holes.

**2. Lidar points, whole scene.** Things that don't move are the same in every frame, so we merge lidar from the whole
20-second scene for them, and use only the current frame for cars and people (from nuScenes' 3D boxes). Coverage goes
up to 50–80% of the picture.

![Single-moment keys (left) vs whole-scene keys (right)](docs/figs/compare_v1_v2.jpg)

**3. Gaussian splats.** Rebuild each scene as a 3D Gaussian-splat model (Street Gaussians in
[drivestudio](https://github.com/ziyc/drivestudio)), fitted to all six cameras, then render it from the drone camera.
The keys become complete pictures: parked cars, an excavator, lane paint and shop fronts that the lidar keys miss.
They cost about 1.5–2 GPU-hours per scene, so we can only afford them for some scenes.

![Lidar keys (left) vs splat keys (right)](docs/figs/splat_vs_lidar.jpg)

## What we got so far

Data: nuScenes (850 driving scenes of 20 s, Boston and Singapore, 6 cameras, keyframes at 2 Hz). Every number below is
on the official **val** scenes, which the models never saw. Masked L1 is the average pixel error (0–255) where the key
has data; LPIPS measures how different two pictures look to a neural network (lower is better). The "mean image"
baseline is the average answer key: what you score if you ignore the cameras completely.

| Model | Training scenes | Size | Masked L1 | LPIPS |
|---|---|---|---|---|
| Mean image (ignores the cameras) | – | – | 30.6 | 0.655 |
| Small renderer | 290 | 45M | 28.1 | 0.412 |
| **Bigger renderer** | 700 | 110M | **27.8** | **0.374** |

*30 val scenes, 4 sampling steps, lidar keys.*

![Unseen val scenes: answer key (left) and model (right), three frames each](docs/figs/full_b_sheet.jpg)

- **It reads the cameras.** On streets it has never seen, the model draws the right kind of road: curves where the road
  curves, kerbs, trees and buildings roughly where they are. LPIPS drops from 0.66 (ignoring the cameras) to 0.37.
- **It is blurry and copies the key's flaws.** It draws the lidar's ring-shaped stripes and holes, because that is what
  the key looks like. That is the main reason for the splat keys and the plan below.
- **Two steps are enough.** With 2 denoising steps instead of 32 the picture is just as good, and a small model runs
  at about 20 frames per second on one seventh of an H200 (including turning the result back into pixels).

## What we learned

- **Data matters most at first.** Trained on 8 scenes, the model memorises them perfectly (pixel error 3) and draws a
  grey blur on anything new (error 20–56). With 290 and then 700 scenes it starts drawing real streets.
- **The pixel score hides progress.** Blur scores well on L1, so the bigger model's L1 barely moves (28.1 → 27.8)
  while its pictures are visibly sharper; LPIPS (0.41 → 0.37) tracks what you see.
- **Splat keys break when the camera is too high.** The real cameras are 1.5 m up; at 8.5 m the splat scene shows
  floaters and smears where no camera ever looked.

## Problems we hit (and fixed)

- **Parked car, night scene, crash at step 0.** With the car standing still, the lidar contains exact duplicate
  points, so the starting size of some splats was log(0). A new seed and switching off the regulariser didn't help;
  clamping the starting distance to 0.1 mm did.
- **Splat preprocessing thrashed.** Eight workers, each holding all of nuScenes' metadata, hit the memory limit and
  did 6 scenes in 2 hours. One small job per scene fixed it.

## Honest limits

- **The answer keys are not ground truth.** They are lidar points or 3D reconstructions, each with their own errors.
  Our scores say how close the model gets to the key, not to reality.
- **2 Hz.** nuScenes labels and lidar keyframes come at 2 Hz, so our "video" is 2 frames per second for now.
- **Small and slow compared to Tesla.** Our renderer is 45–110M parameters (Tesla's is about 1B, plus a 3B driving
  model), and our training data is about 4 hours of driving (700 scenes of 20 s).
- **Speed numbers are on a data-centre GPU**, not a car computer.

## What's next

1. **Lower drone camera** (9 m back, 4.5 m up) so the splat keys are cleaner, and compare models at both heights.
2. **Trust masks:** grade the model only where the key is reliable (splat seen by a real camera, depth agrees).
3. **Camera geometry as input:** give each camera token its viewing ray, so the model doesn't have to guess where each
   camera points.
4. **Lidar as input** (a non-Tesla variant, clearly labelled): how much better does it get with lidar at run time?
5. **Start from a pretrained image model** (Stable Diffusion 1.5 with LoRA and a camera-token adapter; Difix to clean
   splat keys) so the model already knows what streets look like.
6. **Distil to 1–4 steps**, then a final evaluation and video results.

## For researchers

- Data: nuScenes v1.0-trainval (700 train / 150 val, official split) and v1.0-mini, from the public AWS mirror.
- Inputs: 6 × 448×252 photos → DINOv2-small (frozen), patch tokens 2×2 average-pooled to 6 × 144 × 384.
- Keys: 512×288 RGBA (alpha = covered), chase camera 13 m behind / 8.5 m above the ego origin looking at a point 12 m
  ahead on the ground, 60° horizontal FOV. Lidar keys: static points from all keyframes of the scene, dynamic points
  (inside nuScenes boxes of movable classes, 15% margin) from the current keyframe only, coloured by the first camera
  that sees them, 0.12 m squares. Splat keys: drivestudio `streetgs.yaml`, 6 cameras, 10 Hz, 30k iterations (15k for
  trainval), no SMPL.
- Latents: SD VAE (`sd-vae-ft-mse`) at 256×144 → 4×18×32, scaled by 0.18215 and normalised by their std.
- Renderer: DiT with adaLN-Zero, patch 2, cross-attention to the camera tokens (+ learned per-camera embedding and
  2D sin-cos positions); 45M (384×12) and 110M (512×16). Flow matching (x_t = (1−t)·x + t·ε, predict ε − x,
  logit-normal t), AdamW 2e-4, batch 64, EMA 0.999, loss weighted by key coverage (0.05 where the key is empty).
  Video variant: previous latent with noise s ~ U(0, 0.3) concatenated on channels, pure noise 10% of the time.
- Evaluation: Euler sampling, masked L1 on decoded 256×144 frames, LPIPS (AlexNet) on coverage-masked frames.

## Code

`mrv/nusc.py` (poses, point colouring, chase cameras, point splatting) · `mrv/model.py` (renderer) ·
`scripts/make_targets.py` (lidar keys) · `scripts/encode.py` (DINOv2 tokens + VAE latents) · `scripts/train.py`
(training, video, style-mixed training) · `scripts/evaluate.py` (held-out evaluation) · `scripts/steps_speed.py`,
`scripts/distill.py` (speed) · `scripts/ds_render_chase.py`, `scripts/sky_masks.py` + `slurm/ds_*.sh` (splat keys with
drivestudio) · `slurm/` (cluster jobs). Lab notes: `PROGRESS.md`.

nuScenes is free for non-commercial use; this project is a non-commercial learning project.
