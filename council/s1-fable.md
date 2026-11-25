# Sim data for the drone-view renderer: findings and plan (seat: fable, round 1)

**Short answer.** Use the simulator for layout, not for looks, and keep it in the mix instead of only pretraining on it.
The best first use is one the brief does not list: ScenarioNet replays every *real* nuScenes scene, so we can render a
complete drone-view key of the same moment and pair it with the **real** six cameras ("twin keys"). That gives full-pixel
geometry supervision with no input gap. The owner's sim-cameras -> sim-drone pretraining gets a fair pilot next to it;
the literature says to expect "faster early, about equal at 40k". CARLA waits for a positive pilot.

## Findings

1. **Geometry transfers from sim, appearance does not, and small encoders transfer worst.** Stable Diffusion fine-tuned
   only on synthetic scenes predicts depth on real photos (Marigold, 2023, [2312.02145](https://arxiv.org/abs/2312.02145)).
   But Depth Anything V2 (2024, [2406.09414](https://arxiv.org/abs/2406.09414)) found that with synthetic-only training
   only DINOv2-Giant transfers; smaller DINOv2 models "suffer from severe generalization issues", and the fix was to
   route through real images. We use DINOv2-Small. Frozen DINOv2-Large trained on GTA5 does reach about 61-64 mIoU on
   real streets (Rein, 2024, [2312.04265](https://arxiv.org/abs/2312.04265)).
2. **The closest published attempts transfer only partly.** GCD (2024, [2405.14868](https://arxiv.org/abs/2405.14868))
   fine-tuned a video diffusion model on 1,533 synthetic driving scenes to draw an elevated view behind the car; on real
   driving it shows qualitative results only and "sometimes generalizes" (wrong truck depth, blurry people). ReCamMaster
   (2025, [2503.11647](https://arxiv.org/abs/2503.11647)) blames such failures on the narrow sim domain. 6Img-to-3D
   (2024, [2404.12378](https://arxiv.org/abs/2404.12378)) maps the six nuScenes-rig cameras to 3D in CARLA (PSNR 18.7 on
   a held-out town) and reports only first zero-shot attempts on nuScenes, hard because of rig, FOV and weather
   differences. Nobody has shown clean sim-only transfer for our task.
3. **Joint training with a domain label beats pretrain-then-finetune in the most careful study.** Wei et al. (2025,
   [2503.22634](https://arxiv.org/abs/2503.22634)): "methods that employ data mixing outperform methods that train on
   sim-and-real separately"; good models learn to tell sim from real; a one-hot domain input helps; sim gains plateau and
   real data sets the ceiling; results are sensitive to the ratio. Maddukuri et al. (2025,
   [2503.24361](https://arxiv.org/abs/2503.24361)): co-training gives +38% on average, but misaligned cameras and too
   much sim cost a lot. MVDream (2023, [2308.16512](https://arxiv.org/abs/2308.16512)) keeps 30% real image batches
   while learning multi-view from renders, to protect image quality.
4. **Pretraining mostly buys speed once real data is not scarce.** Sim-then-real wins when real data is tiny (Nowruzi et
   al., 2019, [1907.07061](https://arxiv.org/abs/1907.07061); Tremblay et al., 2018,
   [1804.06516](https://arxiv.org/abs/1804.06516): +2.1 AP, shrinking as real images are added). With more labels and
   strong augmentation the benefit fades and can turn negative (Zoph et al., 2020,
   [2006.06882](https://arxiv.org/abs/2006.06882)). We have 28k real frames and strong augmentation.
5. **Tell the model which look to draw, then ask for the clean one.** AnimateDiff (2023,
   [2307.04725](https://arxiv.org/abs/2307.04725)) trains a separate "domain" LoRA to absorb a dataset's look and drops it
   at inference; SDXL (2023, [2307.01952](https://arxiv.org/abs/2307.01952)) conditions on nuisance factors and sets them
   to the clean value. Our style token is the same idea and already switched looks in `mix_high`.
6. **Making sim images look real is not worth it for us now.** EPE (2021, [2105.04619](https://arxiv.org/abs/2105.04619))
   needs engine G-buffers and about a million training steps, and warns that when sim and real layouts differ the
   translator invents content ("trees in the sky"). CARLA2Real (2024, [2410.18238](https://arxiv.org/abs/2410.18238))
   packages it for CARLA but the content gap stays. SimGen (2024, [2406.09386](https://arxiv.org/abs/2406.09386)) uses
   exactly our stack (MetaDrive + ScenarioNet nuScenes replays, camera at the nuScenes pose) and takes only layout from
   it, because the simulator lacks background detail. We would also have to repaint seven views consistently.
7. **Generated or sim data helps as an addition, and gives us a metric.** MagicDrive (2023,
   [2310.02601](https://arxiv.org/abs/2310.02601)) adds generated images 1:1 to real: road IoU +5.3, vehicle +4.3.
   Panacea (2023, [2311.16813](https://arxiv.org/abs/2311.16813)): +2.3 NDS added to real, but generated-only reaches 77%
   of real. Both score geometry by running a real-trained perception model on generated images against ground-truth
   layout (real images: road IoU 72; MagicDrive: 61).

## Which simulator

| Option | What it gives our drone view | Verdict |
|---|---|---|
| MetaDrive procedural | Empty highway, no city content | Skip (narrow domain, finding 2) |
| **ScenarioNet nuScenes replay** | Real layouts and logged traffic at 10 Hz (frame 5k = keyframe k); map has drivable area, dividers, crossings, walkways; cars, pedestrians, cyclists, cones, barriers. No buildings or trees, flat. Gives twin keys for real cameras and sim->sim pairs | **Use first**, about 3 GPU-h to render |
| CARLA via the SEED4D generator (2024, [2412.00730](https://arxiv.org/abs/2412.00730); BSD-3, docker, nuScenes rig built in, custom outside cameras) | Buildings, trees, parked cars, pedestrians, weather seen from the drone; only 8 towns | Only after a positive pilot and a Vulkan check on a GPU node |
| Ready-made sets (SEED4D static 2,002 scenes with 100 outside views; ParallelDomain-4D; Bench2Drive; SimBEV) | None has our fixed drone pose; we have no target-pose input | Do not download |

## Plan

1. **Checks before any training (E0).** Overlay twin renders on lidar keys for 5 scenes; road edges and cars must agree
   within about 0.3 m. The converter stores the ego at the nuScenes ego-pose point without moving it to the car centre,
   and appears to swap the ego's length and width, so hide the ego in all seven shots. Also measure the DINOv2 token
   cosine between real and sim cameras of the same moment (phone augmentation gave 0.81), and feed sim cameras to
   `sd_run1` to see whether it already draws the right layout.
2. **Rig.** Use each scene's real calibration, not the test script's 70 degrees for all six: nuScenes intrinsics give
   about 65 degrees for five cameras and about 89 for the back one. Compute rays from the sim camera with the `calib.py` code.
3. **Twin keys (style 2).** All 700 train scenes, keyframes only (about 28k). Fixed plain look, no randomisation: random
   colours the cameras cannot explain only teach blur, as the noisy splat keys did. Trust mask from MetaDrive's semantic
   camera: weight 1 on road, markings, walkway and objects, 0.05 on terrain and sky, where reality has buildings.
4. **Sim->sim rows (style 2).** Same replays, six sim cameras; one clean copy and one with rig jitter (0.3 m, 5 degrees,
   FOV 15%) and exact rays, plus the photometric phone augmentation. Full-weight key.
5. **Training.** `nn.Embedding(4)` for styles. Sim rows are 20% of a batch (16 real + 4 sim) so real samples per step
   match the baseline. Everything trains, lr 1e-4, 500-step warm-up. Keep the 10% camera drop but keep the style token
   when dropping, so guidance never mixes domains. Render with style 1, CFG 2. If the sim look leaks, let sim rows
   update only the adapter and cross-attention LoRA, or give them a drop-at-inference LoRA.
6. **Layout score (new metric, E1).** Project nuScenes map polygons and 3D boxes of val frames into the drone camera.
   Run the cached SegFormer-B5 on rendered images. Report road IoU, kerb error in metres, car hit rate (boxes within
   30 m at least half covered by car pixels) and false car pixels, on ground the lidar key covers or boxes. Accept the
   metric only if real keys score far above the mean image and shuffled-camera renders. Use 2 sampling seeds for a
   noise floor. Over 80% of nuScenes val samples lie within 5 m of a train sample (Lilja et al., 2024,
   [2312.06420](https://arxiv.org/abs/2312.06420)), so also report scenes far from any train pose.

| Run | Data | Steps | GPU-h | What it tells us |
|---|---|---|---|---|
| E0 | Alignment overlay, token cosine, sim cameras into `sd_run1` | none | 0.5 | Is the twin aligned; how big is the input gap |
| E1 | Layout score on keys, mean image, shuffled cameras, `sd_run1` checkpoints | none | 1 | Is the metric valid; noise floor |
| R | Render twins and sim cameras, 200 scenes for the pilot (all 850: about 3) | none | 1 | Data |
| P-a | Fork `sd_run1`@10k, real only | 4k | 1.2 | Control |
| P-b | Same fork, 16 real + 4 twin | 4k | 1.5 | Does a complete layout key improve real geometry |
| P-d | Owner's idea, small: sim->sim 3k from scratch, then real 5k (same data as `sd_run1`'s first 5k) | 8k | 2.4 | Versus `sd_run1`@5k: does sim pretraining help at equal real steps |
| Evals | 30 val scenes x 3 runs x 2 seeds | none | 1.5 | Layout score, LPIPS, picture sheets |
| F-B (if P-d wins) | Sim->sim 8k, then the `sd_run1` recipe | 48k | 14.5 | Faster or better, at 5/10/20/40k real steps |
| F-C (if P-b wins) | From scratch, 16 real + 4 sim (best pilot mix) | 40k | 15 | Co-training versus `sd_run1` at equal real samples |
| F-D (gated) | CARLA, 2k+ scenes, one town held out | 40k | 15 + setup | Value of buildings and trees with perfect keys |

**Go rule (pilot is about 8 GPU-h).** Continue an arm if road IoU or car hit rate beats P-a by more than twice the seed
spread, LPIPS on trusted pixels is not worse by more than 0.01, and style-1 pictures show no sim look.

**Failure signs.** Flat grey roads, grass hills or missing facades in style 1 (leak); cars shifted by a constant
offset (alignment bug); sim loss falls fast while the real layout score does not move (sim-only shortcut); F-B ahead at 5k
but level at 40k (report as "faster, not better").

*Caveats.* Twin keys are my proposal, grounded in findings 3, 6 and 7, not a published recipe. Several numbers were
read from paper summaries, not from the tables (Maddukuri details, 6Img-to-3D on nuScenes, GCD failure cases). The
length/width swap and the FOV values come from reading code and intrinsics; E0 checks both.
