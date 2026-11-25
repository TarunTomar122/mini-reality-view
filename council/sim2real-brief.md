# Council brief: how should we use simulator data? (round 1)

## The project in short
We are rebuilding Tesla's FSD v15 "Reality View" at home: from a car's 6 surround cameras, draw a photo-like view of
the car's surroundings from a virtual "drone" camera behind and above it. Camera-only at run time. Data: nuScenes
(850 scenes of 20 s, 6 cameras, keyframes at 2 Hz; 700 train / 150 val official split).

Read these files in this folder (repo root) before you start:
- `README.md` (what we have built and measured, in plain words)
- `PROGRESS.md` (lab notes, newest first: every result, failure and decision)
- `mrv/sd.py`, `scripts/train_sd.py` (the current main model)
- `scripts/sim_metadrive_test.py`, `scripts/sim_scenarionet_test.py`, `docs/figs/sim_metadrive_cams.jpg`, `docs/figs/sim_metadrive_drone.jpg`

### Current main model (sd_run1, training now)
- Frozen Stable Diffusion 1.5 UNet + LoRA (r=64 on attention/proj/ff) + a trainable "camera adapter": DINOv2-small patch
  tokens of the 6 cameras (16x9 pooled per camera, 384-d) + per-patch Plücker viewing rays (ego frame) + per-camera
  embedding + a style token -> 865 tokens replacing the CLIP text prompt. CFG via learned null tokens (10% drop).
  SD's DDPM schedule, epsilon prediction, loss weighted per pixel by key coverage/trust.
- Output: 512x256 drone view from a "low drone" 9 m behind, 4.5 m up, looking at a point 15 m ahead and 0.5 m up,
  60 deg horizontal FOV.
- Answer keys ("ground truth") are imperfect: (a) lidar keys = camera-coloured lidar points seen from the drone
  (holes, ring stripes, 20-80% coverage); (b) Gaussian-splat keys (drivestudio Street Gaussians, 70 scenes) rendered
  from the drone, with a per-pixel trust mask (seen by a real camera, not occluded, angle/distance weight). Style token
  0 = lidar key, 1 = splat key. Training mix 70/30.
- At 5k of 40k steps it already draws photo-like streets roughly where the key has them (see README). Phone-style
  augmentation of the camera images (crop/zoom, colour, blur, JPEG, with recomputed rays) is being added; the final
  goal includes running it on 6 phone photos taken in a street.

### The sim idea (from the project owner)
In a simulator the drone-view key is perfect (every pixel known, including what no real camera saw). Idea: pretrain
the same SD renderer on sim pairs (6 cameras placed like the nuScenes rig -> sim drone view), then fine-tune on the
real keys, and compare with real-only at equal real steps. The owner explicitly wants this done with research behind
it, not winged.

### What we know about simulators so far
- MetaDrive (pip, Panda3D) works headless on our GPUs: 6 rig cameras + drone view in 0.16 s/frame. Procedural maps are
  an empty highway in grassy hills: no buildings, trees, parked cars; traffic far away (see docs/figs/sim_metadrive_*).
- ScenarioNet (replays real nuScenes layouts + logged traffic inside MetaDrive): installing/testing now.
- CARLA 0.9.x: much better graphics (towns, buildings, pedestrians, weather); the cluster has apptainer/singularity;
  Vulkan on GPU nodes unverified; ~20 GB.
- Compute: shared university Slurm partition: RTX 2080 Ti (11 GB), one A6000 node, H200 MIG slices (18 GB / 71 GB).
  SD LoRA training: ~1.8 min / 100 steps at batch 16 on an 18 GB slice. Max 2-day jobs.

## Your job
Do real research (search the web/arXiv; cite paper, year and link for every technique you lean on), then write a
concrete plan. Questions to answer:
1. What does the literature say about mixing synthetic and real data for (a) conditional image/video generation and
   (b) driving perception / view synthesis? E.g. pretrain-then-finetune vs joint training, mixing ratios, domain/style
   tokens or embeddings, domain randomisation, photorealism enhancement of sim images (e.g. EPE), layout-conditioned
   driving generators (MagicDrive, BEVControl, Panacea, DriveDreamer, etc.), feature-level alignment, and when sim
   data hurts (negative transfer). Be specific about what transfers (geometry/layout) and what does not (appearance).
2. Which simulator for us, given effort vs value? MetaDrive procedural, MetaDrive + ScenarioNet nuScenes replays, CARLA,
   or something else (e.g. existing synthetic datasets with surround cameras that we could re-render or that already
   include an extra viewpoint)? Be concrete about what each gives for *our* drone view.
3. Exact data recipe: how many episodes/frames, what to vary (maps, weather, time of day, traffic/pedestrian density,
   camera jitter), how to match the nuScenes rig and image statistics, hiding our own car in the drone view, whether
   sim keys should also get coverage/trust masks, and whether to pass sim camera images through the phone augmentation.
4. Exact training recipe for our SD + LoRA + camera adapter: which parts train in the sim phase vs the real phase,
   steps, learning rates, mixing ratio, style token for sim, whether to freeze the adapter or LoRA at some point,
   CFG null-token handling, and how to keep the comparison with real-only fair (equal real steps).
5. Evaluation: how to tell whether sim helped *geometry* (road edges, car positions) and not just texture, on the 30
   real val scenes with imperfect keys. Propose a cheap, honest metric if masked L1/LPIPS are not enough.
6. A cheap pilot (a few GPU-hours) that decides go/no-go before the full run, and the risks/failure signs to watch.

## Output
Write ONE file, `council/s1-<your seat name>.md` (seat names: `astra` or `fable`), at most ~2 pages: findings with
citations, then a numbered plan with an experiment table (run, data, steps, GPU-hours, what it tells us). Plain words.
Do not edit any other file, do not run jobs, do not touch the cluster. Finish by stating the file path you wrote.
