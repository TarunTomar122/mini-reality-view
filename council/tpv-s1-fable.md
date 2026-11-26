# Third-person game view: round 1 (Fable)

## 1. Initialisation

**Recommend B (trained adapter + LoRA from sd_run1 40k), with A (plain SD, fresh adapter) as the control.**

- Every close relative in the literature starts from pretrained SD and fine-tunes; none trains a renderer from
  scratch on nuScenes-sized data. MagicDrive (SD + box/camera tokens through cross-attention + a ControlNet-style
  map branch, https://arxiv.org/abs/2310.02601), BEVControl (https://arxiv.org/abs/2308.01661), BEVGen
  (https://arxiv.org/abs/2301.04634), Zero-1-to-3 (SD fine-tuned for a new viewpoint,
  https://arxiv.org/abs/2303.11328). So A is the literature default and B is "A plus 40k steps of our own task".
- What those 40k steps bought is the slow part: reading 6 cameras into a layout (car F1 0.03 at 5k, 0.26 at 10k).
  Our pilot says look changes fast and placement changes slowly, so a renderer that already places cars should
  re-learn "game look + lower camera" quickly. The risk is that it also keeps habits from the messy keys (smears
  behind the car). Clean, fully-trusted keys should wash that out, but I cannot prove it: hence the A control.
- C (trained adapter, fresh LoRA) I would skip. Adapter and LoRA were trained together; the LoRA holds the
  "tokens -> pixels" half of the geometry. C keeps half of a matched pair and costs a slot.
- Caveat: the pilot's sim head start vanished at equal total updates; B may also be a head start, not a higher ceiling.

**Other base models: not now.** SD 2.1 has the same cost and no clear gain. SDXL(-turbo) is tight on an 18 GB slice
and throws away the trained adapter. Small pixel models: our 45M DiT already lost to SD.

**Non-diffusion decoder: worth it later, not first.** The key is exact, but the answer is not determined by the
photos (hidden and far cars are guesses). A plain regression loss averages over guesses and gives blurry or ghost
cars, exactly where we are scored. Two paired-data results suggest a good second stage: SD fine-tuned end to end as
a one-step deterministic model works well for depth/normals (https://arxiv.org/abs/2409.11355), and pix2pix-turbo
does paired translation in one step from SD-Turbo (https://arxiv.org/abs/2403.12036). Both need image-shaped input;
ours is tokens, so transfer is uncertain. Fold this into the planned distil step.

**Free checks before training.**
- VAE ceiling: encode + decode the game keys and score them with the metrics below. Thin lane lines and small far
  cars may not survive 8x compression; if so no model can draw them.
- Game renders have large flat areas. SD 1.5's noise schedule leaks overall brightness
  (https://arxiv.org/abs/2305.08891, cited from memory, not re-checked today). Watch for washed-out grass/sky; only
  fix if seen.

## 2. Is ~7,900 pairs enough?

Honest answer: unknown, and I lean "enough for the look, maybe not for placement". Frames 1 s apart in one drive are
near-duplicates, so the real count is ~508 drives. A 40k x 16 run is ~80 passes over the data, so memorising is a
real risk. MagicDrive-class models use all ~28k train keyframes.

**Test:** same recipe on all drives vs a quarter of the drives (experiment 3), scored on held-out drives and on a
fixed sample of train frames. Two signals: the val gap between the two, and the train-minus-val gap of each.

**Growth order if data-limited:**
1. All 700 train drives (+~190 new drives). New places are worth far more than new frames.
2. Every keyframe (2x, same drives). Cheap, modest gain.
3. Non-keyframes last. 10x rows but almost no new content, and camera/sim timing can differ by tens of ms
   (~0.4 m at 10 m/s) with interpolated boxes: that puts noise back into a key whose point is exactness.

Phone-augmented tokens are input variety, not new scenes; add them in the long run, keep them out of the tests.

## 3. Evaluation

Held-out drives rendered the same way. Note `sim_render.py` only accepts official train scenes; either add a val
option (preferred: tokens and judge frames exist for val) or hold out 30 train drives by a fixed list.

- **Primary: car placement F1 against exact game instance masks.** Render an instance/semantic pass for each val key
  (MetaDrive ships semantic and instance cameras; check they work headless in our build). Run the Mask2Former judge
  on the model's picture, match to game car masks (IoU >= 0.3, >= 8 px tall, own car excluded), pooled F1.
  Report by distance (0-15, 15-30, 30+ m).
- **Judge ceiling first:** the judge was trained on real streets and may misread generic game cars. Score the true
  keys with it. If ceiling F1 < ~0.7, use a simpler judge: game cars have few flat colours, so a tiny segmenter
  trained on (key, mask) pairs from train drives is cheap and exact.
- **Road:** drivable-area IoU against the game road mask (replaces kerb error; no projection needed).
- **Pixels:** L1 and LPIPS vs the key, whole image (no trust mask now), reported only. The own car and sky are the
  same in every frame, so show two floors beside them: "no cameras" (null tokens) and "shuffled cameras".
- Paired scene bootstrap with 95% CIs, as before.

## 4. Three experiments (~3 GPU-h each, in parallel)

Common: TPV game keys only, clean real tokens, style id 2, `--n_styles 3`, bs 16, lr 1e-4, 500 warmup, 10,000 steps
(~2.4 h at 70 steps/min), `--save_at 2500,5000`, `--seed 0` with `--set` so all arms see identical draws, DDIM 25,
CFG 2, 30 held-out drives. Scoring of snapshots runs afterwards on the 2080 Ti / A6000.

Changes to `train_sd.py` (small):
- `--init` with 2 -> 3 styles: copy the old style rows, new row = copy of row 1 (now `load_state_dict` would fail).
- `--scene_list file`: keep only listed train scenes in `load_set` (for E3).
- `--render_scenes 30`, `--render_style 2`; also render a fixed 200 train frames for the train/val gap.

| | Tests | Init | Data |
|---|---|---|---|
| **E1** | reference | sd_run1 40k (adapter + LoRA) | all ~7.9k pairs / 508 drives |
| **E2** | does our old training help? | plain SD 1.5, fresh adapter | same as E1 |
| **E3** | is data the limit? | same as E1 | fixed random 127 drives (~2k pairs) |

**Decision rules** (primary = car F1 at 10k, check 5k for trend):
- **Init:** E1 beats E2 by >= 0.03 with CI excluding 0 -> use B. Otherwise use A (simpler, no dependence on the
  messy run; matching B at 10k from a cold start means it is climbing faster). If E2 has almost no cars at 10k, B
  wins outright.
- **Data:** E1 - E3 >= 0.05 F1 (CI excluding 0), or E3's train-minus-val LPIPS gap clearly larger -> data-limited:
  render the remaining train drives and every keyframe before the long run. Gap < 0.02 -> 7.9k is enough, spend on
  steps. In between -> add the 190 drives only.
- **Go:** long run (40k, winning init, phone tokens added) if the best arm reaches >= half the judge's ceiling F1 at
  10k and road IoU is not falling between 5k and 10k. If below a quarter of ceiling, stop and come back to the
  council (likely the camera adapter, not the data, is the limit; that is where the BEV idea returns).

## Uncertainty

- Two points on a learning curve is thin; a small E1-E3 gap at 10k steps can hide a gap that opens at 40k.
- One seed per arm. Earlier CIs were about +-0.04 F1, so differences under ~0.03 are not readable.
- Whether the Mask2Former judge works on game pictures is untested; the ceiling check gates everything.
- MagicDrive-style models get boxes as input; we infer them from photos. Their numbers say nothing about how well
  placement can be learned here. Our own 0.29 plateau is the only evidence, and it was on noisy keys.
