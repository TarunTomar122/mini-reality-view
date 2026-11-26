# Car map round 2: Fable

Read: brief, PLAN-carmap v3, `mrv/carmap.py`, `train_carmap.py`, `sd_render.py`, `carmap_diag.py`, `bev_extract.py`,
`bev_box_judge.py`; mmdet3d v0.17.1 source; the T2I-Adapter and ControlNet papers. Nothing was run on the cluster.

## 1. Why the map is ignored (most likely first)
1. **The loss signal sits where the map is useless.** With uniform t and epsilon loss, the gradient is dominated by low
   noise (MSE 0.22 at t=100 vs 0.019 at t=700), where the noisy latent already shows the cars. Placement is decided at
   high noise. T2I-Adapter reports this failure ("guidance information will be ignored during training if t is sampled
   from the later section") and samples t cubically toward high noise.
2. **The cameras make the map redundant** in 90% of rows, cars cover a few percent of cells, and the base is frozen.
   ControlNet drops the prompt 50% of the time for this reason.
3. **Too short for a from-scratch branch.** A zero-init projection on a random stem starts on a plateau. ControlNet's
   "sudden convergence" comes within 10k steps with a pretrained copy; T2I-Adapter trains for about 200k steps (my
   arithmetic: 164k images x 10 epochs / batch 8). The only steady gradient is through the biases and the constant
   "available" channel, hence the map-independent shift.
4. Stacked unreliability (degradation + 15% dropout + phantoms carrying real scores) weakens the signal further.
5. Low, not measured: the UNet weights are bf16 (`unet.to(dev, AMP)`). A 0.07-RMS residual at the full-resolution
   scale is only a few bf16 rounding steps, so small map-dependent differences can round away.
6. Injection point and encoding: I found no wiring fault by reading; shapes and block order look right.

## 2. Experiment R (two arms, identical draws, boxes only, no grid)
Changes shared by both arms: t = (1 - u^3) x 1000; camera dropout 50%; loss weight x8 on cells under true-box hulls
(dilated one cell; a training weight, never an input), normalised; branch width 128 with a 4-conv stem; UNet weights
fp32 under autocast; 4k updates on A6000. First the 100-update timing and the skipped branch-off = E3 render.
- **R1:** frozen E3, branch only. Keeps the signed control and X3 as written.
- **R2:** R1 + LoRA unfrozen at 5e-5 (adapter frozen). Tests whether a frozen base can be steered at all.

**Early check at 500 and 1k updates (dev drives, cameras on).** Single step at t = 700 and 900, fixed noise: car-cell
MSE with the right vs swapped map, and RMS(eps_right - eps_swapped) recorded directly (Astra's point). Pass: right <
swapped on >= 75% of keyframes with cars and a mean gap >= 2% (X1 showed about 0.1%). Then a rendered map-only swap on
dev: correct - swapped >= 0.02 car F1. An arm failing at 1k stops and is never scored on decision drives. This
replaces the 2k decision-drive stop rule.

**Gates.** All signed gates unchanged, comparator E3, one scoring at 4k. For R2 only, the swap must remove at least
half the gain, since extra LoRA training could otherwise claim it. If both pass, R1 wins. R2 as sole winner needs an
X3 amendment, because X3 freezes the temporal winner.

## 3. Decision drives
Keep them for the signed adoption gates only: one scoring per surviving arm at the locked 4k checkpoint. Both looks so
far were nulls and nothing was selected on them, so they remain usable. All early checks and recipe choices move to
dev. Six dev drives are thin for rendered F1: add about 12 unused official-val drives from the P3 calibration pool
(boxes already cached) as a second dev set, if game keys exist or can be rendered.

## 4. Before new training
- **New: the box yaw convention looks mirrored.** mmdet3d 0.17.1 rotates LiDAR boxes clockwise (x' = dx cos + dy sin,
  y' = -dx sin + dy cos; `output_to_nusc_box` uses yaw_nus = -yaw - pi/2). `carmap.py:54-59` and `bev_box_judge.py`
  rotate counter-clockwise. If so, hulls of cars not parallel or perpendicular to the lidar axes are mirrored, and the
  detector heading channel is reversed relative to `gt_frame` for cars parallel to us, which would confound the oracle
  test. This is from reading the source: confirm with one overlay of a turning or angle-parked car. The fix needs no
  re-extraction; the hull numbers (0.417 / 0.444) and P3 need re-running.
- **Per-camera ego poses.** Rebuild lidar2img from fixed calibration only, re-extract, repeat the 0.397 hull gate and
  P3 (with hull F1), then re-degrade. The plan already requires repeating the adoption gates after a motion-source
  change, so do it before the 4k runs. The 1k dev check may start on the current caches (yaw fixed) as a mechanism
  probe, labelled non-adoptable.
