# PLAN-video (DRAFT for signature) — pretrained video model pivot, 28 Nov ~06:40 UTC
Sources: council/pivot-video-r1{,-astra,-fable,-neon}.md, council/pivot-cond-r1{,-astra,-fable,-neon}.md. Owner directives: pivot
to a ~1.3-2B pretrained video model with a deep council plan; "make sure we find the best way to connect our images to the video
model dont try to reuse and loose".

## Converged so far (both seats)
- Backbone: Wan2.1-T2V-1.3B (Apache-2.0; the base of CausVid / Self-Forcing and later open streaming work). Wan VAE: 4x temporal,
  8x8 spatial (Neon to confirm channels); at 10 Hz one latent frame = 4 ticks. Fallback: Cosmos-Predict2.5-2B (licence / compute).
  Rejected: LTX-Video 2B (8-frame group, cells larger than cars, revenue-capped licence), CogVideoX-2B, Wan 2.2 (no 1.3B).
- Conditioning: HYBRID, GEOMETRY FIRST. Per tick, estimated metric depth for each of the 6 views (monocular net, scaled by the known
  rig / camera height; may be fine-tuned with training-drive lidar as SUPERVISION only — camera-only at run time) -> unproject and
  splat the 6 views into the chase camera -> a warped guide image + validity mask fed to Wan as extra input channels (zero-init
  patch-embed extension). Raw-view attention (per-view tokens + Plücker camera embeddings via cross-attention in place of text) is
  a second path, kept only if it adds car accuracy. CameraAdapter / DINOv2-small tokens are NOT assumed (owner: no reuse bias).
- Depth source: open, camera-only, live-capable, commercially usable where possible (Neon: Depth Pro, UniDepth, DUSt3R / ViewCrafter,
  SEVA are non-commercial; VGGT is up to scale; Metric3D v2 / Depth Anything V2-small to be checked); timed first (est. 30-50 ms /
  tick on a 2080 Ti, unmeasured).
- Live target: 10 Hz playback on a 4090-class GPU / the H200 (2080 Ti = measured stretch); a fixed chunk lag (~0.3-0.5 s) is the
  expected cost of the 4x temporal VAE — an owner decision after stage 0 numbers.
- Evaluation: signed metrics (temporal_metrics 764dd3ec95db / temporal_compare fc7ee2b886a5), A@20k as the single-frame reference,
  AGREED bars (smoothness > 9.4% vs A on ID and W; car F1 >= A - 0.03; late gap <= +0.03; ghost / road conditions as A4), A1 timing,
  N1, own-history checks for any causal model.

## Stages (compute on the H200-3g slice unless noted; each with a kill rule)
S0a (no training, 2080 Ti / H200, ~1 h): Wan VAE round trip of the 6 dev drives' answer keys (5-tick-aligned chunks), re-encoded
    into the signed scorer's input and scored vs the keys, next to the SD-VAE round trip. KILL: car F1 after the Wan round trip
    below A@20k's car F1 (the codec alone would lose cars) -> re-council (Cosmos 2.5 fallback / higher resolution).
S0b (no training, ~0.5 h): timing of the stock Wan 1.3B transformer at our resolution (2080 Ti, H200) per latent chunk and per step;
    depth net timing per tick.
S0c (~3.5 2080 Ti-h, no video model; Fable's discriminating test, Astra's variant noted): two equal-size readers predict the key's
    car / road masks for each dev tick, one from the warped guide (+ mask), one from full-resolution raw-view tokens + Plücker;
    the one better on car F1 by >= 0.03 leads; within 0.03 -> guide leads (cheaper, causal by construction) and tokens stay the
    second path. (Pre-registered; exact tie rules in pivot-cond-r1-fable.md.)
S1 bidirectional teacher: Wan 1.3B + guide channels (+ tokens if S0c says so), LoRA / partial fine-tune on 144 train drives as
    clips of 4k+1 ticks; checks on the dev panel vs A@20k with the signed metrics (bidirectional = smoothness / accuracy ceiling).
    KILL: no checkpoint by the agreed budget with car F1 >= A - 0.03.
S2 causal student (CausVid / Self-Forcing style): block-causal attention + KV cache over latent chunks, trained on its own rollouts
    (self forcing, against exposure bias); gated under AGREED bars from 8k-equivalent updates (A1).
S3 few-step distillation (2-4 steps) + small decoder; live timing (p50 / p95 camera-to-display, sustained fps) on the target GPU.

## Open points for the signature round
1. S0c design: Fable's reader test vs Astra's ~4 GPU-h comparison (with or without the video model) — pick one.
2. Depth source + licence choice; whether lidar-supervised fine-tuning of the depth net is in scope.
3. S1 budget, resolution (512x256 vs a Wan-native size) and LoRA rank / which blocks train.
4. Kill thresholds for S1 / S2 in hours.
