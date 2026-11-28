# PLAN-video — pretrained video model pivot (drafted 28 Nov ~06:24 UTC; SIGNED by Astra + Fable ~06:30; earlier '~06:40 / 06:50' labels in this file were ahead of the clock — Neon)
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

## SIGNATURE ROUND RESULT (28 Nov ~06:30 UTC) — supersedes the open points above
Sources: PLAN-video-{astra,fable}.md, PLAN-video-swap-{astra,fable}.md. Fable's corrections C1-C9 accepted by both seats (tie rule
as Fable revised; S0c keeps the LiDAR-depth ceiling arm, wrong-drive control and "still improving -> unresolved" rule; lag = 300 ms
first-to-fourth-tick wait at 10 Hz + compute (est. 0.4-0.5 s on a 4090, 0.6-0.7 s on a 2080 Ti; S0b measures); per-tick still
mode measured in S0a / S0b; causal conversion and few-step self-forcing are ONE stage; S0a kill compares Wan vs the SD round trip
too). Builder merge of the final numbers (Fable's final = midpoint inside the range Astra accepted; Astra asked to confirm):
- S0c: <= 5 aggregate 2080 Ti-h incl. caching and scoring; matched guide and token readers + hybrid report + LiDAR ceiling +
  wrong-drive control; a lead needs >= 0.03 car F1, paired drive interval > 0, positive in every LODO, no road-IoU loss; otherwise
  both inputs go into S1 (S1 trains with independent input dropout; hybrid / guide-only / tokens-only compared at fixed noise on
  the same checkpoint; equal quality -> lower jitter, then lower measured runtime).
- Depth: Depth Anything V2-Small (Apache-2.0 metric checkpoint), per-camera scale fitted on train drives and frozen; LiDAR as
  training supervision only if the LiDAR ceiling beats the estimated-depth guide by >= 0.05 dev car F1. Neon to check the VKITTI2 /
  nuScenes licence points.
- S1 (bidirectional teacher): 24 H200-3g slice-hours (+ one 12 h extension), verdict only after >= 12k updates (A1: earlier =
  unresolved, not failed); 512x256, 17-tick clips (floor 13); LoRA rank 32 on self-attention, cross-attention and MLPs in all 30
  blocks (rank 16 if it fits train but not dev); new guide / token projections trained fully; base and VAE frozen. Pass: complete
  paired dev panel with car F1 >= A (0.4456), road IoU >= A, ID and W each > 9.4% better than A, camera-swap check passes;
  extension only if car F1 in [A-0.03, A) and the rest pass; else stop and re-council.
- S2 (causal + 4-step self-forcing, own-rollout training): 40 slice-hours (+ one 16 h tranche); A1 from 8k-equivalent exposure;
  A4 on whole dev drives with own history; cap before an eligible verdict = unresolved.
- S3 (2-step student, tiny decoder, live timing): 24 slice-hours hard; advance only with A4 retained, sustained 10 Hz and the
  owner's capture-to-display limit. 30-drive exam sealed until recipe lock + N1.
- OWNER DECISION before S1 (after S0 numbers): latency contract (native 4-tick groups ~0.4-0.7 s behind vs per-tick stills).
- All hour caps are restated in updates once S0b measures the rate. Worst case ~4-5 days on one slice (Fable estimate).

## OWNER DECISION — latency contract (28 Nov ~06:30 UTC, verbatim): "im okay with the delay go go go"
Native Wan 4-tick latent groups are accepted: 10 Hz playback with a fixed lag (~0.3 s group wait + compute; S0b measures). The
per-tick still mode is no longer required (kept only as an S0a measurement). S1 proceeds with native 17-tick clips.

## Neon fact check (council/PLAN-video-neon.md, 28 Nov ~06:35)
- Wan2.1 VAE confirmed: 4x temporal, 8x8 spatial, 16 latent channels, 1+4k frames; DiT patch [1,2,2] -> 16x16 px tokens.
- Licences: nuScenes is CC BY-NC-SA 4.0 + Motional terms -> every nuScenes-trained checkpoint (S1-S3) is NON-COMMERCIAL with
  attribution. Depth: use "Depth-Anything-V2-Metric-VKITTI-Small" (Apache-2.0 tag; VKITTI2 data itself is NC); the untagged
  Metric-Outdoor-Small is not used; plain -Small is the fallback. No "cleanly commercial" claim.
- S2 budget: Self-Forcing on Wan 1.3B at 832x480x81f used 64 H100 x ~1.5 h (DMD, with a 14B real-score net) / 2-3 h (SiD);
  our 40 slice-hours are viable only because 512x256x17f is ~12.8x fewer tokens and our teacher is 1.3B; the causal ODE-init is
  budgeted INSIDE S2's 40 h; gradient accumulation to a global batch ~32-64; S0b / pre-S2 must time the 3-network DMD / SiD step
  on 71 GB.

## Notes recorded with the signatures (Fable final sign; Neon 06:30)
- S1 floor = 12k updates at batch 8 (global batch via accumulation); after the 12 h extension the pass bar stays car F1 >= A
  (0.4456): still below A -> stop and re-council.
- S0a reading (Fable 06:30): the informative comparison is Wan round trip vs the SD round trip / the key itself scored DIRECTLY on
  RGB (scripts/score_rgb.py, unchanged signed scorer): a Wan drop of > 0.03 car F1 or more flicker vs the key = problem.
- S0c is not a launch gate for S1 (S1 trains both inputs with input dropout).

## C9 dropout amendment (28 Nov ~08:40 UTC; Fable proposed 07:30, Astra + Fable accepted 08:30)
Input dropout is one draw per clip after the two guard updates: both inputs dropped 10%, guide only dropped 10%, tokens only
dropped 10%, hybrid 70% (flags --p_drop_both / --p_drop_guide_only / --p_drop_tokens_only). Marginals: guide dropped 20% (as
signed), tokens dropped 20% (was 10%) so the guide-only ablation and the unconditional branch have direct training examples. Both
20% marginals are recorded with every comparison.
