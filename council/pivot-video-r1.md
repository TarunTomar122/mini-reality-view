# PIVOT round r1 — from SD 1.5 + bolted-on temporal layers to a PRETRAINED VIDEO MODEL (28 Nov 06:15 UTC)
Owner (28 Nov ~06:11 UTC, verbatim): "i think we are going to struggle with a image mdoel so im now going to ask u to tell main
thread to pivot the startegy and reflect that inr eadme docs and stuff we are going to use a video model now im okay with either
1.3b or 2b i dont mind as long as we get good results and need to run a deep council research round to plan this".
Seats: Astra + Fable design (converge via swaps); Neon verifies FACTS (weights, licences, sizes, papers, speeds) and audits claims.

## Goal (unchanged)
Six real car cameras in -> game-style chase view ("drone view") out, as a SMOOTH, LIVE (causal, frame by frame, never looking
ahead) video with cars in the right place. Camera-only at run time. Live target: ~10 Hz on modest hardware (we have 2080 Ti
11 GB for inference tests; an H200 3g.71gb slice for training, shared; 2080 Tis for rendering / scoring).

## What we have (keep)
- Data: nuScenes, 144 train / 6 dev / 30 exam drives at 10 Hz; per tick 6 camera images + DINOv2-small tokens (data/enc_tpv10),
  per-camera ray maps (data/calib_rays.pt), game-style answer keys rendered from a 3DGS / sim pipeline (data/sim/keys_tpv10,
  512x288 -> 512x256 crops). Signed evaluation: scripts/temporal_metrics.py (764dd3ec95db) + temporal_compare.py (fc7ee2b886a5):
  car jumps (ID), wobble (W), car F1, road IoU, late / early LPIPS gap vs a cold baseline, lingering ghosts, flips, paired drive
  bootstrap, LODO. Single-frame reference A@20k (SD 1.5 + LoRA r64 + CameraAdapter).
- Lessons (PROGRESS.md 27-28 Nov): vlive (shared training noise) jumpier than A (+58% ID @16k); vpriv (private noise) smoother
  (12k: ID -15%, W -7% vs A) but car F1 / drift over limits, lingering ghosts x4; pixel-locked temporal attention cannot follow
  scene motion; a model's own generated history behaves differently from real history (exposure bias); fp16 batch layout
  changes latents (parity checks need same layout); merged-LoRA SD 1.5 = 1.63 s / frame at 25 steps, 0.31 s at 4 on a 2080 Ti.

## Questions (each seat <= 45 lines, to council/pivot-video-r1-<seat>.md; cite papers / model cards / repos with links)
1. Backbone shortlist: compare the top OPEN-WEIGHT ~1.3B-2B pretrained video models (e.g. Wan 2.1 / 2.2 1.3B (T2V / I2V /
   VACE), LTX-Video 2B (incl. distilled / real-time variants), Cosmos-Predict2 2B (Video2World), CogVideoX-2B, others you know):
   weights + licence (commercial / research), architecture (3D full attention vs factorised), VAE temporal / spatial compression
   and what that means for a per-tick live stream, known causal / streaming conversions (CausVid, Self-Forcing, autoregressive
   diffusion with KV cache, few-step distillation), reported speed on consumer GPUs, fine-tuning cost / memory at our resolution.
2. Conditioning: how to feed 6 synchronized surround cameras (+ rays / Plücker, ego pose) into the chosen backbone to produce the
   chase view (cross-attention tokens, channel concat of a projected / warped condition, ControlNet-style, multi-view encoders);
   reuse of our CameraAdapter / DINOv2 tokens? Driving-specific prior art (e.g. MagicDrive-V2, Vista, GAIA, DriveDreamer, Cosmos
   driving post-training) and what transfers.
3. Live path: causal streaming chunk size vs latency (temporal VAE groups of 4 frames?), KV cache, 1-4 step student, expected
   2080 Ti latency; is a 2080 Ti realistic, or should the live target be stated on the H200 / an RTX 4090-class GPU?
4. Plan: the ONE recommended backbone + recipe, staged (stage 0 sanity: run the stock model on our frames; stage 1 conditioning
   fine-tune bidirectional; stage 2 causal / streaming conversion; stage 3 few-step distillation), compute per stage on the H200
   3g.71gb slice, what to measure at each stage with our signed metrics vs A@20k, and kill criteria per stage.
5. Risks and what would make you choose a different backbone.
End with "PROPOSAL: <backbone> — <one line>". Read-only: write only your answer file; no code edits, no cluster jobs, no commits.
Neon: instead answer "FACT CHECK" (<= 35 lines, to council/pivot-video-r1-neon.md): verify sizes, licences, weight availability,
VAE compression factors, published speeds and the existence / claims of the streaming / distillation papers above, with links;
flag anything doubtful.

## OWNER ADDENDUM (28 Nov ~06:25 UTC, verbatim): "make sure we find the best way to connect our images to the video model dont try to reuse and loose"
- Conditioning (question 2) is chosen ON MERIT for the new backbone. Do NOT favour reusing our CameraAdapter / DINOv2-small
  tokens / ray maps / A@20k pieces because we have them; reuse only if it is genuinely the best option. Compare the strongest
  ways to feed 6 synchronized surround cameras (+ calibration / poses) into a video model (e.g. view-to-target geometric
  warping / unprojection into the chase-camera frame, multi-view cross-attention with camera embeddings (Plücker), a strong
  pretrained image encoder per view (DINOv2-L / SigLIP / the backbone's own VAE latents), BEV / 3D lifting, ControlNet-style
  branches, channel concatenation of a rendered guide), with evidence (papers / results) for which gives the most accurate
  car placement and smoothness. A dedicated round on this follows r1.
