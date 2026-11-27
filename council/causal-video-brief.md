# Council brief: a CAUSAL (live) video model (27 Nov ~18:00)

Owner (verbatim): "this shouldnt be this hard also we are traiinng an image model rn right?" ... "no bt i want the ability to get
live view and us aying that a video model cant do it is stupid".
So: the target is a LIVE (causal, frame-by-frame, no look-ahead) chase-view video that is smooth (no jumping cars) and accurate.

## Why we are here
- Today's one-frame "memory" (zero-init 3x3 conv on the previous rendered latent beside conv_in of the SD 1.5 + LoRA E3 renderer)
  always learns to copy: drift grows with training under every recipe (stale cache, live cache, lever 2: harder true-key history).
  Lever 2 longC3: 10k late gap +0.014 but smoothness 7.9% < 9.4% bar; 15k 12.6% smoother but +0.039 drift, F1 -0.036 vs A; 20k
  +0.071, F1 -0.059 vs A (memory OFF on the same weights: F1 0.452 >= A 0.446). Full story: PROGRESS.md (27 Nov),
  council/drift-*.md, council/review-20261127-1735-*.md.
- Already written (held, never trained): mrv/video.py (zero-init temporal attention block after each of the 16 spatial
  transformers of the FROZEN E3 UNet, frame positional encoding, bidirectional), scripts/train_video.py (16-frame clips of the
  144 train drives, PYoCo mixed noise, one timestep per clip, epsilon loss, only temporal blocks train), scripts/video_rollout.py
  (sliding 16-frame windows, NOT causal). Design history: council/video-next-*.md.
- Hardware: 1 A6000 (48 GB) for training, ~110 updates/min for the image model at batch 16; 2080 Ti x up to ~12 for rollouts;
  owner cap 16 cluster jobs. Data: data/enc_tpv10 (DINO tokens + key latents per 10 Hz tick, 144 train drives, ~185 ticks each),
  dev = 6 drives, signed scorer scripts/temporal_metrics.py (764dd3ec95db) and temporal_compare.py (fc7ee2b886a5) score any
  rollout saved in the temporal_rollout format.

## Question: design the causal video model we should train NOW
Ground it in published causal / streaming video diffusion (e.g. Diffusion Forcing, CausVid, Self Forcing, StreamDiT, rolling
diffusion, GameNGen / DIAMOND / Oasis history conditioning, AnimateDiff / Video LDM temporal layers) and say how each choice maps
to our setup. Cover:
1. Architecture: causal temporal attention (mask) over a window of K past frames (which K at 10 Hz?), KV cache at inference;
   keep E3 frozen (accuracy) and train only temporal layers? or also LoRA? positional encoding for a rolling window.
2. Training objective: Diffusion Forcing (independent per-frame noise levels) vs teacher forcing on clean past frames vs
   Self Forcing (train on own rollouts with a clip-level loss); what is the cheapest one that avoids today's copy/drift failure?
3. Inference: how a new frame is generated live (denoise only the newest frame with past frames clean / slightly noised as
   context; noise schedule; DDIM steps; CFG); cost per frame on a 2080 Ti vs the image model.
4. What exactly to change in mrv/video.py, scripts/train_video.py, scripts/video_rollout.py (concrete), budget (A6000 hours),
   first checkpoint to look at, and the dev gate vs A (smoothness > 9.4% on the smaller ID/W reduction, car F1 >= A - 0.03, late
   gap vs own cold-equivalent <= +0.03 — say what "cold" means for a video model).
5. Risks (what makes this fail like today's memory) and the early warning.
Write council/causal-video-s1-<your name>.md (<= 60 lines) and reply with a 5-line summary. Read-only (repo + read-only ssh to
`mlp` with < /dev/null if needed); no code edits, no jobs, no commits.
