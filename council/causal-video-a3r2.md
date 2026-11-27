# A3 round 2 (27 Nov ~22:45 UTC): choose the NEXT ARM before vlive's 16k lands (~23:30)

A3 is DRAFT (Neon 22:30: the builder misread Astra's swap note; council/AGREED-causal-r1.md marks it not binding).
Read: council/causal-video-a3*.md, council/review-20261127-2230-{astra,fable,neon}.md, PROGRESS.md top entries.

## New evidence since the A3 swap
- V2k live stream (rho 0.95, signed, informational): vs signed A@20k ID +61.5%, W +30.7%, F1 -0.008, late +0.003, lingering
  ghosts 0.25 vs 4.33; vs own cold (off at 0.95) ID +34.4%, W +19.6%, F1 -0.007, ghosts 0.25 vs 3.42 -> memory makes the live
  video JUMPIER at 2k.
- Offline noise probe @8k (runs/probe/noise_8000.json): content_only / off = 5.3x / 7.0x / 10.0x (t 900 / 700 / 300), up from
  4.0 / 4.3 / 5.3x at 2k: the blocks rely MORE on the shared past noise; difference loss diff_on / diff_off 0.466 vs plain
  on / off 0.463 at t300 (the shortcut already cuts the difference term as much as the plain one).
- Gradient-cosine probe (plain MSE vs difference loss on train clips, ckpt 2k / 8k) running (runs/probe/gradcos_*.json).
- 4k / 8k live checks scoring; vlive 16k ~23:30.

## Candidate arms (all from frozen A@20k, same tests / bars / A1 timing; one arm only)
 (a) the draft A3 arm: rho 0.95, MSE + lambda x difference loss (lambda / pair range as signed text).
 (b) Fable: half the training clips with FULLY PRIVATE noise per tick (rho 0 for those clips), so the shared-noise trick fails
     on half the batch; inference rho stays 0.95 (or chosen by the same off rule).
 (c) Fable: cleaner-past design (from the 21:30 note) -- needs a full design.
 (d) no training: history guidance at inference (blocks off on the uncond row) on vlive checkpoints (~1 GPU-h to test).
 (e) other.

Each seat, <= 12 lines to council/causal-video-a3r2-<seat>.md: pick ONE arm (plus optionally (d) as a free side probe),
exact recipe (noise policy in training AND rollout, loss, steps, from which checkpoint), trigger (vlive 16k non-pass with safety
met, as before), tests / signature, and end "SIGNED: <A3 text>". Read-only. 2-line summary.
