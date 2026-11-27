# Fidelity check (owner, 27 Nov 06:40): "if something is proven in research and doesnt work for us then maybe we are not
# doing it correctly instead of jumping to a new thing". Before building a new method, list how OUR versions differ
# from the published recipes, and which difference most likely explains the failure.

1. Arm B (history input = previous frame, noise-augmented; like GameNGen, arXiv 2408.14837) and arm C (B + own-rollout
   history): ours = 4k updates (cut from 6k for budget) from E3, ONE previous frame, s ~ U[0, 0.7] noise level fed as a
   plane, zero-init conv beside conv_in, LoRA r64 + adapter trained at 5e-5, batch 16 (12 dense + 4 singles), C's own
   frames from two cache refreshes (1k, 2.5k). Results: dev ID -9..-23%, W -5..-9%, but lingering ghosts +7.6..+14.2 per
   run, placement -0.03..-0.07, drift +0.04..+0.07 at every s. History conv weight norm still growing at 4k.
   GameNGen: many context frames, long training (hundreds of thousands of steps), noise augmentation with the level
   given, model sees its own errors... Please list the concrete deviations (code: scripts/train_temporal.py,
   scripts/temporal_rollout.py) and rank them.
2. Car map (ControlNet / T2I-Adapter / GLIGEN-style layout conditioning): ours stopped at a 1k single-step gate (R: input
   conv; F: bias-free 4-scale branch) after a 2k rendered gate for X1 / X2. ControlNet reports "sudden convergence"
   after several thousand steps. Closed by the signed rule; the owner has NOT asked to reopen it - list the deviations
   for the record only.
Already running: A and B continued 4k -> 12k (equal updates; checkpoints 6k / 8k / 12k) to test undertraining; B 2k dev
trend queued. Answer in <= 300 words: the ranked deviations for 1 (and 2, briefly) and the ONE fix to test first.
Write to council/fidelity-<you>.md.
