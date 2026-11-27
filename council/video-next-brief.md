# Next video step (owner, 27 Nov 06:20): "i need the jumps to go away and video to become safe and then further training
# to make the video smooth". ONE round + one swap to converge. Budget: about one day of compute (A6000s / 2080 Tis,
# <= 12 jobs; H200 slices mostly unavailable).

## What we know (scorer 764dd3ec95db, compare fc7ee2b886a5; PROGRESS.md has all numbers)
- Renderer: E3 = SD 1.5 + LoRA r64 + CameraAdapter (DINOv2 tokens of 6 cameras + rays), one frame at a time, DDIM 25,
  CFG 2, 512x256 game-style chase view. Frames are drawn independently -> cars flicker / jump between 10 Hz ticks.
- Warm start (SDEdit from the previous output, s = 0.7), decision set: ID -22.5% (fewer jumps) but W +3%, car F1 0.258
  vs 0.317, late drift +0.058 LPIPS, lingering ghost cars 13.7 vs 5.1 per run -> rejected.
- Arm B (previous output as a corrupted extra input through a zero-init conv beside conv_in, trained 4k), dev: ID -9 to
  -23%, W -5 to -9% vs arm A, but lingering ghosts +7.6 to +14.2 per run, car F1 -0.03 to -0.07, own drift +0.04 to
  +0.07 at every history strength -> failed dev screen. Arm C (B + training on its own rollouts) dev results pending.
- Pattern: any feedback of the previous output lowers jumps but makes cars persist after they leave (ghosts) and drift.
- Old DiT line (pre-SD, splat targets): full_c_self (own-rollout training) reached drift +1.8 L1 over 20 s with s_prev 0.3.
- Data: 144 train drives with 10 Hz game keys (data/enc_tpv10: DINO tokens per tick + key latents), ~20 s each.
- Signed PLAN-temporal fallback: "Neither B nor C passes -> the next round goes to input memory" (past camera tokens /
  BEVFormer temporal features), still a candidate.

## Options (not exhaustive)
1. Multi-frame video model: temporal attention / motion module (AnimateDiff-style) inserted into the frozen-ish E3 UNet,
   trained on clips of N consecutive 10 Hz keys, drawing N frames jointly (sliding window with overlap at inference).
2. Longer / stronger self-rollout training (C-style own-frame history, longer clips, more updates).
3. Causal smoothing of the camera conditioning (e.g. EMA / attention over the last k ticks of DINO tokens) - input memory,
   so no output feedback (should not create ghosts).
4. Input memory via temporal features (past camera tokens or BEVFormer's temporal BEV) as an extra condition.
5. Anything else you think is better.

## Questions
Which gives the biggest drop in jumping WITHOUT ghost cars or drift, for about a day of compute? Give ONE plan: what
exactly to build / train (clip length, layers, what is frozen, updates, data), how to evaluate with the signed scorer
(gates: ID / W down >= 15% vs E3 independent with intervals, placement / recall / road not worse, no lingering ghosts,
late drift <= +0.02), early stop rule, and expected cost. Write <= 500 words to council/video-next-<you>.md.
