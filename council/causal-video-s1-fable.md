# Causal (live) video model: design (Fable, s1, 27 Nov)
**Pick: "V-live" = the held 16-frame model made causal: one-way temporal attention over the last 8 ticks, every past frame
kept exactly as noisy as the frame being drawn, one cache per sampling step.** This is Live2Diff's recipe (2407.08701: SD 1.5 +
temporal layers, one-way mask, "for every combination of layer and denoising step, we keep a separate KV-cache"), with our six
cameras in place of their source video. Causal now; real-time speed is a second stage (section 3).
**Why.** Today's memory hands the model a FINISHED previous frame while the new frame is still noise. At 10 Hz that frame is
nearly the answer and the cameras are weak evidence (car F1 0.45), so training rewards copying; every recipe only slowed it.
GameNGen / DIAMOND / Oasis accept that trade because they cannot see the present. We can. In V-live nothing finished exists
when the layout is decided (high noise): the past offers only its camera evidence and its half-made picture. Read today,
2609.22283 ("Rethinking Streaming Video Diffusion"): same-level history beats clean history (VBench 85.24 vs 84.45, dynamic
degree 84.7 vs 77.8), and clean history leaves backgrounds "nearly frozen", which is our freeze.
## 1. Architecture
- **Base: freeze A@20k** (runs/longA/ckpt_20000.pt: UNet, LoRA, adapter), not E3 10k. A is the gate comparator and was trained
  on the dense ticks. Blocks off = A exactly, so A's scored dev renders are the control: no control run. Format fits `--init`.
- **Train only the 16 temporal blocks** (zero-init output, 161 M weights), as AnimateDiff (2307.04725) and Video LDM
  (2304.08818). No LoRA in stage 1, so any accuracy loss is the blocks' doing. LoRA at 1e-5 only after a pass.
- **Window K = 8**: self + the 7 ticks before (0.7 s). Published windows run from 4 frames (DIAMOND) to 16 (Live2Diff); 8 is
  what the cache allows on a 2080 Ti. Train on 16-tick clips so ticks 8-15 see a full window whose members had their own past.
- **Position: relative** (RoPE over the tick index on queries and keys). The current code adds absolute clip positions 0-15,
  which a rolling window never reproduces; Live2Diff leans on bidirectional warm-up frames, which a live start does not have.
## 2. Training objective: same-level, on true keys only (the cheapest of the three)
- As `train_video.py` does now: ONE timestep per clip, PYoCo mixed noise, epsilon loss; no rollouts, cache or refresh jobs.
- **Not Diffusion Forcing** (2407.01392, independent level per frame): it trains "clean past, noisy present" pairs, today's
  shortcut. Arm C was that plus GameNGen's noise augmentation (2408.14837); its noise dial "delays, not cures".
- **Not Self Forcing yet** (2506.08009; CausVid 2412.07772): needs a few-step student, backprop through its own rollout and a
  clip-level distribution loss with a teacher. It is stage 2: the step that makes it fast also closes the remaining gap.
- **Gap that remains:** live, past states are the sampler's own, not noised truth ("matching historical noise levels alone
  does not ensure matching state distributions", 2609.22283). I expect less drift than today, not zero; section 5 measures it.
## 3. Inference (live)
- Tick n: x = (z_drive + z_n)/sqrt(2) as trained; DDIM 25, CFG 2 (signed settings). At step j, block l, branch cond / uncond:
  attend over the cached features of the last 7 ticks AT THE SAME STEP j, plus self; then store own. Emit after 25 steps.
  Hard break: clear the cache. Zero look-ahead. Equals joint causal denoising of the drive, so it has an exact unit test.
- **Cost per frame = the image model's**: 50 UNet passes (+ est. 5-10%), ~2.8 s per tick on a 2080 Ti (image model: 22 s /
  8 ticks). Causal yes; 10 fps no, and neither is the image model today. Cache 4.0 GB per stream (fp16): 2 streams per 2080 Ti.
- **Real-time = stage 2, after the gate:** distil to 2-4 steps without CFG (CausVid / Self Forcing; Live2Diff reports 16 fps
  at 512x512 on a 4090). At ~55 ms per pass: est. 4-8 fps on a 2080 Ti, 10 Hz on an A6000-class card.
## 4. Changes, budget, gate
- `mrv/video.py`: own q/k/v attention (scaled_dot_product_attention) in place of nn.MultiheadAttention; RoPE by tick; band mask
  (j <= i, i - j <= 7); stream mode (one new frame per stream, ring cache keyed by step, block, branch); `reset_cache()`.
  Tests before training: fresh blocks = A (max |diff| 0); stream = joint causal on 24 ticks (fp32, max |diff| < 1e-4).
- `scripts/train_video.py`: `--init runs/longA/ckpt_20000.pt --window 8`; recipe otherwise unchanged. Add a 500-update probe
  on 8 fixed dev clips at t 900 / 700 / 300: loss with blocks on, off, past from another drive, own cameras swapped.
- `scripts/video_rollout.py`: tick loop with the cache in place of windows; lockstep `--batch 2`; `--off`; `--swap` (lag test
  as temporal_rollout.py); `--reset_at N` (clears the cache at tick N). Output format unchanged.
- Budget: time 100 updates first (`--budget_min`). Est. 1.1-1.3 s / update -> 12k in about 4 A6000-hours, 20k in 7. A dev
  check = 12 streams on 6 x 2080 Ti, about 20 min + scoring; A's side is already rendered and scored.
- Look first at **2k** (tests, probes, one side-by-side video; no gate). Gate at **4k / 8k / 12k** vs A@20k: smaller ID / W
  reduction > 9.4%; car F1 >= A - 0.03; late gap <= +0.03. Two checks running past the F1 or drift line = stop. Pass at 12k
  -> 20k -> the 30-drive exam once. Under 3% smoother at 4k with a flat block norm = not learning, back to council.
- **"Cold" for a video model**: (a) GATE cold = blocks off on the same weights = A itself (stricter than "own cold").
  (b) Reported beside it, short-memory cold = cache cleared at tick 134: long minus short on ticks 150-169 = accumulated drift.
## 5. Risks and early warnings
1. **Copying returns through the low-noise steps** (past nearly finished there, and self-made). Warnings, cheapest first:
   swapped cameras move the t >= 700 prediction less than a wrong past does; per-tick latent change under 0.7x A's or
   falling between checkpoints (today's signature, 0.043 -> 0.021); lag after `--swap` over 3 ticks; long minus short > +0.01.
2. **Frames agree on a wrong car** (lingering ghosts with no feedback). Reported per check, not blocking (owner ruling).
3. **Not smooth enough** (the private noise half wins). Free probe at 4k: private share 0.5 -> 0.25 at inference. Trained
   lever: progressive history (older frames 1-2 steps cleaner; best row in 2609.22283, 85.60). Never a clean past.
4. **Overfit** (161 M new weights, 144 drives): watch the probe's train-vs-dev gap. **Not verified today:** Live2Diff's
   one-timestep-per-clip training; DIAMOND's window; every timing / memory figure here is an estimate until the 100-update run.
