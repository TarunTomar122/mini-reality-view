# Next video step: plan (Fable)

**Pick: option 1, a joint multi-frame model in which every frame is equally noisy and no finished frame is ever fed
back.**

**Why.** Warm start and B fail the same way: a finished frame is treated as evidence, so a wrong car is copied forward.
In joint denoising nothing cleaner than the cameras exists to copy and nothing accumulates, so drift has no mechanism.
Option 2 keeps the feedback loop. Options 3 / 4 steady the inputs, but (my reading, untested) the jumps come from the
sampler amplifying small input changes.

**Build** (E3 fully frozen: UNet, LoRA, adapter)
- 16 temporal blocks, one after each spatial transformer: GroupNorm, per-pixel self-attention across frames (8 heads,
  sinusoidal frame position), feed-forward, zero-initialised output, residual. Frames pass through the 2D UNet as a
  batch, so each keeps its own camera tokens. Random init, not AnimateDiff weights: step 0 must equal E3
  (max |diff| 0 on 16 renders).
- Clips: 16 consecutive usable ticks (1.6 s), 144 train drives. Batch 2 clips, one timestep per clip, epsilon loss,
  AdamW 1e-4 (500 warm-up), clip 1.0, bf16, camera dropout 10% per whole clip.
- Noise (PYoCo, arXiv 2305.10474): eps_i = (eps_clip + eps_i') / sqrt(2). Fully shared noise would start every drive
  as a frozen picture; fully independent is our measured worst case.
- 12k updates, checkpoints 2k / 4k / 8k / 12k, one A6000, about 1.2 s / update (time 100 first).

**Inference.** Per contiguous segment, all ticks start from noise (shared half = the signed per-drive tensor, private
half from a seeded per-tick stream). Each DDIM step: windows of 16, stride 8, predictions averaged in the overlaps.
DDIM 25, CFG 2, about 2.5x independent cost. **Not causal:** a frame sees cameras about 1 s ahead. Fine for rendered
video, not for live use; the owner should know.

**Evaluate** (scorer 764dd3ec95db, compare fc7ee2b886a5, comparator = E3 independent, already scored)
- Dev controls: E3 with the mixed noise and blocks off (isolates the noise change); the zero-init checkpoint.
- Dev screen (6 drives, seeds 0 / 1) per checkpoint, locked mechanically as in s_lock.py: eligible if car F1, recall,
  road, ghost growth, >= 5-tick ghosts and late gaps are inside the gates; then smallest max(ID ratio, W ratio);
  ties -> fewer updates.
- Late gap: cold reference = E3 independent itself (stricter than own-cold; the base is frozen).
- Decision: 30 drives x 2 seeds, count list council/count_indep_dec_k3.json, floor 20, the brief's gates. Then the 24
  final drives once. Lag (--swap) reported.

**Early stop.** At 4k, dev ID and W both under 5% better -> stop; next round = input memory (option 4). Car F1 below
-0.03 or lingering ghosts above +2 per run at a checkpoint, and no better at the next -> stop.

**Risks.** (1) Agreement across frames may steady a wrong car (more >= 5-tick ghosts without feedback); the ghost gate
decides. (2) W -15% is the hard gate: B reached -5 to -9%.

**Cost.** Training 4-5 A6000-hours; dev 6 x 12 rollouts; decision 60, final 48; at most 8 jobs.

**If it passes:** same recipe to 40k, then unfreeze the LoRA at 1e-5.
