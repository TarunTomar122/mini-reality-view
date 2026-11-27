# Fidelity check (Fable)

**Checked.** No bug: training and rollout feed history the same way. B's history weights are not a hidden warm start (overlap with the input conv 0.11 = chance). GameNGen has **no training on its own outputs**.

**1. Ranked deviations**
1. **History is never wrong in B's training.** Same in GameNGen, but there history is the only truth and its errors look like noise. Ours: cameras are the truth and errors are whole fake cars (E3 draws 1.9 per tick). Noise cannot imitate those, so B learns "keep the last frame": ghosts, lost cars and drift together.
2. **C's dose is tiny.** 896 own frames reused 20 times each, at most 6 ticks deep, s = 0.3 only. C's history weights ended 25% smaller than B's (1.16 vs 1.54).
3. **Scale.** 64k samples vs ~90M; LoRA plus one conv vs full UNet. But more B training strengthens copying. Prediction: B 12k has fewer jumps and no fewer ghosts than 4k.
4. **One past frame vs 64.** Costs motion, not ghosts.
5. **Minor.** Level fed as a plane (weights 0.15 vs 1.54). Their guidance is on history, ours on cameras: in our favour.

**Fix to test first: C at full dose.** Own-history rows from whole-drive own rollouts (`temporal_rollout.py --split train --method hist`, s 0 to 0.5). Fork B 4k, 48 drives, refresh at 6k, stop at 8k, compare with A and B 8k. 8 GPU-hours per refresh (2.9 s per frame, 2080 Ti). Read C 4k dev first: ghosts not below B's means do not pay.

**2. Car map (record only).** ControlNet trains a full encoder copy (~360M weights) on a full-resolution hint, converging after thousands of steps. Ours: one conv (26k weights) or a 0.4M branch, map pooled to 32x64, 16k samples at the gate.
