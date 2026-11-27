# AGREED — Astra, causal video round 3
At ρ = 1, subtracting two training ticks cancels the noise exactly, while the rollout starts with zero difference. That creates a shortcut that could reward copying.
Keep one diffusion level per clip. Use εᵢ = √ρ ε_clip + √(1−ρ) εᵢ,private in training and the same ρ in rollout, with A’s shared draw per drive/seed and an independent draw keyed by drive/seed/absolute tick.
Before update 1, render `--off` at both 0.5 and 0.8 on the six dev drives, seeds 0/1. Choose 0.5 if both ID and W are ≤1.20× their respective signed A@20k values. Otherwise choose 0.8 if it meets both limits. If neither qualifies, return to council before training. Exclude 1.0 from selection.
Keep signed A@20k as the smoothness and car-F1 comparator. For the late gap, cold is frozen A@20k with temporal blocks off at every tick, rendered once at the selected ρ with matched cameras and noise. Require candidate minus cold LPIPS-to-key ≤+0.03 on eligible ticks 150–169.
Average ε-MSE over ticks 1–15. Tick 0 supplies history. Its own loss has zero temporal-parameter gradient, so including it would scale the update gradient by 15/16.
T2 passes when `--off` at ρ = 1.0 reproduces one dev drive’s signed A latents with max absolute error <1e−2 using matched noise, dtype and batch layout.
T5 passes when replacing cameras and private draws at ticks 30–59 of a 60-tick rollout leaves ticks 0–29 bit-identical. Hold the shared draw fixed.
T6 passes when `--reset_at 20` clears every ring before tick 20 and the resulting suffix equals a fresh run starting at absolute tick 20 with identical inputs and draws.
Every 500 updates, record paired losses on eight fixed dev clips at t = 900, 700 and 300 with blocks on, blocks off, another drive’s past, and current cameras swapped. Hold targets and draws fixed. At t ≥ 700, flag copying if swapping cameras raises loss less than replacing the past.
