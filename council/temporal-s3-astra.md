BLOCK

1. **Edge-car ID can reward a frozen car.** If the key changes by 0.20, a faithful render scores 0 and an unchanged crop scores −0.20. Take the absolute difference between render and key cosine distances **per track transition**, before averaging. Score every eligible key track using its key-defined crop even when the render misses. Compute percentage reductions only for positive reference scores.

2. **The placement gate allows a confirmed regression.** A car F1 difference with a 95% interval of [−0.019, −0.011] currently passes despite the brief’s placement requirement. Require nonnegative point differences for both car F1 and road IoU as well as the stated lower bounds. For B/C, apply these checks against both A and the frozen comparator used for adoption.

3. **Sampling noise can confound the history comparison.** The merged plan fixes history noise for B/C but only explicitly fixes sampling noise for frozen independent. Specify one sampling-noise tensor per drive and seed, reused across ticks, methods and cold rerenders. Draw history corruption from a separate stream. Keep fresh per-frame noise exclusive to the metric check.

4. **The historical drift comparison uses a different quantity.** The DiT’s +1.8 was last-five minus first-five rollout L1 (30.3 − 28.5; PROGRESS.md, line 302). Keep the rollout-versus-cold gates. For a numerical comparison, reproduce the original sampling cadence, windows and RGB scale and label the dataset difference. Otherwise drop the numerical comparison.
