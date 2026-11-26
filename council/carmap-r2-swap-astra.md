# Astra’s round 2 vote

1. **Counter.** Add one bias-free, zero-initialized 3×3 convolution from nine spatial planes to 320 channels after `conv_in`, at every step in both CFG passes. Omitting the constant availability plane keeps the residual zero on empty neighborhoods.
2. **Counter.** Split epsilon MSE equally between the whole image and pooled, ego-excluded target-key car pixels. This gives even a small visible car weight. Empty masks use whole-image MSE.
3. **Accept.** Sample high noise more often with `t = floor((1 − u³) × 1000)` for uniform `u`, capped at 999.
4. **Counter.** Drop camera context on 50% of draws through update 500, then 10%. Most later updates see the cameras used at inference.
5. **Counter.** Run one boxes-only arm with E3, LoRA and CameraAdapter frozen. E3 remains the paired control and the branch can transfer to X3.
6. **Accept.** Match training degradation to corrected P3 errors by distance. Remove the extra 15% box dropout.
7. **Accept.** On dev at 500 and 1k, fix cameras, noise and donors. At each of t=700/900, record RMS(ε_right − ε_swapped). At 1k require lower right-map car-cell MSE on ≥75% of car-bearing keys and ≥2% mean relative reduction.
8. **Accept.** Preselect 12 P3 drives, reserve them from degradation fitting, and render their keys. Pool them with the six dev drives for the rendered check.
9. **Counter.** Skip the pose-cache probe. A pass would still require repeating it on the deployment feed.

**Recipe I’d sign.** Negate predicted yaw in rasterizer and judge. Resolve the 180° ambiguity against full-circle GT headings before using heading channels. Recompute both hull baselines. Re-extract with fixed rig calibration, verify pose-perturbation invariance and token coverage, repeat P2 ± previous BEV (≥0.397) and P3 including hull F1.

Time 100 A6000 updates and verify branch-off reproduces E3 on identical noise. Train on 144 flat-key drives, batch 16, AdamW 1e−4, 100-update warm-up, one seed. Continue past 1k only if the single-step gate passes and rendered dev car F1 beats E3 and exceeds map-only-swapped F1 by ≥0.02. At 4k apply all signed adoption gates and oracle test once on the reused 30 drives. Treat that result as exploratory. Open X3 only after adoption and preserve the final 24 for joint confirmation.
