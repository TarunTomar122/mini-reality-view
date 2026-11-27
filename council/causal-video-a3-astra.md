Before 8k, keep vlive at ρ=0.95 under A1. Its 2k/4k panels and all teacher-forced probes are diagnostic only.
Activate A3 only after vlive completes 16k with valid signed panels, no joint pass at 8k/12k/16k, and both safety bars met at 16k. An earlier safety stop, broken run or invalid panel goes back to council.
Fresh arm from signed A@20k with freshly initialized temporal blocks and optimizer. Match vlive’s seed, batches, schedule, frozen weights, K=4, ρ=0.95 and rollout for an equal-update comparison.
For each clip let rᵢ = ε̂ᵢ − εᵢ in FP32. Use M = mean(rᵢ²) over ticks 1…F−1 and D = mean((rᵢ − rᵢ₋₁)²) over i=2…F−1, averaging all latent elements within each term and excluding warm-up tick 0.
Loss = mean over clips of w(t)·(M + D), with SNR = ᾱₜ/(1−ᾱₜ) and w = min(1,5/SNR), using w=1 at SNR=0. This tests the two loss changes jointly.
Re-pass T1–T7. Fixtures must reproduce legacy M with D disabled and w=1, exclude tick 0 and cross-clip pairs, cancel constant residuals in D, and match hand-computed loss and gradients for two clips with SNR 1 and 10.
Astra and Fable must agree on the amendment. Astra signs tested code hashes before launch. Neon audits evidence.
Both arms use signed A@20k on six dev drives × two seeds. Both eligible ID/W reductions must exceed 9.4%, F1 ≥ A−0.03, and late LPIPS gap vs matched own cold ≤ +0.03 on eligible ticks 150–169.
Apply A1 to A3: informational 2k/4k, gate from 8k, first joint pass to the 30-drive exam, stop after two consecutive safety misses from 8k, extend an improving run through 12k/16k subject to A1’s broken-run rule.
SIGNED: After a valid, safe 16k non-pass under the trigger above, test a fresh matched ρ=0.95 arm with adjacent-residual loss and Min-SNR γ=5 under AGREED+A1.
