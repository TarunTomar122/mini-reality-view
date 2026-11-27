# Astra review of vpriv at 2adc1a6

The CPU script passed **8** checks. With two private clips, both ε tensors also equalled their original per-tick draws exactly in my replay. The trainer preserves draw order, uses the same ε for corruption and FP32 MSE on ticks 1–15, and stores the training mixture separately from rollout ρ=0.95.

1. **Matched initialization is unverified.** `scripts/train_causal.py:117` constructs temporal blocks before any global seed is set. Only their output projection is zeroed. QKV and feed-forward weights are random (`mrv/causal_video.py:30`), so matching the data seed cannot reproduce vlive’s first gradients. Recover and verify vlive’s initial tensors or RNG state. If unavailable, obtain a council decision on the unmatched initialization before launch.
2. **Resume is unchecked.** The trainer compares `private_clips` and restores the saved update, but no fixture saves or resumes. Test a split after an odd update against uninterrupted training, checking the next slot, draws, batch hash and loss. Changing the private count must fail before an update. Keep old vlive resumes on its pinned trainer.
3. **The shared slot differs slightly from vlive.** At seed 0, step 0, 57,281/131,072 shared values differed (max 4.77e−7). `mrv/noise.py:14` rounds ρ to FP32 before subtraction. Compute the original scalar mixture and replace only the private slot. Assert exact ε for both slots at both parities.
4. **Complete the signed fixtures.** Compare default-mode batch hashes and losses for updates 1–3 under matched initialization, check picks/timesteps across all modes for 100 updates, read ρ=0.95 through rollout from a saved private checkpoint, and run the T7 frozen-weight check with two private clips. The current T7 uses zero.
5. **Reporting follow-up by the first check.** `scripts/noise_probe.py:33` reads only checkpoint ρ. Add a probe-only noise choice and report paired shared/private results at each check. Probe scores do not select the arm.

Launch requires valid signed 8k/12k/16k panels, no joint pass at any, and both safety bars met at 16k. Record both 16k paired estimates against off@0.95. Select one alternating private clip only if both ID and W are strictly lower, otherwise two. Start fresh from A@20k on the same slice. Only the 8k result is recorded here.

SIGN-IF items 1–4 are resolved, T1–T7 (including separate T2) and the A3 fixtures pass on the deployed hashes, Astra verifies those hashes and results, and the signed panels satisfy the launch and mixture rule above.
