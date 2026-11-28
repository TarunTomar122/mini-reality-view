# S1 timing round 1 — CONVERGED (28 Nov 10:33 UTC; Astra + Fable answers and swaps; builder merge)
Measured inputs: timing 3756964 (council/s1-timing-r1.md); reader profile 3757043 (cold 8-clip batch 12.0 s / 4.05 s with 4
threads / 2.40 s with 8; warm 0.15 s); staging profile 3757045 (cold copy + sha256 to /dev/shm: 114 MB/s with 8 streams, 113 MB/s
with 4 -> aggregate Lustre limit); data/wan_tok_v1 58 GB for 101 of 150 drives (~86 GB at 150).
1. Panels inside the 24 h: ONE (both seats). wan_forecast --n_panels 1 is the gate; the two-panel number is printed beside it and,
   if < 12,000, reported as "no spare for a second read". The second panel belongs to the conditional +12 h extension.
2. F1 staging first (both): each train / extend / timing / overfit / resume job copies the token + latent caches of its drives and
   the Wan weights to a job-local /dev/shm directory (8 streams), removed by the EXIT trap; memory request = staged bytes + 48 GB;
   every stage-in charged to the ledger; start-up measured from allocation to first update. Estimated stage-in ~13-14 min per
   start at 114 MB/s. Reader threads only as a fallback (memory not grantable, or a measured cold start > ~15 min).
3. F2 one rule: a byte is trusted only if it was hashed. One CPU job (after the S0c-drive rebuild) hashes every cache / token /
   guide / enc file of train + dev against its sidecar and writes ONE read-only manifest (path, size, mtime, sha); its sha goes into
   the gate key and beside DATA_SHA. Every file a job opens is hashed while copied and must equal the manifest. Size / mtime is a
   change alarm only (mismatch -> refuse, re-hash in a CPU job). No sampled hashes.
4. F3: rollouts on the slice with Wan loaded once per panel (state / RNG reset per run; one receiver rendered by the load-once loop
   byte-equal to the per-rollout-load render as a rehearsal); RGB + sha manifest written on the slice, slice released; scoring on
   2080 Ti jobs under their own ledger, cap 1.25 x measured scoring GPU-h per panel, reported separately; before PASS: A@20k replay
   parity on the 2080 Ti and sharded == unsharded on the same RGB, bound to device / weights / code / RGB manifest. Full panel kept
   (6 drives x 2 seeds, every eligible tick, four modes, swap, references, six valid donors); every signed bar unchanged.
5. F5: timing sweeps --grad_ckpt {1, 0} x micro {1, 2, 4, 8} with a matched-update check at fixed inputs / noise before selecting;
   --use_tokens 0 GPU time recorded as a diagnostic only. --micro and --grad_ckpt explicit in every mode, recorded in the recipe; a
   mismatch refuses.
6. F4 (2x2 token pooling) held: only if the re-time misses; then a precomputed per-camera 11 x 19 pooled cache (mean over valid
   cells, rays recomputed at true cell centres), new identities, all gates and the cost gate again.
7. Re-time pass rule (all must hold, else no overfit): full 144-drive list, cold allocation; >= 200 consecutive production updates
   (Astra; stricter than Fable's 2 x 32), forecast at the slowest of the full-run mean and the last two 32-update block means;
   queue wait <= 5% of wall and wall <= 1.10 x gpu_s per block; geometry / init parity / update-0 sensitivity PASS on the final
   recipe; staged hashes == manifest; 8 staged batches byte-equal to the same steps from Lustre; peak memory incl. staged files
   <= 85% of the request; scorer parity off the slice; six valid donors; panel-loop rehearsal byte-equal; forecast from the actual
   ledger (both timings, 2,000-step overfit with evals / verdict, three resume-test starts + 32 updates, every production start
   across 12 h jobs incl. staging, probes / saves, ONE measured slice-only panel) -> floor_12k_reachable and >= 13,200 updates.
   FAIL -> F4 + re-time; FAIL again -> owner with the affordable update count and the shortfall. The 24 h cap is not restated now.
