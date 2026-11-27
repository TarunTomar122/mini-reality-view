# Astra review of 061b3f9

No code change is required to start a fresh production run once the three conditions below are evidenced. The current cluster results leave the launch on hold.

1. **Blocker — T1–T7 results.** Require both suites on the reviewed code to exit zero and report all named passes. A6000 job 3755043 was pending with no log. The latest 2080 Ti log, ct2-3755044.out, showed T2 starting on scene-0043 seed 0 without a comparison result.
2. **Blocker — valid rho evidence.** Both scoring jobs (3754837, 3754841) were dependency-pending and the decision file was absent. Verify the following against the completed sweep before update 1.
   - Establish that the signed scorer actually read the 12 intended full, unmodified off-renders per rho (six dev drives × seeds 0/1) from frozen A@20k, and used the signed A@20k comparator. Job commands, dependencies and immutable inputs suffice. A newer metric mtime, matching keys and a checkpoint pathname alone cannot establish that lineage. The trainer rechecks metric and comparator SHA-256 prefixes, which cover changes after the decision. See scripts/rho_decision.py lines 23–37 and scripts/train_causal.py lines 48–63.
   - For each row used by the ordered decision, require eligible ID and W, finite signed rel values, positive comparator means and both ratios outside the ±0.0005 band around 1.20. Choose 0.5 if both ratios pass, or 0.8 only after an admissible 0.5 miss and two passing 0.8 ratios. Otherwise return to council. Verified cluster scorer and comparison hashes match AGREED (764dd3ec95db and fc7ee2b886a5).
   - Exercising the actual expressions at scripts/rho_decision.py lines 39–47 gave pass / council / miss for rel 0.199 / 0.200 / 0.201. None goes to council. A literal NaN with a positive B gives rel_ok=True, borderline=False, meets=False and can incorrectly advance the choice to 0.8. Inspecting both finite values on every decision-relevant row closes this for the run.
3. **Blocker for production scheduling — timing.** Record the agreed 100-update A6000 timing with seconds per update and peak GB. T7 runs only two updates. Use a fresh production output from frozen A@20k after timing.

**Non-blocking follow-ups**

- Automate the finite-value and lineage checks above, including a render-content manifest and reconstruction of the ordered decision at trainer entry. A positive comparator mean that rounds to zero currently sends a row to council unnecessarily.
- Checkpoint hash/load now share one byte read. Rollout reuse checks receiver encoding/rays and donor name. Before reusing swapped outputs, also bind donor encoding/rays and validate both tick vectors and donor-limited length. Pin training inputs, rays and base provenance for later resumes.
- T5 records the scene used for camera lookup and the key used to seed private noise. Its assertions check the scene and salt suffix. Extend coverage to complete expected keys, donor tick IDs and usability. Align noise metadata on hard-break ticks and document positional intervention controls for gapped tick IDs.
- Account for logging and the terminal save/probe in end-to-end timing. Measure ring-copy cost and 2080 Ti camera-to-display latency before a live-performance claim.

SIGN-IF T1–T7 pass on 061b3f9, the signed rho evidence satisfies condition 2 and passes trainer hash checks, and the 100-update A6000 timing records step_s and peak GB before production scheduling.
