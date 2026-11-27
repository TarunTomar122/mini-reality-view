# Astra A2 code review · cd2605d

Keep launcher 3755408 held. The admissible 0.8 result (ID 1.325×, W 1.143×) activates A2.

I accept the 90-minute deadline in this review, after the 0.8 result. My original swap signature covered four admissible failures only. Record the timing of this assent in A2.
Before 22:10:42 UTC, wait for both higher rows. At or after that time, an incomplete ladder goes only to a recorded 0.8 override if 0.5 and 0.8 are admissible failures, off@0.8 clears the F1 floor and no available pair regresses on both ratios. A failed guard goes to council. A complete panel uses the lowest passing rung, or the guarded override if both fail.

1. **Rung order — not fixed.** Lower rungs cannot be skipped. With 0.9 passing and 0.95 missing, `scripts/rho_rule.py:45–53` chose 0.9 at both 10 and 90 minutes. Supplying a 0.95 row worse on both metrics then sent that same panel to council. Require the complete panel before choosing a higher rung. Replace the contrary fixture at `scripts/test_rho_rule.py:21`.
2. **Decision clock — fixed for this run.** Cluster state records 20:40:42 UTC and matches the live 0.8 metrics digest prefix a9f68b0ef69474eb. `scripts/rho_decision.py:67–76` checks that binding and computes elapsed time. The 89.99/90.0 fixtures pass. Trainer verification remains finding 4.
3. **F1 floor — not fixed.** `scripts/rho_rule.py:50–56` checks the selected rung but accepts fallback at 0.9 with off@0.8 F1 = 0.300 and A = 0.4456, contrary to my signed 0.8 guard. Enforce both floors. `scripts/rho_decision.py:66` uses means rounded at `scripts/temporal_compare.py:221`. Candidate 0.41556 vs A 0.44564 fails at full precision and passes after rounding. Test equality and just below it using unrounded paired means.
4. **Stale overrides — not fixed.** One decision record removes the stale sidecar. `scripts/train_causal.py:53–63` trusts cached flags and elapsed minutes and skips null digests. A higher score arriving after a timeout decision but before update 1 goes unchecked. Rebuild from the four canonical evidence paths and bound clock before admission, compare mode/rho/digests, and bind the recorded override to the accepted A2 revision.
5. **Launcher dispatch — fixed. Test gate — not fixed.** Local stub submissions forwarded the correct rho and decision path in all three modes and rejected an unknown mode (`slurm/launch_vlive.sh:18–28`). The manifest covers the new files. `slurm/causal_tests.sh:8` calls chk before its definition on line 9, so fixture failures do not increment FAILS. Define it first and verify an intentionally failing fixture makes the suite fail.

All 16 CPU fixtures passed locally and in cluster suite 3755407. Its log confirmed “chk: command not found” and had reached T1/T3/T4 without a final result at last read. Re-review the corrected diff and passing T1–T7 evidence before release.

DO NOT SIGN
