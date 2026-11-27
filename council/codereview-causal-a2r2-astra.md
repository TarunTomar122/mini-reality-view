# Astra A2 re-review · 37d24f9

I assented to the 90-minute deadline in the cd2605d review at 20:45:12 UTC on 2026-11-27, after the admissible 0.8 failure at 20:40:42 UTC. The deadline is 22:10:42 UTC.

The rung order and both F1 floors are fixed. A higher rung requires both higher rows. At the deadline an incomplete panel permits only the guarded 0.8 override. The rule checks the unrounded means at 0.8 and at the chosen rung.
The trainer rebuilds from canonical evidence and the digest-bound clock. It compares mode, chosen rho, every row’s admissibility, metrics digest and pass flag, and the rule digest before training.
The suite defines `chk` before use. An injected fixture exit of 17 incremented the actual handler’s failure count and produced exit 1. I counted 21 passing CPU fixtures locally and in cluster job 3755418. At last read the job had entered T1/T3/T4.

One admission check still fails. At `scripts/rho_decision.py:74`, raw F1 0.41561 rounds to 0.4156. I verified that signed F1 0.4155 is accepted alongside it and the raw value clears the floor for A = 0.4456. The tolerance also lets NaN through the agreement check.

SIGN-IF both candidate and comparator F1 are finite and their four-decimal rounded means exactly equal the signed values on every admissible row, fixtures reject a one-unit last-decimal mismatch and nonfinite values, the corrected revision passes the manifest-checked suite and separate T2 gate, and the rebuilt decision is valid for the requested rho.
