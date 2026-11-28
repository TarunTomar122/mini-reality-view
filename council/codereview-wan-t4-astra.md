# Astra T4 · Wan S1 production sign-off
I verified all ten evidence-file hashes and all three scorer hashes against the local freeze. I rely on the reported results for job 3757153.
Its reported ALL PASS releases the 2,000-update overfit. The forecast covers 21,652 updates against the 13,200 minimum with one panel reserved.
I accept gc0_m1. Its worst relative tensor-delta difference against gc1_m1 was 0.0026, within the 0.05 limit. Micro 2/4/8 each failed on two of 643 tensors and stay excluded.
Batch-shape effects in bf16 are plausible. The comparison restores Adam state after the sweep, so it does not establish a first-step sign-normalization cause.
The frozen trainer checks the selected setting, evidence identity and live ledger forecast at launch. Both gate results are pending.
SIGNED (overfit / production): yes | if the 2,000-update overfit and resume_test both record PASS for the same frozen gc0_m1 gate key and the existing launch checks pass
