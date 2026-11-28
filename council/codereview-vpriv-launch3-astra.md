# Astra launch review of 7df6691
Keep launcher 3755966 held. The builder reports suite 3755915 still running. This review used local files and mocked scheduler responses only.
Items 1, 3 and 4 close. The trigger compares and hashes one count-list snapshot. PENDING precedes training submission under lock. All 25 local hashes verify, and the launcher checks the exact 13/6/6 manifest entries and four job-specific final lines.
The pulled panels still select two private clips. No gated checkpoint passes. At 16k, F1 −0.0105 and late gap +0.0016 meet safety. Own-cold ID +0.0453 and W +2.8717 select two.
Item 5 remains open. Both signed compares now precede the decision, incomplete drive/seed panels fail, and LODO runs last.
1. At `slurm/vpriv_checks.sh:33` each score depends only on its own metrics. I reproduced a 12k decision exiting 3 when 8k had yet to finish. Chain 12k and 16k decisions to the preceding successful gated decision as well as their own metrics.
2. At `scripts/a1_decide.py:45–50` cancellation targets every job named vpriv and ignores the return code. With two misses and a mocked scancel failure, the script recorded a stop and exited 0. Target the SUBMITTED job ID and verify cancellation or terminal state before recording success.
3. The launcher requests 16k uninterrupted. On the control run, 12k was still scoring after 16k finished (`PROGRESS.md:28`). A 30-minute check cannot enforce that stop. Hold further updates at 12k until the 8k and 12k decisions permit continuation.
4. At `slurm/vpriv_checks.sh:16` an ambiguous submission is retried up to five times. A lost receipt after acceptance can duplicate jobs and exceed the 16-job cap. Stop for reconciliation on an ambiguous receipt before resubmitting.
SIGNED: no
