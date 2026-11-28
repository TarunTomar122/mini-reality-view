# Astra launch review of 70804d6
Keep launcher 3755958 held. All 23 deployment hashes verify locally. The builder reports suite 3755915 is still running. Cluster state and the dry run were not independently verified.
The four pulled panels support two private clips. The smaller reductions at 8k/12k/16k are −0.511/−0.480/−0.577. At 16k, F1 −0.0105 and late gap +0.0016 pass safety. Own-cold ID +0.0453 and W +2.8717 select two.

1. The trigger checks metrics paths, drive/seed keys and candidate checkpoint provenance. It compares and hashes the same metrics copies. The count list is still read live at `scripts/a3_trigger.py:76` and hashed later at line 87. Snapshot it once and use that copy for both.
2. Separate launch and dry-run directories, exclusive record creation and checked receipt writes resolve the overwrite finding.
3. The launcher checks the queue and existing marker under the lock. At `slurm/launch_vpriv.sh:48–53`, interruption after Slurm accepts the job can leave no marker. If the job exits before writing args, a retry can submit again. Persist a pending intent before submission and require reconciliation of an ambiguous receipt before retry.
4. The four final log lines are bound to their supplied job IDs. Lines 34–36 accept a one-line manifest containing any reviewed entry. Require exact hash/path sets for the suite (13 entries) and each fixture (6 each), then verify all four named results, including the final suite pass.
5. `slurm/vpriv_checks.sh:10–37` can count a failed queue query as zero, ignore a failed submission, and mask a failed comparison behind successful LODO. Verify queue and submission status, both signed outputs and panel completeness. Assign an operator or gate to act on the earliest pass or two consecutive safety misses from 8k while training can still be stopped.

SIGNED: no
