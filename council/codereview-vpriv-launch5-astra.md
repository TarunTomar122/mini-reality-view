# Astra launch review of `ec8e913`
Keep launcher 3755984 held. Local mocks reproduced two remaining failures.
`slurm/vpriv_checks.sh:21–25` calls sbatch once and rejects zero or two distinct IDs. If either lookup fails and the other returns one ID, the watcher adopts it. Require both lookups to succeed before counting distinct IDs.
`slurm/vpriv_score.sh:21–35` now checks lock acquisition, the stop marker and the continuation intent. A matching decision permits retry, and a pending intent blocks resubmission.
`scripts/a1_decide.py:49` accepted stored JSON `"stop": 1` for a recomputed `true` and exited 0. The score script reads `1`, misses its `True` branch and can resume after two safety misses.
Use type-sensitive decision comparison and require a JSON boolean before branching. The numeric-stop retry must fail before submission.
SIGNED: no
