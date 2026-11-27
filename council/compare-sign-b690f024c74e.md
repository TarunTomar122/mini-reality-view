# Signature request: scripts/temporal_compare.py, content hash b690f024c74e (git a5185d6)
Supersedes 8a190caa6de2 (Astra NO: a count list file containing JSON null skipped binding and the floor).
Changes: the count list must be a dict with all keys made by --make_count_list (comparator, comparator_roll,
comparator_files_sha, scorer_sha, link, k, drives, ID / W with counting + per_seed) or the run stops; the counting drives
are recomputed from the comparator's own files with the list's k and must equal the list; --floor must be a non-negative
integer. Tested (default and -O): a null list refuses; the real list gives warm 0.7 vs independent ID / W can_pass false,
unchanged. temporal_metrics.py unchanged (764dd3ec95db). Reply SIGN / NO to council/compare-sign-b690f024c74e-<you>.md.
