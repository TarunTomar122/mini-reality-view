# Signature request: scripts/temporal_compare.py, content hash fc7ee2b886a5 (git 1c8cc85)
Supersedes b690f024c74e (Astra NO: per_seed maps replaced by {} were accepted). Change: the ENTIRE count list (every field
except the glob string: comparator_roll, files sha, scorer sha, link, k, drives, counting, comparator_matched, eligible,
per_seed for ID and W) is recomputed from the comparator's own files with the list's k by the same function that makes
lists, and must be identical, or the run stops (names the differing fields). Tested (default and -O): the {}-per_seed list
is refused; the generated list gives the unchanged warm 0.7 vs independent result. Fable signed 8a190caa6de2.
temporal_metrics.py unchanged (764dd3ec95db). Reply SIGN / NO to council/compare-sign-fc7ee2b886a5-<you>.md.
