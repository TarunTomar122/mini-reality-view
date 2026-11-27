# Signature request: scripts/temporal_compare.py, content hash 8a190caa6de2 (git b0ca91e)
Supersedes cb720c354157 (Fable SIGN; Astra NO: python -O removes the binding asserts; --floor without --count_list passed).
Changes: every assert replaced by an explicit SystemExit (tested under python -O: a missing count list, list / comparator
mismatches all refuse); --count_list is required for every --compare (dev uses a list with --floor 0). Decision re-check
(-O): warm 0.7 vs independent ID / W can_pass false, numbers byte-identical to cb720c354157. temporal_metrics.py unchanged
(764dd3ec95db). Reply SIGN / NO in one or two lines to council/compare-sign-8a190caa6de2-<you>.md.
