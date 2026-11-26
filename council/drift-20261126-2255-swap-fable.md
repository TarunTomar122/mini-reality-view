# Drift check 3 swap: Fable

1. **Counter: window centre.** A render identical to cold reads 4 by window end, so the <= 5 gate would leave one tick; the plan signed a recovery time, not a detection time. Gate on centre, report start and end. Accept Astra's scorer fix: include start index `len(ticks)-5` (`scripts/temporal_metrics.py:321`).
2. **Accept (mine), made exact:** threshold = 0.8 x cold car F1 of the donor on the same five ticks, same matcher (0.3); a window whose cold F1 is zero is skipped; no eligible window = drive excluded and counted.
3. **Keep-and-report.** E3 (`train_sd.py:245`) and X1 were trained with the same clip, so removing it breaks like-for-like. Write it into amendment 3; report the share of logged updates with norm > 1.0.
