At 2976299, the CPU suite reports `WAN_TESTS FAILS=0`. The reserve helper passes both original S1e repros.
4a remains open. Train and extend forward `--kind` as a ledger label (slurm/wan_teacher.sh:95–96). Changing that label skips both forecast validation and the per-update budget check, leaving the default reserve at 0 h (scripts/train_wan_teacher.py:128,138–150,618). Enforce the reserve from the production run mode independently of the label.
4b remains open. With `PANEL_WIN=2`, the timing rollout still measures width 5, then the forecast labels those rates width 2 (slurm/wan_teacher.sh:67–80). An uninterrupted 81-tick sample takes 5 windows at width 5 and 11 at width 2.
Pass the width into the timed rollout, persist its measured width, and reject a different forecast width before writing the total. The donor plan already uses the requested width and swap position.
SIGNED (production): no
