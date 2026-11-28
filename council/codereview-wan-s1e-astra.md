3. Closed at df48018. `--window=1` reaches preflight and all three rollout call sites as width 1. The option mirror test passes. CPU suite: `WAN_TESTS FAILS=0`.
4. The default trainer path rejects legacy or incomplete forecasts, and the forecast writes a null total for incomplete plans. Two production paths still under-reserve.
4a. Both launchers forward trailing arguments (wan_teacher.sh:93–94). On CPU, `--reserve_hours 0` skipped forecast validation entirely (train_wan_teacher.py:136), and `--reserve_panels 0` reduced a measured 2.2 h to zero. Require at least one measured panel in train and extend.
4b. Forecast fixes width/swap at 5/50 (wan_forecast.py:39). CPU repro: A–F lengths 200/200/180/180/180/180, ticks from 0, all keys present, no hard breaks, only B ticks 50–56 rejected. Both plans pass: width 5 needs zero shared references, width 2 needs one 180-tick reference at two seeds. Bind the reserve to the panel settings it prices.
Guide provenance: restored builder matches 415c470 byte-for-byte, combined code SHA is `f6587caf55d9`, and all four named consumers plus fit_depth_scale enforce split membership.
SIGNED (production): no
