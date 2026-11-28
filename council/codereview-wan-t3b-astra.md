# Astra T3b review of 9b8c3f1
- **8 closed.** With rehearsal end at 200, a failed read at 300 kept `t_valid=100` and failed coverage. A successful read at 310 passed.
- **12 closed.** Panel intake accepted literal `True` and rejected `False`, a missing flag and integer `1` in otherwise identical bindings.
- **Neon C remains open in the recorded forecast.** The rehearsal separates fill and hit rates. For a shared donor, the renderer cuts the swap to `n` ticks, but `panel_render_h` charges both seeds and the cache fill at the receiver’s full length (scripts/wan_rollout.py:270, mrv/wan_plan.py:639).
- **Local CPU repro.** Six single-segment drives of lengths 200, 160, 160, 160, 160, 160 pass donor preflight. At 1 s/hit tick and 0.2 s/fill tick with zero load, the plan costs 10,664 s. The forecast records 10,752 s, charging 40 unused ticks twice for rendering and once for encoding.
- Price swaps at their planned lengths and verify this case in `forecast_final.panel_h_total`. The rehearsal already records both rates needed for the correction.
SIGNED (timing / overfit / production): no | no | no
