# Astra production review of 442b6a2
1–2. Both CPU repros pass. Ineligible ID or W gives INVALID and suppresses the marker. Donor acceptance starting at restart 77 yields zero lag windows, and receiver rejection removes common placement frames.
3. Preflight runs first, but the two parsers disagree. With `--rollout_args="--window=1"` on 150 accepted ticks, I reproduced 15 windows at preflight (width 5) and zero under the rollout parser (width 1). Parse the width once and pass it to both (`wan_panel.py:129`, `wan_rollout.py:50`).
4. Legacy timing can strand the panel. I reproduced a 2.0 h trainer reserve from legacy `panel_h` for a panel requiring 2.2 h with one shared receiver. At 21.9 h spent, training can continue while panel admission already fails (`train_wan_teacher.py:142`, `wan_panel.py:151–157`).
4. A failed donor plan also produces a total with missing receivers omitted. Require six valid assignments and their measured total before train or extend. The publication cap recheck is present (`wan_forecast.py:39–40`, `wan_panel.py:242`).
5–7. Forecast deductions require a matching timed gate key and PASS. Training checks bound A@20k replay parity before its first update. Leg three requires a saved step of exactly 8. GPU outcomes and real donor assignments remain unverified.
SIGNED (production): no
