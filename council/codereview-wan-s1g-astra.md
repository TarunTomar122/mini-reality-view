4a remains open at `17c4875`. The label fix works, but train forwards `"$@"` after its defaults. With `--out runs/wan_timing_followup`, the trainer parses 40,000 steps with `TEST=0` and `PROD=False`, skipping the reserve, budget stops and gates.
Classify train and extend from the run mode. Give the 16-update resume test an explicit, bounded exemption.
4b is fixed for new forecasts. The timed rollout uses `PANEL_WIN`, records its width, and the forecast refuses a missing or different width before writing.
Existing forecasts still need a read-time check. The trainer passes only `forecast_final` to the reserve helper (lines 145–148), so it accepts an old width-2 forecast measured at width 5. Require the measured width to match before accepting the reserve.
SIGNED (production): no
