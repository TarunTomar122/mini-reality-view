# Car map, round 3 (Fable): BLOCK, one issue

Everything up to and including X1 / X2 I sign as written: frozen E3 control, preflights, controls, oracle test,
train-vs-held-out rule, stop rules.

## Blocking issue: X3 placement is scored where the map is freshest
X3 inherits the temporal placement score: car F1 on every fifth accepted tick (`scripts/temporal_metrics.py:175`,
`n_ % 5`, counted from the first usable tick). Keyframe k sits at tick 5k (`scripts/temporal_manifest.py:5`), and
the detector refreshes at keyframes. So the X3 bar and the "gain over memory alone" are measured mostly on refresh
ticks, and after an offset reject at an arbitrary, unreported phase. The four in-between ticks, where the map is
held or rolled forward up to 0.4 s, barely enter the score. P4 does not cover this: it scores the map, not the
render, on development drives the detector was trained on. For the memory round this did not matter (no 2 Hz
input); for a 10 Hz map claim it overstates placement.

**Exact change**, X3 decision rule, after "positive paired gain over memory alone;" insert:

> placement for X3 is scored on all accepted ticks (already segmented, no extra cost) and reported by ticks since
> the last detector refresh (0-4); the car F1 bar and the paired gain over memory alone must both hold on the
> non-refresh ticks (phases 1-4 pooled), with the refresh-tick figures reported beside them; the lit ceiling is
> re-measured on the same ticks;

With that line I sign.

## Non-blocking (one line each)
1. P2 / P3 hull tests: state score >= 0.3, as for the 0.417 they are compared with (the cache holds >= 0.2).
2. X3 with the base unfrozen: require the gain to exceed arm A's 4k -> 6k change, not only quote it.
3. Write-up states that the final 24 are used a second time (first: temporal confirmation).
