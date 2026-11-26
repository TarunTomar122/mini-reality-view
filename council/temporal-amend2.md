# Amendment request 2 to PLAN-temporal.md (contiguous clips), with data

Amendment 1 (50 ms) is applied. Astra and Fable asked for the count of drives with an unbroken >= 190-tick run before
training. Measured (data/temporal_manifest.json, rejects = any camera > tolerance, or the same six files as the
previous tick):

| tolerance | group | drives | >= 190-tick run | >= 150 | median longest run | median ticks rendered |
|---|---|---|---|---|---|---|
| 50 ms | train | 144 | 0 | 0 | 44 | 184 |
| 50 ms | dev | 6 | 0 | 0 | 35 | 185 |
| 50 ms | decision | 30 | 0 | 1 | 37 | 183 |
| 50 ms | final | 24 | 0 | 0 | 40 | 186 |
| 55 ms | decision | 30 | 0 | 1 | 49 | 183 |
| 55 ms | final | 24 | 0 | 1 | 44 | 186 |

Two causes: (1) the MetaDrive replay ends before the last keyframe, so drives have ~18.4 s (median 184 ticks); 190
ticks is impossible for nearly all drives. (2) The ~5% rejected ticks are spread through each drive, so breaking the
sequence at every reject leaves ~4 s runs. A looser tolerance does not fix this.

Proposed change (training unchanged):
1. Evaluation rollouts run through rejected ticks: the model is fed the nearest bundle there anyway (with the
   history chain unbroken), but rejected ticks are excluded from every metric and from the flicker transitions; a
   rollout still breaks at a repeated bundle or a gap in the render. Training pairs still require both ticks accepted.
2. Drift: "late gap" = last 2 s vs the same frames cold, on rollouts >= 17 s (170 ticks) instead of >= 19 s; early gap
   from tick 5 as before. Report the per-drive rollout length and the share of rejected ticks inside each rollout.
3. The final-set confirmation needs >= 12 qualifying drives as before (expected: ~all 24 at 17 s).

## Measured after Fable's conditions (rollout breaks only at a repeated bundle, a render gap, or any camera > 100 ms)
| group | drives | >= 170-tick rollout | median longest rollout | mean rejected (> 50 ms) ticks inside qualifying rollouts |
|---|---|---|---|---|
| train | 144 | 85 | 181 | 8.5 |
| dev | 6 | 5 | 185 | 12.8 |
| decision | 30 | 22 | 181 | 12.0 |
| final | 24 | 18 (>= 12 needed) | 185 | 10.2 |
