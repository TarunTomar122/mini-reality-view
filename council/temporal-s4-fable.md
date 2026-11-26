# Temporal consistency and memory: round 4 (Fable)

**BLOCK**, one issue, text only. Same finding as Astra's round 4, reached independently.

Met in v3: "Confirmed" definition; arm A + warm start; absolute per-transition ID; non-negative placement point
differences; one noise tensor per (drive, seed); no numerical old-DiT drift comparison.

## Blocking: ID and W can still improve by dropping the hardest edge cars

v3 averages ID over the transitions each method matches itself. A method that misses the few cars that flicker most
is scored on easier cars and shows a "reduction" with no car rendered better. W still scores an erased car as
steady. The recall gate uses a point estimate (my round 3 asked for a lower bound), so a 0.02 drop passes.

**Change 1 (ID line):** "…then averaged, per candidate-comparator comparison, over the same key transitions, where
both methods match the car (IoU >= 0.3) at both ticks. W uses the car pixels of those same transitions. Report
shared-transition counts per drive; an eligible drive with none cannot pass."

**Change 2 (gate):** replace "edge-car recall drop <= 0.02 (point difference) vs the same comparators" with "edge-car
recall difference: paired lower bound >= -0.02 vs frozen independent and every adoption comparator, including A +
warm start for B/C".

With both I sign.
