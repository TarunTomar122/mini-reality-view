# Temporal consistency and memory: round 3 (Fable)

**BLOCK**, on three points. Each is a text change; none adds training. With them I sign.

## Blocking

1. **Both primary scores still reward erasing edge cars.** ID crops the render at the key box whether or not a car
   is there. A car missing at both ticks gives road against road, so a near-zero distance, and the signed
   "render minus keys" then goes negative. W has the same hole, and the pooled car F1 gate (-0.02, all cars, every
   fifth tick) is too loose to catch a 15% loss of edge cars.
   **Change:** leave pairs missed at both ticks (no render car with IoU >= 0.3) out of ID, and keep pairs missed at
   one tick. Add a gate: edge-car recall (outer thirds, all ticks) lower bound >= -0.02 against frozen independent,
   and against arm A for B/C.

2. **"Confirm" is undefined.** If the 30 drives pass and the 24 are mixed, the plan leaves the call to us afterwards.
   **Change:** "Confirmed = on the 24 drives, ID and W both improve against the same comparators with intervals
   excluding zero, and every gate passes; the 15% size is reported, not required. Otherwise nothing is adopted and
   the result is reported as unconfirmed."

3. **Missing comparator: arm A with the warm start.** B is tested against A cold and against the frozen warm start
   separately. If more updates and the warm start add up, B can beat each and still lose to the two together, and
   we would adopt a new input branch for nothing.
   **Change:** run arm A with the locked warm strength (evaluation only); B/C must also beat it under the same rule.

## Non-blocking

- Compare trained arms at equal update counts: 6k decides, 4k is reported.
- Leave the 6 development drives out of the keyframe singles.
- Say in the write-up that each arm is one training run, so run-to-run variation is unmeasured.
