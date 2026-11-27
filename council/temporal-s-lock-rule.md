# Proposed lock rule for the B / C history strength s (to be signed BEFORE any sweep is read)

PLAN-temporal: "run-time s swept over {0, 0.1, 0.3, 0.5} on development drives, then locked." Drift check 5 (both): sweep
the 4k checkpoints only; 2k is scored with the same s; write the rule before reading.

Proposal (per arm, B and C separately; 6 dev drives x seeds 0/1; scorer 764dd3ec95db, signed linking; comparator = arm A
4k independent on the same drives / noise):
1. Eligible s: car F1 difference vs A >= -0.02 (point) and road IoU difference >= -0.01 (point); late LPIPS gap <= +0.02
   (point, 5 contiguous dev drives); >= 5-tick ghosts per run not above A's by more than 1.0 (point).
2. Among eligible s, pick the one with the lowest mean of the relative ID and relative W errors vs A on shared
   transitions; ties within 0.01 -> the smaller s.
3. No eligible s -> lock s = 0.3 (the training-cache value) and record that the arm failed its dev screen; it is still
   scored on the decision set (failing is allowed; no adoption without passing the signed gates there).
Dev uses point estimates only (6 drives are too few for intervals to decide); the decision set applies the signed gates
with intervals.
