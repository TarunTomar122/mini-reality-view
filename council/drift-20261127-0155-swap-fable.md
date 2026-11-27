# Drift check 6 swap: Fable (written without reading Astra's swap)

**1. Option (b), k = 3.** Replace the sentence at `PLAN-temporal.md:51-52` with:

"Shared-transition counts are reported per drive. For each comparison, and for ID and W separately, a drive counts
only if the comparator alone matches at least 3 of its eligible transitions (seeds 0 and 1 summed); the list is fixed
from the comparator's scores before the candidate's are opened. A comparison with no shared transitions on a counting
drive cannot pass. Non-counting drives are named with both methods' counts, left out of ID / W only, and kept in recall
and every other gate. Fewer than 20 counting decision drives (12 final): that comparison cannot pass."

The same rule applies to the s-lock (comparator A). Warm 0.7 stays rejected on its other gates.

**2. Accept** worse-of-two. Adoption needs both ID and W to fall, so the mean should never have been allowed to trade one for the other.

**3. LARGER s, and the brief's default has s backwards.** s is the noise put on the history
(`scripts/temporal_rollout.py:120`): larger s = weaker history. "Less history, closer to the frozen comparator" is the
larger s.
