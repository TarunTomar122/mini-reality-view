1. **Sign (b), k = 1, before reading B/C decision results.** “For decision and confirmation, assess ID and W separately for each comparator. A drive qualifies if the comparator matches at least one eligible transition at both ticks with a finite metric value, pooling seeds 0 and 1. Each qualifying drive must have at least one same-seed shared transition with finite values for both methods or the comparison fails. Report eligible, comparator-matched and shared counts by drive and seed, including excluded drives. Retain all drives for applicable recall and other gates.”

2. **Rank by the worse ratio.** Minimize max(ID/ID_A, W/W_A) on shared transitions versus A. Adoption requires both metrics to improve.

3. **Choose the largest s within 0.01 of the minimum score.** I revise my earlier vote. Larger s adds more noise to the previous frame and reduces its signal weight. At s = 0, history is clean.
