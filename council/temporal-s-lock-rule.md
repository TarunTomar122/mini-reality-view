# Lock rule for the B / C history strength s (signed in substance by Astra and Fable, drift check 6; written BEFORE any
# sweep is read). Sources: council/drift-20261127-0155-*.md and -swap-*.md.

Note on s: s is the noise level on the history input (sqrt(1 - s^2) * prev + s * noise); larger s = weaker history.

Per arm (B and C separately), 4k checkpoint only, 6 dev drives x seeds 0/1 (both seeds on all six required), scorer
764dd3ec95db, signed linking; comparator = arm A 4k independent on the same drives / noise. Candidates s in {0, 0.1, 0.3, 0.5}.
1. Eligible s (point estimates on dev): car F1 difference vs A >= -0.02; road IoU >= -0.01; edge-car recall >= -0.02;
   late LPIPS gap <= +0.02 and late car-F1 gap >= -0.03 (contiguous dev drives); ghost growth not above A's; >= 5-tick
   ghosts per run not above A's by more than 1.0; positive reference errors; no counting dev drive with eligible but no
   shared transitions vs A (drive-counting rule: amendment 3, below).
2. Rank eligible s by max(ID / ID_A, W / W_A) on shared transitions (the worse of the two; no cancelling).
3. Ties within 0.01 of the best score -> the LARGEST s (weaker history).
4. No eligible s -> lock s = 0.3 (the training-cache value), labelled "failed dev screen"; the arm is still scored on the
   decision set (failing allowed; adoption only through the signed gates there).
5. Lag is not used. The locked s scores 2k, 4k and every comparator; "A + warm" = warm 0.7. The full sweep table is
   committed together with the lock.

# Amendment 3 (drive counting for ID / W) - agreed: option (b); OPEN: threshold k (Astra 1, Fable 3) and Fable's floor
For each comparison, and for ID and W separately, a drive counts only if the comparator alone matches at least k of its
eligible transitions at both ticks with a finite value (seeds 0 and 1 pooled); the list is fixed from the comparator's
scores before the candidate's are opened. A comparison with no same-seed shared transition on a counting drive cannot
pass. Non-counting drives are named with eligible / comparator-matched / shared counts by drive and seed, left out of
ID / W only, and kept in recall and every other gate. [Fable, open: fewer than 20 counting decision drives (12 final) ->
that comparison cannot pass.] Known data: against independent, scene-0036 (0 matched) is out, scene-0016 (8) in, for
any k in 1..8.
