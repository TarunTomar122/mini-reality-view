# Drift round 3: the 10k gate failed (27 Nov 14:10). Sign off the next step.

Read council/drift-s2-astra.md and council/drift-s2-fable.md (your merged plan), then these new signed results
(6 dev drives x 2 seeds, scorer 764dd3ec95db, temporal_compare fc7ee2b886a5, floor 0):

## Gate: longC2@10k (C@8k -> 10k on own_8k, i.e. cache 0-2k old) vs recorded stale C@10k
- late LPIPS gap (vs own cold): 0.0501 vs 0.0585 -> fall 0.0084, CI [-0.0165, -0.0019]  (rule: >= 0.02 continue, < 0.01 stop)
- early gap 0.0146 vs 0.0149; car F1 0.379 vs 0.383; recall 0.372 vs 0.390; ghost_ge5 14.75 vs 15.25; ID / W equal.
- longC2@10k vs its own cold: late gap +0.050 [0.032, 0.068], F1 0.379 vs 0.415 [-0.059, -0.013], ghost_ge5 14.75 vs 3.9.
- longC2@10k vs A@10k: ID 0.074 vs 0.101 [-0.041, -0.012], W 30.7 vs 32.4, F1 0.379 vs 0.454.
So: with a fresh cache the drift still goes from +0.002 (C@8k) to +0.050 in 2k updates. Staleness is at most a small part.

## Probes from earlier (stale C, inference history noise s; memory on vs same-weights cold)
C@10k: s 0.3 +0.059 / F1 0.383 vs 0.421; s 0.5 +0.027 / F1 0.405; s 0.7 +0.008 [0.003, 0.016] / F1 0.401, ID 0.099 vs
0.113 (-13%), W 30.0 vs 31.7 (-5%), ghost_ge5 7.1 vs 4.9. C@15k: s 0.3 +0.119; s 0.5 +0.095. C@8k s 0.3: +0.002.

## State
longC2 is still training (at ~14.5k, live cache own_10000 in use, own_12000 rendering; refreshes take ~30 min so training
waited ~5 min once at the 4k age limit). Nothing else on the A6000. Owner cap 16 cluster jobs.

## Proposed next step (Fable s2 step 6 / Astra s2 step 5): agree, object, or amend
1. Stop longC2 now (gate rule says stop; Astra's "two breaches" wording would need its 15k check, ~45 min more).
2. Second lever, forked from C@8k with the live cache (same code path as longC2): arm C rows 6-11 (true previous key)
   get s' = 0.3 + s x 0.4/0.7 (same draw) and an extra 40% history dropout from a separate seeded stream; own rows 0-5
   unchanged (s ~ U[0, 0.7]); everything else identical. 10k gate again: late gap <= +0.03 and F1 within 0.03 of own cold.
3. Open question: the probe says inference s 0.7 alone gives +0.008 at stale 10k with smoothness kept vs cold. Should the
   decision s (locked 0.3) be re-locked higher (e.g. 0.5 or 0.7, via the signed s_lock procedure on dev) together with
   lever 2, or tested alone first on longC2@10k (render at s 0.5 / 0.7, ~25 min on 2080 Tis, no training)?
Reply in under 250 words: agree / object per point, and the exact rule for the next gate. Write council/drift-r3-<name>.md.
