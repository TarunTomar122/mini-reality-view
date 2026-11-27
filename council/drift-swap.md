# Drift round 2: swap (27 Nov ~12:35)

Read council/drift-brief.md, then BOTH council/drift-s1-astra.md and council/drift-s1-fable.md. Converge on ONE plan.

## Where you already agree
- Cause: the own-history practice frames went stale (C@4k frames for 12k updates); DAgger / Self Forcing need rollouts of
  the CURRENT model. The cached single-step loss is only part of Self Forcing.
- Fork point: C@8k (optimiser included); history noise s is a trust dial worth probing; 0.3 stays locked for decisions.
- Success: own-cold late LPIPS gap <= +0.03 and car F1 within 0.03 of equal-step A at the next checks.

## Where you differ (settle each)
1. **Test first or fix straight away.** Astra: a clean replay 8k->10k on own_8k (new out dir) vs recorded C@10k, decided
   before a long run (~48 A6000 min). Fable: go straight to the live-cache fork C@8k -> 20k (~2.3 h + ~1 h coding/smoke
   test), with s4 (16k->20k on own_8k, running now) as the stale control, and prove staleness by longC2@20k beating stale
   C@20k on late gap by >= 0.03.
2. **Refresh mechanism.** Astra: refresh from the exact starting checkpoint of each 2.5k stage (each stage restart costs
   ~30 min data loading on the A6000). Fable: in-process reload of the newest complete cache every 200 updates, own rows
   75% newest / 25% previous cache (separate seeded stream), refresh every 2k (12 rollout shards per refresh).
3. **Success rule.** Astra: smaller of ID/W reductions must beat 9.4%. Fable: smoothness headline positive; lingering ghosts
   <= 9.25; Fable adds freezing early warning (tick-to-tick change on the pool falling > 15% between refreshes).
4. **Fable's freezing diagnosis** (memory-on frames change 0.043 -> 0.021 per tick from 5k to 15k; gap present in the first
   2 s; true-key rows reward copying): Astra, do you accept it? It implies part of the measured smoothness at 10k is
   staleness, which matters for the success rule (a frozen picture scores "smooth").
5. **Second lever if the fix fails** (Fable: true-key rows at s ~ U[0.3, 0.7] + 40% history dropout).

## New facts since round 1
- Owner wants it fixed fast; account limit 20 jobs; one A6000.
- longC-s4 (16k->20k) started 11:54 on own_8k (log "stage to 20000 with cache data/own_8k"); s5/s6 (to 25k/30k) are HELD.
  own_12k is complete (48/48). The C@15k signed dev check (s 0.3 + cold) is scoring now.
- Probes launched now (3 shards each, ~20-30 min): C@8k cold + s 0.3 (fork-point check), C@10k at s 0.5 and 0.7,
  C@15k at s 0.5; all scored with the signed scorer against the same-weights cold run.
- temporal_rollout.py now batches (drive, seed) streams in lockstep (same noise streams; verified mean |diff| <= 0.002).

Write council/drift-s2-<your name>.md (under ~600 words): agree / object point by point, then the merged plan as numbered
steps (exact commands / code changes, order, costs, the decision rule). Reply with a 5-line summary.
