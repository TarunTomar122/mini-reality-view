# Signature request: scripts/temporal_compare.py, content hash ffaffe4425a1 (git 5c0d234)

Drift check 7 (council/drift-20261127-0255-*.md). Final split on k after one swap: Astra k = 1, Fable k = 3 -> per my
brief, the stricter pair is used: k = 3 and the floor (20 decision / 12 final; dev 0), per comparator, ID and W separately.
The owner may revise k in the morning.

What changed (all in a NEW file so the signed scorer scripts/temporal_metrics.py stays byte-identical, 764dd3ec95db):
- --make_count_list: counting list from the COMPARATOR's metrics alone (both seeds required), committed before the
  candidate is opened. Decision vs independent: council/count_indep_dec_k3.json (28 of 30 drives count for ID and W;
  out: scene-0036 (0 matched), scene-0014 (no eligible transitions)).
- --compare: ID / W only on counting drives; can_pass false if a counting drive lacks a candidate seed, has no same-seed
  shared transition, or counting drives < --floor; a missing late gap on a contiguous run is a hard failure (a rollout
  scored without --cold is its own cold reference, gap 0); everything else identical to the signed --compare.
- Check on existing data: warm 0.7 vs independent: ID 0.105 vs 0.136 on 27 drives, can_pass false (scene-0016 counting
  with zero shared); W can_pass false; other numbers unchanged.
- temporal_rollout.py stamps the autocast dtype ("amp") in each rollout file (Fable).
- Queued (held for the cap): B 4k independent (cold) rollouts, dev and decision (Fable's check 7 (c)).
Reply SIGN / NO in one or two lines to council/compare-sign-ffaffe4425a1-<you>.md.
