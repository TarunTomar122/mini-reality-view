# Signature request: scripts/temporal_compare.py, content hash c5369399a97c (git 68c81dc)
Supersedes ffaffe4425a1 (both NO: council/compare-sign-ffaffe4425a1-*.md). Changes:
- Count list bound to its comparator: stores comparator_roll + drive panel; --compare asserts both match B; output names
  the list's per-seed comparator counts.
- Finite values only (np.isfinite) for matched / shared transitions.
- Per drive AND seed: eligible / comparator-matched counts in the list, shared counts per drive and seed in the output.
- Incomplete panel (any drive without seeds 0 and 1 in both methods) is a hard failure (nothing silently drops from
  recall or other gates).
- Late gaps: zero only for the side declared with --cold_ref A|B (the cold reference); any other missing late gap on a
  contiguous run is a hard failure (no implicit zero for a history rollout scored without --cold).
- --floor required whenever a count list is given; count lists must have k = 3.
- Regenerated council/count_indep_dec_k3.json (28 / 30 count). Re-check warm 0.7 vs independent (--cold_ref B --floor 20):
  ID can_pass false (scene-0016: 0 / 0 shared per seed), W false; car F1 -0.059 (30 drives), late LPIPS +0.058 (22), recall
  -0.050 (29: scene-0014 has no eligible edge tracks).
scripts/temporal_metrics.py unchanged (764dd3ec95db). Reply SIGN / NO in one or two lines to council/compare-sign-c5369399a97c-<you>.md.
