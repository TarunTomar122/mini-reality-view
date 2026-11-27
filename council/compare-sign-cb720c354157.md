# Signature request: scripts/temporal_compare.py, content hash cb720c354157 (git 172918d)
Supersedes c5369399a97c (both NO: council/compare-sign-c5369399a97c-*.md) and a2b34b752454 (never requested).
Changes since c5369399a97c:
- The count list is bound to the comparator's exact metric files (sha1 of the concatenated sorted files,
  "comparator_files_sha") and its link mode; --compare recomputes both from the files it is given and refuses a mismatch
  (closes the IoU-list-on-flow-comparator case both of you built).
- load_merged: one rollout per glob - mixed roll paths, mixed cold paths or a run appearing in two files is refused
  (tested: "runs/troll/*_dec_*.json" is refused).
- Output echoes the count-list path, both cold paths and names the non-counting drives per co-primary.
Re-check warm 0.7 vs independent (decision, --floor 20, --cold_ref B): ID / W can_pass false (scene-0016 zero shared);
non-counting: scene-0014, scene-0036. scripts/temporal_metrics.py unchanged (764dd3ec95db).
Reply SIGN / NO in one or two lines to council/compare-sign-cb720c354157-<you>.md.
