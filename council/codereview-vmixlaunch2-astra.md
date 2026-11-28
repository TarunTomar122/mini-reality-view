Keep launcher 3756367 held pending one ghost-input check in `scripts/a1_decide_vmix.py:49–56`.
The corrected bootstrap records signed A’s upper bound of 2 and rejects the complete arm with drive mean 4. The road rule still accepts an upper endpoint of zero.
The comparator verifies both seeds, but the decision script reopens the metrics. In a local replay, dropping the arm’s high seed after comparison changed its mean from 4 to 2 and flipped `joint_pass` to true with only 11 arm runs.
The launcher requires verified ρ=1.0 before submission, early continuation requires the recorded completed 8k scorer, and the watcher locks before reconciling intents.
SIGNED: if the decision script requires exactly seeds 0 and 1 for every matched drive in both reopened metrics files and rejects the 11-run replay.
