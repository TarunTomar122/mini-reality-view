# Astra launch review of `07e3391`
Keep launcher 3755981 held. All 26 manifest hashes match the commit locally. The builder reports T7 still running.
Item 1 closes. The 12k and 16k scores each depend on their own metrics and the preceding gated score via `afterok`.
Items 2–3 close for A1 timing. Stage 1 ends at 12k, two safety misses cause no continuation, and a continue decision resumes from step 12000 to 16000. A1 issues no cancellation.
The 12k handoff still masks failures. In `slurm/vpriv_score.sh:21–32`, failed lock acquisition and a failed intent write can fall through to submission. A mocked stop-marker write failure reached `SCORE_DONE` with exit 0, leaving the watcher waiting. Require successful lock, intent and stop-marker writes before proceeding.
A retry after the 12k decision is written exits at `vpriv_score.sh:19` because `a1_decide.py:45` uses exclusive creation. Reuse a verified matching decision under the lock so the retry can reconcile the continuation receipt or pending intent.
Item 4 remains open at `slurm/vpriv_checks.sh:19–23`. With a lost receipt and empty active queue, the mock called sbatch three times. An accepted job can finish before the lookup. With two same-name IDs the function silently adopted the last one.
Stop on an ambiguous receipt unless exactly one job can be tied to this submission. An empty active queue alone cannot authorize a retry. The 16/20 capacity check does not resolve submission identity.
SIGNED: no
