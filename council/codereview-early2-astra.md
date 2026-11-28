At fe0ec15, both early checks require the recorded 8k scorer to complete. Intent lookup starts at the saved time, unknown state trips on pass 10, and the watcher checks gated jobs through completion.
vpriv_checks.sh L79–80 checks two free slots only at helper admission. The next pass can submit a 12k shard into the trainer’s slot while the helper waits.
vpriv_score.sh L22–24 clears CONT_PENDING on the generic sbatch failure prefix, which can also report a lost controller response. Clear it only for explicit rejection reasons and retain it on transport errors.
A definite rejection exits the helper with status 1. vpriv_checks.sh L62–65 then kills the watcher, so it can exit before queuing the regular 12k scorer. Treat a confirmed early rejection as a completed handoff to that path.
FIX: reserve the trainer slot across passes, preserve ambiguous intents, and let the regular 12k path run after a definite early rejection.
