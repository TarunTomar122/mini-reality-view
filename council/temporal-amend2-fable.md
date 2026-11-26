# Amendment 2 (rollouts through rejected ticks, drift at 17 s): Fable

**APPROVE, with four conditions.**

No model output exists yet, so the change is not tuned to results. Slightly stale cameras feeding the history can only hurt the history arms, so it is conservative.

1. **Count before training.** "~All 24" is expected, not measured. Report drives per split with a >= 170-tick rollout under the new rules. Under 12 final drives: the stop rule applies.
2. **Fixed late window.** Use ticks 150-169 of every rollout, not "last 2 s", so every drive is measured at the same age. Require >= 15 accepted ticks in it.
3. **Hard break** when any camera is over 100 ms off: that bundle belongs to another tick.
4. **Cold renders** use the same bundles as the rollout.

The late-gap gates were set for 19 s; the write-up must say they are now checked at 17 s.
