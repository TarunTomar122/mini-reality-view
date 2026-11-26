# Temporal consistency and memory: round 5 (Fable)

**SIGN** on PLAN-temporal.md v5. No invalidating issues remain.

Both round-4 changes are in the text as requested:

- **ID and W on shared transitions:** each candidate-comparator comparison averages over the same key transitions, both
  methods matching the car at both ticks (IoU >= 0.3); W uses the same set; shared counts reported per drive; an
  eligible drive with none cannot pass.
- **Recall gate:** paired 95% lower bound of candidate minus comparator >= -0.02 against frozen independent and every
  adoption comparator, including A + warm start for B/C.

Not blocking: state in the write-up that W's "cannot pass" rule is inherited from ID.
