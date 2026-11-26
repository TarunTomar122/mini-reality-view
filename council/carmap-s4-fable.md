# Car map, round 4 (Fable): SIGN

PLAN-carmap v3 signed as written.

- My X3 scoring rule is in verbatim (decision rules, X3).
- Astra's frozen-winner X3 is sound and stricter than what I asked: the comparator is the same weights with the
  branch off on identical noise, so my non-blocking point 2 (arm A's 4k -> 6k drift) no longer applies. "B/C keep
  their history conv and corruption" still reads consistently: conv frozen, corruption applied while the branch trains.
- Hull tests now state score >= 0.3.

Reading I sign under (no text change needed): "ticks since the last detector refresh (0-4)" means manifest
`tick % 5`, i.e. ticks since the last keyframe, whichever P4 feed wins. If sweep detection is chosen, phases 1-4
are still the gated pool; it is never empty.

Still open, non-blocking: the write-up states how the final 24 were used.
