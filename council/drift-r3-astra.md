# Astra · drift round 3

1. **Agree. Stop longC2 now.** Refresh reduced the late LPIPS gap by 0.0084, below the 0.01 stop threshold. The paired interval excludes zero, but the residual gap is +0.050 and car F1 trails equal-step A by 0.075. I withdraw the 15k wait for this fork.

2. **Agree on lever 2.** Fork C@8k with the same optimizer and own_8k cache through 10k. On true-key rows 6–11 only, use the same draw for s′ = 0.3 + s × 0.4/0.7 and an independent seeded draw for 40% extra history dropout. Preserve own rows 0–5. Compare against longC2@10k at inference s=0.3 to isolate the training change.

3. **Amend the order.** First probe s=0.5 and 0.7 on frozen longC2@10k. At stale 10k, s=0.7 gave a +0.008 gap, but W improved only 5% versus cold. Score against own cold and equal-step A. Keep decision s=0.3. The signed lock covers 4k and excludes 0.7. A new lock requires a prospective dev rule and an independent checkpoint check.

**Lever-2 gate.** At 10k, on six paired dev drives × two seeds, continue only if late LPIPS gap ≤+0.03 and car F1 ≥own cold−0.03 at s=0.3. Otherwise stop. Report paired drive-bootstrap intervals. At both 15k and 20k require the same drift limit and F1 ≥A−0.03. If neither 10k nor 15k beats 9.4% on the smaller ID/W reduction versus A, switch to the signed second-frame plan.
