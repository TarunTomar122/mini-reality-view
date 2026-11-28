AMEND Fable’s exit-6 rule before reading update 2,000. Accept the threshold and budget for verification under the new gate key.
The first exit 6 qualifies only if key cars exist, round-trip F1 > 0, hybrid F1 ≥ 0.80 × round-trip F1, and hybrid F1 > no-input F1 + 0.02.
Require fixed-noise loss at 2,000 ≤ 0.50 × loss at update zero and three strict decreases across updates 1,700, 1,800, 1,900 and 2,000. Any other exit 6 is FAIL. Hold production and diagnose.
For a qualifying exit 6, sign one verdict-only edit and retain LR 1e-4. The edit changes the gate key, so charge the forced re-time (~1 slice-hour) and one fresh 2,000-update gate (~2.5 slice-hours) against the estimated 6.5 hours of slack.
Only the trainer may register PASS. Release production after new-key overfit and bit-equal resume PASS, with a live forecast ≥13,200 updates and one panel reserved. Escalate a budget shortfall to the owner.
The 24 h panel reads the rolling `ckpt.pt` saved at the production budget stop, at its exact step if ≥12,000. Below 12,000, quality stays unresolved. Retain numbered saves for a later, separately signed averaging experiment.
PLAN: 1. Read the 2,000-update record. 2. Apply the signed branch. 3. Verify both same-key gates and forecast. 4. Train to budget stop. 5. Score the eligible checkpoint once.
