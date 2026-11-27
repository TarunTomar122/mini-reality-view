# Next video step: final pick (Fable)

**Pick: merged. Train both as two arms; send one to the decision drives.**

- **Why both.** Training is cheap (about 1 h for Astra's, 4-7 h for mine); rendering is the cost. Astra's arm also tests my untested claim that jumps come from the sampler, not the inputs.
- **Choosing.** Same six dev drives, seeds 0 / 1. Inside the safety gates first, then smallest max(ID ratio, W ratio). Within 0.03: Astra's, because it is causal and cheaper.
- **Budget (undertraining).** Mine: 12k, no gain-based stop before 8k, extend to 20k if dev still improves from 8k to 12k. Astra's: 8k instead of 4k, gain check at 2k, not 1k. Safety stops stay at every checkpoint. Log the new weights' norm: still growing at the end means "possibly undertrained", not "failed".
- **Not causal:** acceptable for rendered chase-cam video. The phone demo would need Astra's arm.
