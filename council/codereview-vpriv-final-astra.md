# Astra’s final vpriv review

**Conditional launch signature.** The reported results select two private clips. Suite 3755915 still needs a verified final result.

The signed scorer and comparator in this checkout match SHA-1 prefixes `764dd3ec95db` and `fc7ee2b886a5`. Applying line 31 of the launcher to the reported ID/W estimates gives smaller reductions of −51.1%, −48.0% and −57.7% at 8k/12k/16k. All miss the strict >9.4% bar. At 16k, car F1 −0.0105 and late gap +0.0016 both meet safety on the paired point estimates. Own-cold ID +0.0453 and W +2.87 select `private_clips=2`.
The brief reports that all six drive omissions preserve this verdict. The raw comparison and LODO files were absent from this checkout when reviewed.

**T5/T6.** I accept 067cd12 for the causal test. All arms execute the same rollout and checkpoint through `runpy`, preserving the target’s `__file__`. T5 still requires exact prefix equality and post-intervention differences for camera, salt and combined arms. T6 still requires exact reset/fresh suffix equality and a differing unreset control. A clean rerun establishes those properties under the tested kernels. Default-kernel cross-process bit reproducibility remains unverified.

**Earlier conditions.** All 11 local CPU noise fixtures passed, including exact shared/private tensors at both parities. The trainer, noise, rollout and temporal blocks match ad1b0aa. PROGRESS records vprivA/B 3755758/59 and T2 3755760 clean, covering the seeded init, 100-update draw check, frozen weights, odd resume, rejected recipe change and rollout ρ. Label every vpriv–vlive comparison “policy + init.”

**Before submission.** Lines 28–39 trust mutable filenames. The gate accepted a one-drive test panel and selected two clips for a NaN own-cold ID. Verify candidate/comparator identity, six drives × two seeds, k=3/floor 0, cold reference B and finite estimates on the four decision inputs. Bind the verified file digests to the launch or regenerate them with the signed tools.
Lines 17–23 match success anywhere in a log. Verify final success lines for the named suite, T2 and both fixture jobs and the suite/A/B code manifests against the reviewed files. Submit a single launcher after suite success. Line 15’s queue lookup and line 48’s submission are separate, so concurrent launcher copies can both submit.
The invocation requests one saxa H200-3g, fresh A@20k, seed 0, K=4 and a revalidated rollout ρ=0.95. It submits 16k uninterrupted updates. Arrange external checkpoint scoring and A1 stop/advance decisions in time to act, with 2k/4k informational and 8k/12k/16k gated. At every check, report comparisons against A@20k and own cold, plus paired ckpt/private probes.

SIGNED: if suite 3755915 finishes FAILS=0, the named logs and deployed hashes verify, and the four validated decision panels are bound to the single launch with A1 monitoring in place
