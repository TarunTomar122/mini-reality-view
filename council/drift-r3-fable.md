# Drift round 3 (Fable)

**1. Stop longC2 now: agree** (my own rule). Cancel the own_12000 refresh. If `ckpt_15000.pt` already exists, score
it at s 0.3 as lever 2's control; report only.

**2. Lever 2: agree, three amendments.**
- New empty `--cache_root` (own_8000 linked to data/own_8k). `maybe_reload` trusts each folder's own `DONE`, so in
  data/longC2 it would silently load longC2's own_10000.
- Smoke check: `true_drop_share` should log about 0.46 (10% base + 40% extra).
- The proposed gate is passed by a model that ignores its memory. Add a memory-used clause.

**Gate at 10k** (6 dev drives x 2 seeds, s 0.3, scorer 764dd3ec95db, vs own cold; control longC2@10k, paired drive
bootstrap):
- Continue: late gap <= +0.03 and at least 0.02 below longC2@10k, interval excluding zero; F1 >= own cold - 0.03;
  ID below own cold, interval excluding zero.
- Stop: late gap > +0.04, or fall < 0.01. Back to council; no third lever by default.
- Otherwise: continue; same rule decides at 12k.
- A pass is re-checked at 12k (drift appeared within 2k updates), then the signed 15k / 20k lines, two strikes.

**3. s: test alone first; no re-lock with lever 2.** The control was scored at 0.3; moving both makes the gate
unattributable. Stale C@15k at s 0.5 still gave +0.095: inference noise delays, not cures. Read the running
longC2@10k s 0.5 / 0.7 probe. Re-lock once, on the surviving checkpoint before the exam, after amending the signed
rule (4k-only, no 0.7 candidate).
