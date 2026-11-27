# Astra’s drift review

**Give the 15k check its signed verdict. If it stops the original run, restart C at 8k with its own 8k cache.** The original 8k→10k stretch already supplies the stale-cache comparison. Replaying those 2,000 updates changes one input and can answer the causal question before another long run.

## What the gap tells us

At 5k, turning memory on gained 0.012 car F1 over C cold. At 10k it cost 0.038, raised late LPIPS error against the keys by 0.059 (95% CI 0.039–0.076), and raised half-second ghost tracks from 4.9 to 15.25 per run. Some placement loss survives with memory off (C cold 0.421, A 0.454). Meanwhile ID improved 11.1% versus A, but only on cars both methods still matched. Better-looking surviving tracks and more persistent wrong cars can coexist.

My leading explanation is that C learned to trust yesterday’s renderer. From 4k through 16k, six rows per batch supply a detached C@4k predecessor, six supply a true key, and four are singles. The own frames already include long-drive errors. What is missing is practice correcting the *current model’s* errors. A coherent misplaced car can survive Gaussian corruption and be reinforced next tick. The shared starting noise is fixed even for cold C, so it alone cannot explain the on/off difference. DAgger trains on states visited by successive learners. Self Forcing generates context with the learner during training and applies video-level supervision. Our cached single-frame epsilon loss implements only part of that idea. [DAgger](https://proceedings.mlr.press/v15/ross11a.html) · [Self Forcing](https://arxiv.org/abs/2506.08009)

## Cheapest discriminators

1. Reuse the signed 5k/10k per-drive scores. Pair seeds within drive and subtract early gap (ticks 5–24) from late gap (150–169) on the same five contiguous drives. A 10k early gap near +0.01 with late +0.059 supports accumulation. An early gap already near +0.05 points to poor conditioning within the first seconds. The published late number alone cannot distinguish these.
2. For a frozen-weight probe, render C@10k once at history noise 0.5 on the six dev drives × two seeds. Reuse its cold and 0.3 outputs with identical noise streams. GameNGen motivates weakening unreliable visual history. If excess reliance dominates, predict late gap ≤+0.03 and F1 closer to cold (≥0.40). A gap still ≥+0.05 weakens this explanation. Report ID/W to expose any return of jumping. This probe cannot identify cache age. Keep 0.3 locked for decisions. [GameNGen](https://arxiv.org/abs/2408.14837)
3. The decisive comparison is repaired C@10k versus recorded C@10k, both forked from the same 8k weights and optimizer, each scored against its own cold. Verify identical cache drive/seed/strength assignments. Predict a reduction of at least 0.02 in late gap if freshness matters materially. If cold placement also deteriorates, report both effects. Merely observing a better score at 20k after a refresh confounds cache age with additional training.

## One change and its cost

Refresh history from the exact starting checkpoint of each short stage. For the replay, use a **new output directory** with `--init runs/longC/ckpt_8000.pt --cache data/own_8k --cache_step 8000 --cache_snapshot runs/longC/ckpt_8000.pt --steps 10000`. An existing output checkpoint overrides init. Preserve optimizer, seed, 48-drive pool, six/six/four mix, learning rates and camera-only inference. Change the stage launcher to require the specified complete cache and checkpoint provenance. The trainer already supports this. If the replay improves, use repaired checkpoints at 10k, 12.5k, 15k and 17.5k for the next stages through 20k, saving each explicitly.

Replay costs ~48 A6000 minutes including loading. Each subsequent 2.5k stage costs ~53 minutes including loading. Budget ~259 A6000 minutes total from 8k to 20k plus four ~10-minute cache shard sets on 2080 Tis, adding ~40 minutes wall time if run serially before stage starts. Allow separately for dev rendering/scoring and queue time. Keep ≤20 account jobs and ~18 rollout workers. If original 15k clears its signed continuation rules, finish its scheduled fresh-cache stage first and reserve the replay for unresolved drift.

## Decision

At repaired 10k and the next 5k check (15k), pair with equal-step A on six drives × two seeds. Use signed scorer `764dd3ec95db`, shared-transition counts and 2,000 drive bootstraps. Call the repair successful at 15k if own-cold late gap ≤+0.03, F1 difference vs A ≥−0.03 and the smaller ID/W reduction beats 9.4%. Report intervals, early/late F1, recall, flips and lingering ghosts. Watch the same turning and busy clips for jumps. Ghosts are descriptive. Honor the two-strike rule and the 10k/15k smoothness rule. Reserve the single 30-drive exam for the surviving locked candidate at 20k.
