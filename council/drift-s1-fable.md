# Drift round 1 (Fable): the memory learned to copy itself, because its practice frames went stale

## What I checked (read-only, login node CPU, no jobs, nothing written)
Latent error to the key and tick-to-tick change, from files that already exist. Unsigned proxy, not the scorer.

**Dev, memory on (s 0.3) minus memory off, same weights** (10 long rollouts; 15k is partial: 6)

| step | early ticks 5-24 | late ticks 150-169 | tick-to-tick change, memory on (off = 0.06-0.08, key = 0.22) |
|---|---|---|---|
| 4k | -0.005 | +0.028 | 0.040 |
| 5k | 0.000 | +0.019 | 0.043 |
| 10k | +0.029 | +0.071 | 0.029 |
| 15k | +0.094 | +0.142 | 0.021 |

**Train pool, same 43 drives in all three caches (paired)**

| cache | error to key | tick-to-tick change |
|---|---|---|
| own_4k | 0.369 | 0.039 |
| own_8k | 0.330 | 0.047 |
| own_12k | 0.377 | 0.034 |

History conv weight norm: 1.16 (5k), 1.36 (10k), 1.54 (15k). Still growing in a straight line.

## 1. Diagnosis
**The model is freezing, not wandering.** Its frames change half as much per tick at 15k as at 5k, and the gap is
already there in the first 2 seconds. Part of the extra "smoothness" at 10k is the picture going stale: cars that
left stay (lingering ghosts x1.6), new cars arrive late (recall 0.39 vs 0.435).

**Why.** Six rows per batch show the true previous key, where copying is always right. This is the copycat
shortcut (Wen et al. 2020): training loss keeps falling while closed-loop behaviour gets worse. The six own rows
are the only brake, and a brake only works on the errors it shows. Ours showed C@4k's errors for 12k updates.

**This is the proven method done incorrectly.** DAgger (Ross et al. 2011) needs rollouts of the *current* model,
gathered again every round; Self Forcing (Huang 2025), BAgger (Po 2025) and SVI's error bank (Li 2025) do the same.
The signed 4k recipe refreshed every 1.5k updates and had no drift. The long bet has trained 4k to 12k updates
stale; stage s4 is on own_8k (own_12k was 13/48 at its start), so 8k to 12k stale.

**Ruled out: "it only fails on unseen drives".** The 48 training drives turn too: 8k was the best point (error
-10%, more motion); by 12k error is +14% and motion -27%, on 40 of 43 drives.

**The noise level is a trust dial.** On the pool, 8k to 12k error rose +0.06 at s 0 / 0.1, +0.05 at s 0.3, +0.03 at
s 0.5. It is also why B (the GameNGen recipe, Valevski 2024) drifted: noising the history covers noise-like errors,
not a believable fake car.

**Cheapest tests (2080 Tis, about 15 min each, existing scripts and scorer)**
- **T1, gain:** C@10k and C@15k dev at s 0.5 and 0.7. If copying is the cause: late LPIPS gap at 10k falls from
  +0.059 to about +0.03 (s 0.5) and +0.02 or less (s 0.7), lingering ghosts 15 to 10 or fewer. Flat across s (as
  at 4k: +0.007..+0.010) would speak against it.
- **T2, fork point:** C@8k dev, s 0.3 and cold. The pool says 8k is healthy: expect late gap +0.02 or less and
  memory-on car F1 at or above memory-off.
- **Free control:** let s4 finish. C@20k on own_8k is the stale twin of the fix below; expect it worse than 15k.

Not separable with existing files: staleness versus sheer training length. Only the fix run can.

## 2. The fix: a live own cache, forked from C@8k
One change: practice frames always come from a checkpoint at most 4k updates old. Rows, loss, learning rates,
s range, sampler and scorer stay untouched.

- **Fork** `runs/longC/ckpt_8000.pt` (optimiser inside) into `runs/longC2`, to 20k; if T2 fails, fork
  `ckpt_5000.pt`. Cancel s5 / s6. Rows and noise stay identical to A, so the 20k exam against A@20k stands.
- **`train_temporal.py`:** `--cache_root data/live`. Every 200 updates look for the newest `own_<N>/DONE`; if newer,
  reload it in-process (48 files, 260 MB, seconds; one 30 min load in total). Keep the current asserts. Own rows
  take the newest cache with probability 0.75 and the one before with 0.25, from a separate seeded stream so A's
  draws are unchanged (aggregation, as in DAgger). Log `cache_step` and `stale`. `--save_at` every 2k.
- **`slurm/own_refresh.sh`:** when `ckpt_<N>.pt` appears (N = 8k, 10k, ... 18k), submit 12 shards of the existing
  pool rollout (same four s groups, seed = refresh number), then write `DONE`. Training never waits.
- **Cost:** A6000 30 min load + 12k updates = about 2.3 h. 2080 Tis: 6 refreshes x 12 jobs x 10-15 min (under the
  20-job limit with the dev checks). Code and a 100-update smoke test: about 1 h, while s4 finishes.
- **Staleness:** 1k to 4k, mean about 2.5k. 4k stale was still healthy at 8k.

## 3. What to measure, and the rule
Signed dev check of longC2 at 10k, 15k and 20k against equal-step A, plus memory-off on the same weights.

**It worked if, at 15k and 20k:** late LPIPS gap +0.03 or less, car F1 within 0.03 of A, lingering ghosts no
higher than at 5k (9.25), smoothness headline still positive. And at 20k longC2 beats stale C@20k on late gap by
0.03 or more, interval excluding zero: that is the proof that staleness was the cause.

**Early warning, free at every refresh:** the paired pool numbers above. Tick-to-tick change falling more than 15%
between two refreshes, or error to the key rising, means it is freezing again.

**If it fails** (late gap above +0.03 at 10k and 15k with a fresh cache): staleness was not the cause. The one next
lever is the signed plan's own: true-key rows at s ~ U[0.3, 0.7] with 40% history dropout, so the clean previous
key stops being a free answer.

Sources: Ross 2011 (DAgger); Wen 2020 (copycat agents); GameNGen arXiv 2408.14837; Self Forcing arXiv 2506.08009;
BAgger arXiv 2512.12080; Stable Video Infinity arXiv 2510.09212.
