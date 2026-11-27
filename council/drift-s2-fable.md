# Drift round 2 (Fable): one run that is both Astra's test and the fix

## Point by point
1. **Test first: agree with Astra, and it costs nothing.** Astra's replay (C@8k to 10k on own_8k, new folder) is
   exactly the first 2k updates of my fix. Run it first with today's code; the live-cache job then resumes it. Both
   orders finish near 16:00 (the A6000 is busy with s4 until about 13:05 and my coding takes about 1 h), but this
   order gives the causal answer about 35 min sooner, from untouched code. My "prove it at 20k" becomes a second
   confirmation (both runs are at 20k there, so it is not confounded with training length).
2. **Refresh: keep the in-process reload; I drop the 75/25 mix and the per-refresh seeds.** Five stage restarts cost
   5 x 30 min of loading, the reload costs 2. Astra is right on purity and provenance: 100% newest cache, the same
   drives / seeds / s groups as own_8k in every refresh, step and checkpoint path asserted at every reload. My mix
   would also have put a quarter of the own rows at 4k to 6k stale, outside the range my own evidence calls healthy
   (up to 4k). Cache age with the reload: about 1.8k to 3.8k.
3. **Success: two questions, two rules.** "Drift fixed" = the signed lines (late gap <= +0.03, car F1 within 0.03 of
   A). "Is one memory frame enough" = Astra's 9.4% bar, unchanged; a miss triggers the already-signed next step (a
   second history frame), not a rollback of the fix. I withdraw "lingering ghosts <= 9.25" as a gate (owner: ghosts
   never stop the run); reported only. The freeze watch stays as a warning, not a stop.
4. **Freezing.** My numbers are an unsigned latent proxy on 6 to 10 rollouts. Astra's discriminator 1 on the signed
   per-drive scores (ticks 5-24 vs 150-169) is the free confirmation: I predict an early gap of +0.02 or more at 10k.
   If so, part of the 10k "6.8% smoother" is a stiller picture, and the 9.4% bar must be read next to the motion number.
5. **Second lever:** kept, but only if the s probes show the gap falling as s rises (C@10k at s 0.5: +0.035 or
   less). If flat, back to council; no new method by default.

## Merged plan
Assumes the C@15k check fails, as my proxy predicts (+0.14). If it clears both signed lines, hold after step 1.
If the C@8k probe shows late gap above +0.03, fork `ckpt_5000.pt` instead (render own_5k first) and refresh every 1k.

1. **About 13:05, when s4 ends. No code change. 48 min.** Cancel s5 / s6.
   `scripts/train_temporal.py --arm C --init runs/longC/ckpt_8000.pt --pool_list tmp/longC_pool48.txt --cache data/own_8k --cache_step 8000 --cache_snapshot runs/longC/ckpt_8000.pt --steps 10000 --save_at 10000 --out runs/longC2`
2. **Meanwhile, about 1 h of code (checked on the login CPU, no GPU needed).**
   - `train_temporal.py`: move lines 99-120 into `load_cache(dir)` (one read per file; exactly one file per pool
     drive; stamps must equal the step and checkpoint path written in `DONE`). New `--cache_root data/live`: at start
     and every 200 updates take the largest N <= step with `own_<N>/DONE`; load into a new dict, run the asserts,
     then swap. On failure keep the old cache and log it. Log `cache_step` and `stale` every 100 updates.
   - `slurm/own_refresh.sh` (login-node loop): when `runs/longC2/ckpt_<N>.pt` appears (10k, 12k, 14k, 16k, 18k),
     resubmit the own_8k commands (jobs 3753929-36) as 12 shards with `--ckpt runs/longC2/ckpt_<N>.pt --out
     data/live/own_<N>`; an afterok job checks 48 files, no tmp, and writes `DONE`. `own_8000` = link to data/own_8k.
3. **At ckpt_10000 (about 13:55), three things at once:** own_10k refresh; dev check of longC2@10k (cold + s 0.3,
   3 shards each); and the live job, which loads for 30 min and so starts on own_10k:
   same command with `--cache_root data/live --steps 20000 --save_at 12000,14000,15000,16000,18000,20000`. 20k near 16:00.
4. **10k gate (about 14:35):** longC2@10k vs recorded C@10k, each against its own cold, paired drive bootstrap.
   Late gap falls by 0.02 or more, interval excluding zero: staleness confirmed, continue. Falls by under 0.01:
   stop (under 15 min of updates lost), go to step 6. In between: continue, decide at 15k.
5. **15k and 20k:** signed check vs equal-step A and own cold. Fixed if both lines hold at both checks; two strikes
   running stop the run. At 20k also score stale C@20k: expect longC2 better on late gap by 0.03 or more. At each
   refresh compare pool tick-to-tick change with the previous cache; a fall over 15% means watch the video. One
   30-drive exam on longC2@20k vs A@20k.
6. **If the gate fails and the probes support it:** `--true_s_min 0.3 --true_drop 0.4` (arm C rows 6-11:
   s' = 0.3 + s x 0.4/0.7 from the same draw; extra dropout from a separate seeded stream), forked from C@8k with
   the live cache.

**Jobs:** train 1 + refresh 12 + dev check 6 + scoring 1 = 20. **A6000:** about 2.8 h in total.
