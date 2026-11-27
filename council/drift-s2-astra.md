# Astra · drift swap

**Start one live-cache fork. Its first 2,000 updates are the replay.**

1. **Test timing.** I accept Fable’s continuous run. Recorded C@10k gives the cleanest early comparison. S4 gives an equal-step stale control at 20k. Its result tests the cumulative refresh policy.
2. **Refresh.** Accept in-process reload every 200 updates and redraw every 2k. Use 100% newest complete cache. A 75/25 mix adds a second intervention and can expose rows older than 4k.
3. **Success.** Keep the signed 9.4% smoothness rule. Record lingering ghosts against the 5k reference of 9.25 per run. The owner made ghosts descriptive.
4. **Freezing.** I accept copying as the leading mechanism. Memory-on latent change fell from 0.043 to 0.021 per tick, with error already visible in the first two seconds. The partial 15k panel and unsigned latent proxy leave causation open. Six true-key rows reward persistence. Lower ID/W can accompany missed arrivals.
5. **Second lever.** Accept Fable’s true-key-only noise/dropout intervention if verified fresh training still breaches the drift or placement rule twice.

## Execution order

1. Read the signed 8k fork baseline, original 15k verdict and higher-noise probes already running. A smaller gap at higher noise supports excess history reliance. Keep decision strength 0.3. Finish s4 for the control, hold s5/s6. Record any original two-strike stop. Use 8k even if impaired to test recovery from identical weights and optimizer.
2. Add `--cache_root` to `scripts/train_temporal.py`. Every 200 updates load the newest completed fork cache at or before the current step. Validate 48 unique drives, predecessor coverage, checkpoint path/step and per-drive method, strength and seed before swapping. Log source and age. Wait in-process before age exceeds 4k. Smoke-test a reload and identical `draw(step)` outputs. Inside the A6000 allocation, fork into an empty output directory.

   ```sh
   bash slurm/py.sbatch scripts/train_temporal.py --arm C --init runs/longC/ckpt_8000.pt --out runs/longC2 --pool_list tmp/longC_pool48.txt --cache data/own_8k --cache_step 8000 --cache_snapshot runs/longC/ckpt_8000.pt --cache_root data/longC2 --steps 20000 --save_at 10000,12000,14000,15000,16000,18000,20000
   ```

3. Match original own_4k/own_8k assignments before attributing the replay difference to freshness alone. For N=10000,12000,…,18000 run `bash slurm/py.sbatch scripts/temporal_rollout.py --ckpt runs/longC2/ckpt_${N}.pt --method hist --split train --drive_list "$group" --s "$s" --seeds "$seed" --shard "$i/3" --out data/longC2/own_${N}` across four original strength groups and three shards each. Publish `DONE` atomically after validation. Budget ~1 h implementation/smoke, ~139 A6000 min including one load, five 10–15 min refresh waves (10–15 GPU-hours), plus dev scoring, stalls and queues. Admit waves within 20 total jobs, including held jobs.
4. At 10/15/20k score six dev drives × two seeds, own cold and equal-step A, signed scorer `764dd3ec95db`. Expect ≥0.02 smaller late gap at repaired 10k. Average seeds within each eligible drive and bootstrap 2,000 paired drive samples. At 20k require stale-minus-repaired gap ≥0.03 with paired 95% drive-bootstrap interval above zero to attribute material benefit. Review turning/busy clips, recall and lingering ghosts at each check. Flag a >15% motion drop on paired pool drives between refreshes.
5. Require late gap ≤+0.03 and F1 ≥A−0.03 at 15k and 20k. Stop on two consecutive checks breaching either drift or F1 limit or if neither 10k nor 15k beats 9.4% on the smaller ID/W reduction. Only the surviving candidate takes the 30-drive exam. For repeated drift/placement failure, retest once with rows 6–11 at s∼U[0.3,0.7], 40% history dropout. A smoothness-only failure follows the existing second-frame plan.
