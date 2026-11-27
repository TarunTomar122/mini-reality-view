# Mini Reality View: progress (newest at top)

## 2026-11-27 14:25 — lever 2 launched (longC3); longC2 stopped
- Fable round 3 (council/drift-r3-fable.md) agrees (stop longC2, lever 2 with a NEW empty cache root -> data/live3 used; decision s stays 0.3; re-lock later on the surviving checkpoint only). MERGED 10k GATE for longC3 (s 0.3, signed): CONTINUE if late gap <= +0.03 AND F1 >= own cold - 0.03 AND ID below own cold (memory still smooths) AND late gap >= 0.02 below longC2@10k (+0.050) with CI excluding 0; STOP if late gap > +0.04 or the fall vs longC2@10k < 0.01 or F1 < own cold - 0.03; otherwise decide at 12k. 15k / 20k: late gap <= +0.03 and F1 >= A - 0.03; 9.4% smoothness rule unchanged.
- Astra round 3 (council/drift-r3-astra.md): agree stop longC2; agree lever 2 from C@8k; probe inference s 0.5 / 0.7 on frozen longC2@10k but keep decision s 0.3 (a new s lock needs a prospective rule + an independent checkpoint). Lever-2 gate at 10k: late gap <= +0.03 and car F1 >= own cold - 0.03 (s 0.3), else stop; 15k / 20k: same drift limit and F1 >= A - 0.03; 9.4% smoothness rule unchanged -> else second history frame. Fable's round-2 plan already named this step (step 6); Fable round 3 pending.
- Cancelled longC2 at 14.9k (ckpt_10000 / 12000 / 14000 kept; it was on own_12000) and its own_12000 / own_14000 refresh jobs.
- longC3 (3754417): --init runs/longC/ckpt_8000.pt, own_8k, live cache data/live3, refresh every 2k, --true_s_min 0.3 --true_drop 0.4, to 20k. 10k check queued (M10k, 2+2 shards + metrics).

## 2026-11-27 14:15 — 10k gate FAILED: fresh cache alone does not stop the drift
- First live refresh worked end to end (own_10000: 48 files, DONE, swapped in at 12.2k after a ~5 min wait at the 4k age limit; refresh waves take ~30 min with 8 jobs). own_12000 rendering.
- Gate (signed): longC2@10k (8k -> 10k on own_8k, cache 0-2k old) late gap +0.050 [0.032, 0.068] vs stale C@10k +0.0585 -> fall 0.0084 [-0.0165, -0.0019] < 0.01 -> stop rule. longC2@10k vs own cold: F1 0.379 vs 0.415, ghost_ge5 14.75 vs 3.9; vs A@10k: ID -26%, F1 0.379 vs 0.454.
- Reading: the drift is produced by training past 8k itself (C@8k +0.002 -> +0.050 in 2k fresh updates), not mainly by cache age. Matches Fable's copy-shortcut: the true-previous-key rows reward copying.
- Next (council round 3 for sign-off, council/drift-r3.md): stop longC2; lever 2 = true-key rows s' in [0.3, 0.7] + 40% extra history dropout (train_temporal.py --true_s_min / --true_drop, written), forked from C@8k with the live cache; plus whether to re-lock inference s higher. Probe running: longC2@10k at s 0.5 / 0.7 (metrics 3754405 / 3754408).

## 2026-11-27 13:10 — old recipe stopped at 15k (strike two); probes back the fix; fixed run longC2 training
- C 15k (s 0.3) vs A 15k (signed): ID 0.082 vs 0.139, W 30.1 vs 31.7, car F1 0.332 vs 0.430 (CI -0.161..-0.046), recall 0.359 vs 0.432, late LPIPS gap +0.119 (CI 0.066..0.161), early gap +0.059, ghost_ge5 24.5 vs 5.1. Second check running past both stop lines -> the stale-cache run is stopped (s4 finished 20k as the stale control; no further stages).
- Probes, memory on vs same-weights cold (signed; count list from the cold run): C@8k s 0.3: late gap +0.002 (CI -0.013..0.015), F1 0.423 vs 0.416 -> healthy fork point. C@10k: s 0.3 +0.059, s 0.5 +0.027 (F1 0.405 vs 0.421), s 0.7 +0.008 (F1 0.401; ID -13%, W -5% vs cold, ghost_ge5 7.1). C@15k: s 0.3 +0.119, s 0.5 +0.095. History noise is a trust dial (supports Fable's second lever), but by 15k the stale-trained weights are too far gone for noise alone. Figure docs/figs/drift_probes.png.
- longC2 (3754199) started ~12:45, data load ~12 min, training from 8k on own_8k. In-job sbatch works on compute nodes (sbtest). Refresh uses 8 rollout jobs (owner cap 16). longC2 10k check queued (2+2 shards, wait for ckpt_10000; metrics after).
- VPN dropped again ~12:50; reconnected.

## 2026-11-27 13:00 — drift council converged: live own cache, forked from C@8k
- Round 2 (council/drift-s2-astra.md, drift-s2-fable.md) converged: fork C@8k (optimiser) into runs/longC2; own-history cache always from the current model: redraw at every 2k (10k, 12k, ... 18k) with the same drives / seed / s groups as own_8k, in-process reload of the newest complete cache (100% newest; age ~1.8k-3.8k; wait if > 4k). The first 2k updates (8k -> 10k on own_8k) are Astra's clean replay vs recorded C@10k.
- 10k gate: longC2@10k vs C@10k, each vs its own cold: late gap down >= 0.02 (CI excl. 0) -> staleness confirmed, continue; < 0.01 -> stop, go to the second lever. 15k / 20k: late gap <= +0.03 and car F1 >= A - 0.03 (two strikes stop); 9.4% smoothness bar kept (a miss -> second history frame). At 20k longC2 vs stale C@20k (s4) on late gap, expect >= 0.03. Ghosts reported only. Second lever if needed: true-key rows s in [0.3, 0.7] + 40% history dropout (only if the s probes show the gap falling with s).
- Code: train_temporal.py --cache_root / --max_stale / --refresh_every (load_cache() refactor with the same asserts + one file per drive + method hist; a bad cache is rejected and logged, old cache kept); slurm/own_refresh.sh (12 rollout shards + CPU DONE writer that checks 48 files from the same checkpoint). The trainer submits the refresh itself after saving ckpt_<N>.
- Queued longC2 (3754199) after longC-s4 (3754121, 16k -> 20k stale control; started training only at ~12:55 after ~40 min loading). Cancelled longC-s5 / s6.

## 2026-11-27 12:35 — drift council round 1 done; probes running; swap round started
- Owner asked for a research round on the growing late drift. council/drift-brief.md -> drift-s1-astra.md, drift-s1-fable.md.
- Both: stale own-history frames (C@4k frames for 12k updates); DAgger / Self Forcing need rollouts of the CURRENT model; fork from C@8k with fresh caches. Fable's read-only latent probe: memory-on frames change less per tick as training goes on (0.043 at 5k -> 0.021 at 15k; key 0.22), the gap is already there in the first 2 s, and on the 48-drive train pool 8k was best (own_12k error +14%, motion -27% vs own_8k on 40/43 drives): the model is freezing (copying its history). Higher history noise s lowers the effect.
- Held longC-s5 / s6 (to 25k / 30k). longC-s4 (16k -> 20k, own_8k) running: stale-ish control.
- Probes (signed scorer vs same-weights cold): C@8k cold + s 0.3; C@10k s 0.5 / 0.7; C@15k s 0.5 (metrics 3754189-93). C@15k s 0.3 check scoring.
- Swap round: council/drift-swap.md -> drift-s2-astra.md / drift-s2-fable.md.

## 2026-11-27 12:00 — C 10k check: smoother, but first stop-rule strike (car F1, late drift)
- C 10k (s 0.3) vs A 10k (signed compare, count list A10k k3, floor 0, cold_ref B): ID 0.0849 vs 0.0955 (-11.1%, CI -0.021..-0.001); W 26.87 vs 28.84 (-6.8%, CI -4.28..+0.92); headline 6.8% (< 9.4% at 4k). car F1 0.383 vs 0.454 (-0.071, CI -0.128..-0.027) -> past the -0.03 stop line; late LPIPS gap +0.059 (CI 0.039..0.076) -> past +0.03; recall -0.045; flips -0.037; ghost_ge5 15.25 vs 4.67 per run.
- Where it comes from: C memory vs C cold (count list from C cold, cold_ref B): 5k F1 0.441 vs 0.429, late gap +0.006; 10k F1 0.383 vs 0.421, late gap +0.059, ghost_ge5 15.25 vs 4.9. So the memory rollout itself started to drift late in drives between 5k and 10k (late gap x10). C cold vs A cold could not be run with the signed script (it hard-fails on cold-vs-cold late gaps); the C cold F1 (0.421) is slightly below A 10k (0.454).
- Likely cause to check: stages s1-s3 trained on own_4k (history frames drawn by C@4k); by 10k the model's own outputs differ from those. The plan's refresh fixes exactly this: longC-s4 starts on the newest complete cache (own_12k if 48/48, else own_8k).
- Rule: stop if a stop line is crossed at two checks running -> the 15k check decides (queued 12 shards now, metrics 3754151/52). Videos vid10k queued.

## 2026-11-27 11:35 — C keeps training past 20k
- Owner: "so now we focus on teaching this memory on for longer?" -> yes. C chained to 30k: longC-s4 (3754121, -> 20k), s5 (3754122, -> 25k), s6 (3754123, -> 30k) via slurm/longC_stage.sh, which picks the newest complete 48-drive own cache at start (own_8k done; own_12k redraw 3754124-31 running). A stays at 20k (the control for the 20k exam). longC-s3 at 14,400.

## 2026-11-27 11:10 — C 5k check: ahead of A, but less than at 4k
- VPN dropped ~10:15-10:58 (FortiTray reconnected); cluster work continued. longC-s1 done, longC-s2 (8k -> 12k) at 11,900. Queued longC-s3 (3754078, 12k -> 16k, own_4k cache because own_8k is 18/48 drives; the 5 held own8k jobs were still held -> released). C 10k check queued (3754079-92, 6+6 shards, batched script).
- C 5k (s 0.3) vs A 5k, signed temporal_compare (count list runs/count/count_A5k_dev_k3.json, floor 0, cold_ref B): ID 0.0929 vs 0.1039 (-10.6%, CI -0.0175..-0.0044); W 29.56 vs 30.81 (-4.1%, CI -2.42..-0.20); headline (smaller) 4.1% (4k: 9.4%). car F1 0.441 vs 0.448 (-0.007; stop line -0.03); late LPIPS gap +0.006 (stop +0.03); ghosts 2.35 vs 2.07; lingering ghosts (>=5 ticks) 9.25 vs 4.25 per run; recall +0.017.
- Reading: A gained from 4k -> 5k (ID 0.109 -> 0.104, W 32.0 -> 30.8) while C stood still or dipped slightly (ID 0.088 -> 0.093, W 29.0 -> 29.6) in its first 1k steps on the new whole-drive pool. No stop rule hit. The plan's decision point is 10k / 15k: beat 9.4%, else add a second history frame.
- Owner watched the 5k videos (11:20): "memory on is clearly better". 5k videos on the live page (docs/videos/scene-0234_5k.mp4, scene-0396_5k.mp4).
- Videos: vid5k jobs 3754093/94 (key | A 5k | C 5k memory off | C 5k memory on).

## 2026-11-27 10:00 — faster checks: 24-job cap, 6-way split, batched rollouts
- Owner: make the slow scoring step fast; use free GPUs, cap raised 12 -> 24.
- A 10k / 15k / 20k dev checks re-split 6 ways (one drive x 2 seeds per job; finished files kept): 3753940-59, metrics 3753946 / 3753953 / 3753960. A 5k rollouts done, metrics 3753876 running.
- temporal_rollout.py now renders up to --batch 12 (drive, seed) streams in lockstep (same per-stream noise and history streams; atomic file writes). Check (eqtest 3753939, C@4k hist s 0.3, 6 dev drives x 2 seeds x 12 ticks): batched vs old mean |diff| 0.0005-0.002 (latent std 0.7; fp16 batch rounding), restart / ok flags identical; 311 s vs 513 s incl. ~2.5 min model load, i.e. ~2x faster compute, not 3-5x.
- longC-s1 (3753891) started 09:17 but spent ~30 min loading; at 4,600 at 09:55. C 5k check queued: cold + hist s 0.3, 6 shards each, start when runs/longC/ckpt_5000.pt appears (3753962-73), metrics 3753974 / 3753975.
- Queued: longC-s2 (3753928, 8k -> 12k on own_4k), own_8k redraw (3753929-36, 8 jobs after s1).

## 2026-11-27 07:55 — C's practice drives done; C waits for the A6000
- own4k done: 48/48 own drives in data/own_4k. scene-0059 and scene-0715 are 1-tick drives in data/enc_tpv10 itself (1 tick, 1 ok), so the 1-tick rollouts are correct.
- longC-s1 (3753864) sat at Reason=Resources: landonia11 (only A6000 node we can use) has 11/12 CPUs and 168/193 GB allocated. Scheduler estimate 14:43 (= longA's time limit). Resubmitted as 3753891 with 4 CPUs (same command, 96G); it is the top-priority runnable A6000 job, so it should start when longA ends (~09:35). H200 nodes are not available on our partition; 2080 Ti would mean fp16 vs A's bf16 and ~3x slower, so not used.
- longA at 9,600 (07:46), loss 0.058. troll-A5k-dev running.

## 2026-11-27 07:25 — long run: A at 6.2k, C's practice drives 30/48
- longA (3753863): A6000, ~100 updates/min, at 6,200 at 07:16 (loss ~0.06); 20k expected ~09:35. ckpt_5000 saved.
- own4k (3753855-62): 30 of 48 own drives rendered. Two drives (scene-0059, scene-0715) have only 1 usable tick, so they add almost nothing to the pool; harmless.
- Queued the A 5k dev check early: troll-A5k-dev (3753875) -> tmet-A5k_dev (3753876), so the 5k comparison is ready when C reaches 5k.

## 2026-11-27 07:00 — long bet started (council converged in one swap)
- Council (council/long-bet-*.md, -swap-*.md): C (one history frame), forked from C 4k with optimiser state, to 20k; A to 20k on identical row / noise draws; own-history rows from WHOLE-DRIVE own rollouts of 48 seeded train drives (12 each at s = 0 / 0.1 / 0.3 / 0.5; tmp/longC_pool48.txt); 6 own + 6 true + 4 singles; refresh caches from C@8k / 12k / 16k, training never waits (keeps the previous cache); 5k checks on 6 dev drives (s = 0.3) vs equal-step A, headline = the smaller of the ID / W % reductions (4k: 9.4%), plus car F1, recall, late drift, lingering ghosts and one side-by-side video; stop if car F1 < -0.03 or late drift > +0.03 twice running, or if neither 10k nor 15k beats 9.4% (next: a second history frame); ghosts never stop the run (owner); one 30-drive exam + one council review at 20k.
- Running: own rollouts of C@4k (data/own_4k, own4k-* 3753855-62, 2080 Tis); longA (3753863, A6000, 4k -> 20k); longC stage 1 (3753864, 4k -> 8k on own_4k, afterok on the rollouts). train_temporal.py --pool_list; rollouts stamp snapshot / s / method.
- Live page: new "Right now" box at the top (goal, the bet, latest numbers, timeline, the two side-by-side videos); plan picture simplified to a straight line; stale sections relabelled.

## 2026-11-27 06:45 — owner: ONE long bet; C dev screen (best so far); videos; council picks the bet
- Owner decisions (side chat + here): one candidate trained long (~20k) + A at the same length; cheap 5k checks (6 dev drives + a side-by-side video); one exam + one council review at the end; hourly drift checks stopped (scheduled task disabled); "spend more compute testing one good thing then 3 okay things"; video smoothness is the only goal now; "if this works then we will figure out how to get rid of fake cars later" (ghosts measured and shown, not blocking). Car map parked (only 1-2k steps), not dead.
- Cancelled: tmpA-12k / tmpB-12k, A 2k (+ warm) decision rollouts, B 0.3 decision rollouts, B 2k dev rollouts, A 4k decision scoring, A 4k + warm decision rollouts, B 4k cold decision rollouts, R dev18 renders.
- C 4k dev s-lock (mechanical, council/s_lock_C.json): ID 0.088-0.093 vs 0.105-0.109 (-12..-19%), W 28.1-29.2 vs 31.3-32.0 (-9..-11%), car F1 -0.011..-0.020, recall -0.007..-0.018, own late LPIPS +0.007..+0.010, late F1 +0.001..+0.007 -> passes every clause except lingering ghosts (+3.9..+6.0 per run vs A; B was +7.6..+14.2). Supports Fable's reading: training on own outputs removes drift and most ghosts. Locked s = 0.3 under the rule ("failed dev screen" on the ghost clause only).
- Owner videos (scripts/compare_video.py): pulled/videos/scene-0234_s0.mp4 and scene-0396_s0.mp4 (key | E3 | A 4k | E3 + warm 0.7 | B 4k s 0.3).
- Council (one round + converge): council/long-bet-brief.md -> which arm to train long and with what recipe.

## 2026-11-27 06:35 — owner: undertraining? + videos
- Owner: "we might be undertraining". Evidence: B's history-conv weight norm still growing at the stop (0.85 at 1k, 1.27 at 2k, 1.37 at 2.5k, 1.55 at 4k; slowing, not flat); losses too noisy to call (A / B / C 1k-windows 0.073-0.098). All arms were cut 6k -> 4k for the slice budget. So "failed dev screen" may partly be "not trained enough". Test queued (held for the cap): B 2k at s = 0.3 and B 2k cold on dev -> does the ghost / drift problem shrink or grow from 2k to 4k? The question goes to the council swap on the next video step (is longer training the fix?).
- Owner: wants to SEE the video. scripts/compare_video.py makes side-by-side MP4s from saved rollouts (game key | E3 no memory | A 4k | E3 + warm 0.7 | B 4k s = 0.3), 10 fps; jobs for scene-0234 (turn with parked cars) and scene-0396 (busy) queued (vid-*, held for the cap; first slot that frees).

## 2026-11-27 06:25 — OWNER PRIORITY + B dev screen: B fails at every s (lingering ghost cars)
- Owner (side chat, 06:20): "i need the jumps to go away and video to become safe and then further training to make the video smooth". -> top priority = the A / B / C verdict on the 30 decision drives (aim midday); no more rule polishing unless a reviewer finds an error that would change a result; keep the safety gates; car map stays closed; plan a longer training of the winner; one short council round on the next video step.
- B 4k s-lock (scripts/s_lock.py, mechanical; council/s_lock_B.json; panels and late windows OK): vs A 4k on 6 dev drives, ID 0.081-0.092 vs 0.099-0.107 (-9 to -23%), W 28.3-30.0 vs 31.0-32.0 (-5 to -9%), BUT car F1 -0.034 to -0.069, recall -0.017 to -0.055, own late LPIPS drift +0.044 to +0.073, late F1 -0.039 to -0.070, >= 5-tick ghosts +7.6 to +14.2 per run vs A. No s eligible -> s = 0.3 "failed dev screen" (rule). Same pattern as warm start: fewer jumps but lingering fake cars and drift.
- B decision rollouts at s = 0.3 started (3 shards) - scored for the record (failing allowed). C scoring running; A 4k decision scoring running.

## 2026-11-27 06:05 — drift check 10 outcome: s-lock made mechanical before any sweep score exists
- Both confirm the late-gap reading: the gate is the candidate's own drift (arm at s minus the same arm's cold rollout, ticks 150-169, 5 contiguous dev drives, <= +0.02 / F1 >= -0.03 as point estimates on dev); A's 0 is bookkeeping (--cold_ref B, A second). Against A + warm 0.7 later: --cold_ref none, the late gate read only from the comparison against A independent (Fable).
- scripts/s_lock.py (committed while 0 sweep score files existed) names the exact output field and threshold of every clause and picks s mechanically (score = max(ID.A/ID.B, W.A/W.B); ties within 0.01 -> largest s; none eligible -> 0.3 "failed dev screen").
- scripts/panel_check.py --late_cold (Astra): every contiguous dev drive must have finite late LPIPS and late car-F1 gaps on BOTH seeds against the arm's own cold folder; run ids printed; any gap blocks the lock.
- council/temporal-s-lock-rule.md updated (k = 3 settled; commands).

## 2026-11-27 05:50 — temporal dev sweeps rendered; scoring queued
- Rollouts done: B 4k s = 0 / 0.1 / 0.3 / 0.5 (dev, 12 runs each), C 4k s sweep (dev, 12 each), A 4k dev (12; scored, panel OK). Count list for the s-lock frozen from A 4k dev before any sweep is scored (council/count_A4k_dev_k3.json: 6 / 6 drives count).
- Scoring queued on scorer 764dd3ec95db: B cold + B sweep (afterok on B cold dev, 3753785-89); C cold + C sweep (3753790-94, held for the cap).
- Still rendering: A 4k decision (59/60), A 4k + warm 0.7 decision (43/60), B 4k cold decision (21/60), B / C cold dev (11 / 9 of 12). A 2k (+ warm) decision queued (held).

## 2026-11-27 05:10 — drift check 9 outcome (both)
- Car-map closure confirmed by both. Write-up wording (both): F reached about two thirds right-map wins, FLAT from 500 to 1k (63.4 -> 65.6%, 64.5 -> 66.7%; reductions flat), uneven across drives (t700: 18/19 on scene-0234, 2/7 on scene-0043); F also changed tmax (not architecture alone); X1 / X2 stopped at a 2k rendered-F1 rule on mirrored-yaw, pose-using boxes; R / F stopped at the 1k single-step rule; the planned F@1k pooled-18 renders were not run and R / F were never scored on renders, so no placement claim is made.
- Temporal (both): arms are compared at EQUAL updates - B / C 2k only against A 2k and A 2k + warm 0.7 (with count lists made from those) using B / C 2k cold; 4k against A 4k. Order (Fable): 4k to completion first, then 4k lag (--swap rollouts for every method; none exist yet, so nothing can pass until then), then 2k. Each sweep scored with --cold = the arm's own cold folder; compared with --cold_ref B, --floor 0, count list from A 4k dev.
- Added scripts/panel_check.py (exact drive x seed panel from data/temporal_lists.json before every comparison; Astra) and sweep tables will carry each rollout's sacct submit line (provenance; Fable).
- Queued (held for the cap): A 2k independent and A 2k + warm 0.7 decision rollouts (3753770-75).

## 2026-11-27 04:56 — car map F FAILS its 1k gate -> car-map track CLOSED (per the signed stop rule)
- F (bias-free 4-scale branch, tmax 962), single-step check on 6 dev drives / 93 car-bearing keyframes (degraded boxes):
  500: t700 right beats map-only-swapped on 63.4%, mean car-cell error reduction +4.3%; t900 64.5%, +3.9%.
  1k (gate): t700 65.6%, +4.5%; t900 66.7%, +3.2%. Gate needs >= 75% AND >= 2% at each t -> FAIL on the share (reductions pass).
  Unlike R, F does use the map (RMS(eps_right - eps_swapped) 0.016 / 0.0068; projection norms 0.60 / 0.90 / 1.51 / 1.27 at 900, grown from 0 steadily), but on only about two thirds of keyframes does the right map beat a wrong one.
- Stop rule (drift check 8, both): a 1k failure closes the car-map track - no third form, no retuned thresholds / LR. So: no X3; final = E3 + the memory winner (if any arm passes). Written up as: "the map signal was learned only partly in 1k updates (4% error cut, 2/3 of keyframes); the gate required 3/4".
- Remaining car-map jobs are records only (R dev18 renders, held); F dev18 renders not run.

## 2026-11-27 04:20 — F preconditions pass; F training released
- Branch-off: zero-projection F checkpoint vs E3, 2 dev drives x 4 keyframes x 2 seeds: max |pixel difference| 0 on all 16 renders (exact).
- F timing: 0.423 s / update, map prep 1.7 min -> projected 4k 32 min (budget 180). F (3753757) released: 1,000 updates, --arch branch4 --tmax 962.
- R 500 single-step (record): t700 48.4% right-better, +0.35%; t900 47.3% - no map-specific signal at 500 either.

## 2026-11-27 04:10 — drift check 8 outcome: last car-map arm F (bias-free 4-scale branch)
- Both: (a) run the named fallback once as the LAST car-map arm (F = mrv/carmap.CarMapBranchBF: bias-free, nine planes, zero-init projections, every step, both CFG passes), R's data / boxes / seeds / draws / batch / AdamW 1e-4 / 100 warm-up / dropout 50% -> 25% at 1k / equal whole + car-cell eps loss; R's gates unchanged (1k single-step >= 75% and >= 2% at each t; 2k pooled-18 beats E3 and correct - swapped >= 0.02); either failure closes the car-map track (no third form, no retuned thresholds).
- Recipe swap: (i) t = floor((1 - u^3) x 962), clamped at 961 (R spent ~34% of draws above the highest DDIM-25 step) - accepted by both; (ii) dividing each row's error by alpha-bar_t - Fable yes, Astra no -> NOT applied (rule: only changes both accept).
- Applied: projection norms logged separately; the branch runs in fp32 outside autocast in carmap_check.py and sd_render.py (as in training); checker refuses a branch whose empty-map output is not exactly zero; R written up as "not learned in 1k updates" (Fable: R's weight norm grew like a random walk).
- Submitted: branch-off render with a zero-projection F checkpoint vs E3 (cmF-off 3753755, 2 dev drives); F timing (3753756, --budget_min 180); F to 1k (3753757, held until the branch-off renders equal E3 exactly).

## 2026-11-27 03:55 — car map R FAILS its 1k single-step gate (stops per v4)
- runs/judge/cmR_1000_check.json (6 dev drives, degraded boxes, 93 car-bearing keyframes): t700: right map beats the map-only-swapped map on 59.1% of keyframes, mean car-cell error reduction +0.82%; t900: 53.8%, -0.41%. Gate: >= 75% and >= 2% at both t -> FAIL. RMS(eps_right - eps_swapped) 0.0105 / 0.0082 (vs eps RMS ~1): the map changes the prediction by ~1%, and barely more for the right map than for a wrong one. Map-conv weight norm grew to 0.57, which Fable shows is random-walk growth (x2.6 from 200 to 900 updates, vs x2.5 expected for noise), so the map was NOT learned in 1k updates; this does not show the map is useless (written up as "not learned in 1k updates").
- Per PLAN-carmap v4: R stops and is never scored on decision drives. Dev-only records still running (500 check; pooled-18 renders right / map-swap; E3 on the same 69 frames). Next step goes to the council (drift check 8): run the named fallback (bias-free 4-scale branch) or stop the car-map track.

## 2026-11-27 03:50 — arm C done at 4k; C dev sweep queued
- tmpC-b finished: C resumed from 2,500 with the C@2.5k cache (snapshot path + all clip ticks asserted), 4,000 updates, own_share 0.5, loss ~0.075. Checkpoints runs/tmpC/ckpt_2000 / 2500 / 4000.
- Queued (held for the cap): C 4k cold (independent) dev rollouts and C 4k history s = 0 / 0.1 / 0.3 / 0.5 dev rollouts. B sweep rollouts ~1.5 h in; B cold queued. R 1k single-step gate running.

## 2026-11-27 03:25 — comparison script SIGNED by both (fc7ee2b886a5)
- scripts/temporal_compare.py content hash fc7ee2b886a5 (git 1c8cc85) signed by Astra and Fable (council/compare-sign-fc7ee2b886a5-*.md), after five rejected revisions (ffaffe4425a1, c5369399a97c, cb720c354157, 8a190caa6de2, b690f024c74e). It recomputes the whole count list from the comparator's own files and refuses any difference; strict schema; python -O safe; one rollout per glob; comparator files / link / panel bound; --floor and k = 3 enforced; explicit --cold_ref; incomplete panels and missing late gaps on contiguous runs are hard failures. Scorer unchanged (764dd3ec95db). Decision count list vs independent: council/count_indep_dec_k3.json (28 / 30).
- Usage from now on: every decision comparison = temporal_compare.py --compare CAND COMP --count_list <list made from COMP> --floor 20 --cold_ref <side that is the cold reference>; dev s-sweep uses --floor 0 with a list made from A 4k dev.
- R reached 1k (map-conv weight norm 0.57, cam dropout 0.5, no clipping); single-step 1k gate check running.

## 2026-11-27 03:10 — compare script signature rounds; R running
- temporal_compare.py signature rounds: ffaffe4425a1 (both NO: count list not bound to its comparator; implicit zero late gap), c5369399a97c (both NO: list accepted for a differently-linked comparator), cb720c354157 (Fable SIGN; Astra NO: python -O strips the binding asserts; --floor without --count_list passed), now 8a190caa6de2 (git b0ca91e): explicit failures only, count list required for every comparison, list bound to the comparator's exact metric files + link mode + drive panel, one rollout per glob, provenance echoed. Signatures requested. No B / C decision score has been opened.
- R (arm R, 1k) running on an A6000 since ~03:00 (degraded fixed-calibration boxes, depth-tested faces, map cache keyed by the rasteriser hash). Queued (held for the cap, afterok on cmR): single-step checks at 500 / 1k on the 6 dev drives (gate at 1k), rendered pooled-18 at 1k with right map and map-only swap, E3 on the same 69 frames.
- C: cache from C@2.5k done; tmpC-b (resume to 4k, snapshot + tick asserts) waiting for an A6000. Rollouts (A, A + warm, B sweep) ~50 min in; B cold rollouts queued.

## 2026-11-27 03:20 — drift check 7: amendment 3 settled (k = 3 + floor), comparisons split into temporal_compare.py
- Final votes: floor 20 decision / 12 final per comparator, ID and W separately (both). k: Astra 1, Fable 3 after one swap -> stricter pair per my brief: k = 3 + floor (owner may revise).
- To keep the SIGNED scorer byte-identical (scripts/temporal_metrics.py = 764dd3ec95db, restored from fded34e), all comparison rules moved to the new scripts/temporal_compare.py (hash ffaffe4425a1, git 5c0d234): --make_count_list (comparator file alone, both seeds required), --compare with counting drives, --floor, missing-seed / zero-shared failures, hard fail on a missing late gap of a contiguous run (a rollout scored without --cold = its own cold reference). Signature requested from both.
- Counting list vs independent on the decision set (council/count_indep_dec_k3.json): 28 / 30 drives count for ID and W (out: scene-0036, 0 matched; scene-0014, no eligible). Re-check: warm 0.7 vs independent -> ID and W can_pass false (scene-0016 counting, zero shared); verdict unchanged.
- Fixes: C resume checks the snapshot path on every cache file (Astra); rollouts stamp the autocast dtype (Fable). Queued (held, cap): B 4k independent / cold rollouts dev + decision (3753725-28) (Fable: B's late gaps need B's own cold rollouts).

## 2026-11-27 02:20 — drift check 6 swap results; trained-arm rollouts started (not read yet)
- Converged (both): option (b) for drive counting; rank s by max(ID / ID_A, W / W_A); ties -> the LARGEST s (s = noise on the history, so larger = weaker history; my brief had the direction backwards). council/temporal-s-lock-rule.md rewritten with the agreed rule. OPEN for the next hourly check: counting threshold k (Astra 1, Fable 3; identical on known data) and Fable's floor (< 20 counting decision drives / < 12 final -> that comparison cannot pass). The counting rule then goes into --compare (non-swap edit -> both re-sign) before any B / C decision file is read.
- Rollouts on the 2080 Tis (A6000 queue busy; nothing read until the rule is signed): A 4k independent dev (3753705); B 4k history s = 0 / 0.1 / 0.3 / 0.5 dev (3753706-09); A 4k independent decision (3 shards) and A 4k + warm 0.7 decision (3 shards) (3753710-15; two held for the cap).
- cmR (arm R, 1k) waiting for an A6000; C cache from C@2.5k at 26 / 85 drives.

## 2026-11-27 02:10 — drift check 6 outcome; R timing passes; scene-0036 problem
- Both: warm 0.7 rejection upheld (third drive missing from ID = scene-0014, which has no eligible edge-car transitions). s-lock rule amended where they agree (recall >= -0.02, late F1, ghost growth and lingering ghosts in eligibility; an s with an eligible-but-no-shared dev drive is ineligible; both seeds on all 6 dev drives; lag not used; same s for 2k, 4k and every comparator; full sweep table committed with the lock). Swap running on ranking (worse-of vs mean) and tie direction.
- Applied: C resume asserts the cache snapshot path and every tick of all 128 clips (tmpC-b re-queued 3753702 with --cache_snapshot runs/tmpC/ckpt_2500.pt); R map cache key includes the rasteriser code hash.
- R timing (cmR-timing4): 0.458 s/update, map prep 1.8 min, save 1.2 s -> projected 4k 34 min (budget 180) -> passed; cmR 3753661 queued on an A6000.
- Fable's check: on decision drive scene-0036 independent matches 0 of 23 eligible ID transitions (both seeds); scene-0016 independent 5 and 3 of 17. Under "no shared transitions on an eligible drive cannot pass", no method can ever pass ID against independent -> amendment question sent to both before any B / C decision file is read.
- VPN dropped once (~02:00); reconnected via FortiTray; no jobs affected.

## 2026-11-27 01:55 — temporal frozen methods, DECISION SET: warm start fails; independent stays
- Scorer 764dd3ec95db (signed), 30 decision drives x seeds 0/1 (60 runs each), late gaps on 22 contiguous drives; warm 0.7 vs independent (paired drive bootstrap):
  ID 0.105 vs 0.136 (-22.5%, [-0.043, -0.019], 27 drives with shared transitions; scene-0016 and scene-0036 have eligible but no shared transitions -> the comparison cannot pass on that alone); W 35.3 vs 34.2 (+3%, [-0.6, +2.8]); recall 0.307 vs 0.357 (-0.050 [-0.080, -0.025]); car F1 0.258 vs 0.317 (-0.059 [-0.082, -0.041]); road 0.830 vs 0.853 (-0.023 [-0.031, -0.015]); late LPIPS gap +0.058 [+0.043, +0.073] (22 drives); late car-F1 gap -0.014 [-0.039, +0.010]; ghosts / tick 2.33 vs 1.90; >= 5-tick ghosts per run 13.7 vs 5.1 (+8.6 [+6.1, +11.6]); flips 0.171 vs 0.197 (descriptive). Link coverage 63.0% of 16,150 edge cars; W valid share 0.908.
- Rule: warm adopted only if ID AND W fall >= 15% (intervals excluding zero) and all gates pass -> W fails, car F1 / road / recall / late LPIPS / lingering ghosts all fail -> warm start NOT adopted. Best qualifying frozen method = independent. Picture: pulled/warm_dec_sheet.jpg (warm 0.7 adds lingering cars and degrades the own car on scene-0105).
- C: at 2,400 (own_share 0.5). Cache from C@2.5k (tmpC-cache2500) and resume to 4k (tmpC-b, --cache_step 2500) chained with afterok (exit codes now real).

## 2026-11-27 01:25 — wrapper exit-code bug; P3 on all 30; R and C resubmitted
- BUG (all our py.sbatch jobs so far): the wrapper printed "rc=$?" after $(date), so it logged and RETURNED the exit code of date (0), not of python. Failed jobs looked COMPLETED and afterok chains went ahead regardless. Fixed (rc captured right after python, exit $rc). Known affected tonight: cmR-timing2 / timing3 crashed (BrokenProcessPool) but showed COMPLETED; cmR was held, so nothing ran on them. Earlier chains were checked by log lines (TMETRICS_DONE / ROLLOUT_DONE etc.) before use.
- Map cache: the parallel rasteriser kept crashing (BrokenProcessPool, most likely memory in forked workers); now single-process with an on-disk cache keyed by the box file (tmp/carmap_maps_<crc>.pt). Timing re-run cmR-timing4 3753691 (--budget_min 180, exits 3 if over); cmR 3753661 released, afterok on it.
- P3 hull F1 on all 30 seeded train drives (scene-0705 has one keyed frame: 117 frames, runs/judge/keyidx_train30): GT 0.317, detector raw 0.331, degraded 0.290; val84 GT 0.411, detector 0.380. Raw detector gap -0.049 vs GT gap -0.094 -> exceeds by +0.045 (< 0.05); degraded -0.090 vs -0.094 (matched).
- Arm C: rollcache_1000 complete (85 drives, 128 clips); files stamped with snapshot runs/tmpB/ckpt_1000.pt / step 1000 from the job record (made before the stamp existed); tmpC-a 3753692 with --cache_step 1000 (to 2,500).

## 2026-11-27 01:20 — drift check 5 outcome (both)
- Correction to the 00:00 entry: the 11 short calibration drives have ONE keyed frame each (not 3), so val84 = 73 x 4 + 11 = 303 frames is complete (Fable counted on the cluster).
- Both: R may start only after (Fable) the pooled-18 frame list is fixed and committed and the timing gate is mechanical, and (Astra) P3 hull coverage is documented for all 30 train drives and the measured 4k budget fits. Done / running:
  - council/carmap_dev18_frames.json (generator scripts/carmap_dev18_frames.py): 18 drives (6 dev, degraded boxes; 12 reserved, raw boxes), 69 frames (scene-0771 has one keyed frame, used once), seeds 0/1, fixed map-swap donors (drive half the list away, same relative frame). sd_render.py --frames_file renders exactly that list (token checks, count assert).
  - train_carmap2.py --budget_min 180: projects map prep + 4,000 updates + checkpoints + load from the measured steps 10-100 and exits 3 if over; cmR-timing3 3753686 (degraded fixed-calibration boxes); cmR 3753661 afterok on it, still held until the all-30 P3 hull recompute is in (keyidx-train30 3753681 -> raw 3753682, degraded 3753683; scene-0705 has one keyed frame, so 117 frames).
  - Arm C: cache files now record the snapshot (path, step); train_temporal.py --cache_step asserts it and that the cache covers all 85 clip drives. tmpC-a cancelled; rollcache_1000 (made before the stamp) will be stamped from its job record (runs/tmpB/ckpt_1000.pt, step 1000) and C resubmitted with --cache_step 1000.
- Temporal rules (both): sweep B / C strength s on the 4k checkpoints only, with the lock rule written before reading; 2k scored with the same s; warm 0.7 stays the A + warm comparator; no method declared passing until the owner rules on the lag position (failing allowed); state 30 drives (22 for late gaps) beside decision figures.

## 2026-11-27 01:00 — temporal scorer SIGNED (764dd3ec95db, both); decision scoring released; C queued
- Astra and Fable both signed scorer content hash 764dd3ec95db (git fded34e) for non-swap rollouts (council/scorer-sign-764dd3ec95db-*.md). Fable's reading conditions: "ghosts not growing" = paired difference vs independent on BOTH ghost growth and >= 5-tick ghosts (growth alone does not separate warm 0.5); state the drive count beside every late gap / ghost growth (decision set should be 22); flips descriptive only. Lag reported under the interim rule (signed for reporting). Report 65.6% dev link coverage beside ID / W.
- Released the six decision metric jobs (independent and warm 0.7, 30 drives x seeds 0/1, 60 rollouts each, on 764dd3ec95db).
- Arm C queued: fork of B@1k with data/rollcache_1000 (cache 69/85 drives done), to 2.5k (tmpC-a, afterok on the cache job); then cache from C@2.5k and resume to 4k.

## 2026-11-27 00:50 — dev controls on the candidate scorer (764dd3ec95db); signatures requested
- runs/troll/v4_*_dev.json vs independent (6 dev drives, seeds 0/1): ordering holds (key video ID 0 / W 11.4; VAE ID 0.064 vs 0.110 / W 13.7 vs 29.4; fresh noise ID 0.157 vs 0.102 / W 41.1 vs 30.1).
- With placement now on manifest ticks % 5: warm 0.7 car F1 0.423 vs 0.485 (-0.062 [-0.113, -0.014]) and road 0.797 vs 0.854 -> fails the placement gates on dev; it also leaves lingering ghosts: >= 5-tick ghost tracks 10.3 per run vs 2.4 (independent), ghosts / tick 2.25 vs 1.61. warm 0.5: 30.1 lingering ghosts per run.
- Signature request council/scorer-sign-764dd3ec95db.md sent to both. On two signatures: release tmet-indep_dec0-2 / tmet-w07_dec0-2.
- C cache from B@1k at 63 / ~85 drives. Car map timing + R (1k) waiting for an A6000.

## 2026-11-27 00:20 — car map R preconditions met (timing pending); R staged to 1k queued
- Branch-off = E3 on the same noise: zero-weight map conv vs E3, 2 dev drives x 4 keyframes x 2 seeds = 16 renders: max |pixel difference| = 0 on all 16 (exact).
- P3 hull F1 complete (fixed-calibration boxes, yaw fixed, Mask2Former key masks, score >= 0.3): train30 = 29 keyed drives x 4 = 116 frames: GT 0.313, detector raw 0.325, detector degraded 0.283; val84 = 303 frames (73 drives x 4 + 11 drives that have a single keyed frame; nothing skipped): GT 0.411, detector 0.380. Raw detector train - val gap -0.055 vs GT gap -0.098 -> exceeds by +0.043 (< 0.05: no hull trigger); degraded gap -0.097 vs GT -0.098 (matched). Recall / precision / centre triggers already fired -> degradation in place (tmp/bevF_boxes_trainD.json).
- Arm R queued: train_carmap2.py --boxes tmp/bevF_boxes_trainD.json --steps 1000 --out runs/cmR (cmR, afterok on the timing run 3753654). Then single-step gate at 500 and 1k (6 dev drives, degraded boxes); pooled-18 rendered F1 recorded at 1k.
- Temporal: dev controls on the candidate scorer (tmet4) 5/6 running; C cache from B@1k running.

## 2026-11-27 00:00 — drift check 4 outcome; scorer candidate fded34e (content sha 764dd3ec95db)
- Fable (council/drift-20261126-2355-fable.md): signs the scorer for NON-swap rollouts once (1) per-tick ghosts stored and compared, (2) cold-tick / consecutive-tick asserts, (3) scorer version + link mode stamped and checked by --compare, (4) dev controls re-run on that exact version with the ordering holding, (5) later edits only inside the swap block. Lag block not signed until windows at / after the first post-swap reset are ineligible. Lag position final: Fable centre, Astra end -> interim outcome rule (Fable): median START <= 1 passes under both, > 3 fails under both, in between = lag undecided (owner rules).
- Implemented all of it in commit fded34e (scorer content hash 764dd3ec95db stamped in every output; --compare refuses mixed / mismatched versions or link modes; ghost_per_tick, ghost_growth (late 150-169 minus early 5-24 window), ghost_ge5 in --compare; cold rollout tick list asserted equal; drift window requires 170 consecutive ticks; lag windows only before the first post-swap reset). Dev controls re-run on it: tmet4-* 3753641-46. Then both signatures on 764dd3ec95db before releasing the six decision metric jobs.
- P3 hull F1 (fixed-calibration boxes, raw; yaw fixed): train30 (29 keyed drives x 4 = 116 frames) GT 0.313 / detector 0.325; val84 first pass 292 frames (11 drives with only 3 keyed frames were skipped - fixed: they use all 3; re-run 3753652 -> 325 frames expected): GT 0.411 / detector 0.376. Gap check (first pass): detector train - val -0.051 vs GT -0.098 -> detector gap exceeds GT gap by +0.047 (< 0.05). Degraded train boxes hull F1: 3753653.
- Arm R timing crashed (BrokenProcessPool: map cache used the node's CPU count); fixed to the job's CPUs; re-run cmR-timing2 3753654. Branch-off: cmR-off done, E3 reference render running.

## 2026-11-26 23:57 (real) — drift check 4: Astra does not sign the scorer yet; fixes applied
- Astra (council/drift-20261126-2355-astra.md): signed linking fallback correct (report 65.6% coverage with ID / W); NO SIGN on the scorer until: placement frames chosen by manifest tick % 5 (not every 5th accepted entry); ghost LIFETIMES stored (the plan's ">= 0.5 s exited-car" veto needs them); lag: window end (final word), five consecutive distinct donor ticks, no reset inside; swap rollouts must not repeat the donor's last bundle. Also: P3 hull F1 (verify 120 train / 336 calibration frames) and exact same-noise branch-off before R's update 1. Disclosure: the 0.927 rasterizer coverage comparison used 30 DECISION frames (boxes only, no renderer, no model choice); the picture check used dev frames.
- Applied (plan compliance): temporal_metrics.py placement on kt % 5 == 0 (placement_ticks stored); ghost tracks (unmatched render cars linked across adjacent ticks at IoU >= 0.5) with lifetimes and a count of >= 5-tick ghosts; lag windows need 5 consecutive distinct donor ticks and no restart inside; temporal_rollout.py stops at the end of the donor drive (no repeated last bundle). Waiting for Fable; then dev controls re-run on this revision for both signatures.

## 2026-11-26 23:50 — car map P3 redone on fixed-calibration boxes; degradation re-fit on the 84
- P3 (scripts/bev_leak.py --val_scenes, runs/judge/bevF_leak84.json): 30 seeded train drives (1,207 frames) vs the 84 calibration drives (3,378 frames): recall 0.846 vs 0.726, precision 0.784 vs 0.722, centre error 0.56 vs 0.64 m -> leak flag (as before). By distance recall 0.98/0.88/0.70 vs 0.94/0.79/0.51.
- Degradation re-fit on the 84 (tmp/bevF_boxes_trainD.json; drop 3.7/10.2/27.0%, sigma 0.15/0.26/0.42 m, phantoms 0/0.05/0.24): degraded train30 vs 84 recall 0.94/0.78/0.51 vs 0.94/0.79/0.51, precision 0.94/0.80/0.49 vs 0.96/0.81/0.48, centre error 0.63 vs 0.64 m, flag off (runs/judge/bevF_leak84_degraded.json).
- P3 hull F1 part: scripts/key_frames_cache.py (judge frames + Mask2Former key masks for key-only drives) -> bevbox-P3val84 (3753631 -> 3753632) and bevbox-P3train30 (3753635; first submission used one box shard only, cancelled). bev_box_judge.py now merges comma-separated box globs and works without renderer scores.
- Branch-off = E3 on the same noise: zero-weight map conv checkpoint (runs/cmR_zero.pt) vs E3 on 2 dev drives (cmR-off 3753636, cmR-E3ref 3753637). Arm R timing (3753600) running on an A6000. tmpB done (4k). C cache from B@1k (3753630) waiting for an A6000.

## 2026-11-26 23:55 — flow-warped linking fails its bar; signed linking kept (precise, 66% coverage)
- Dev re-scored with --link flow (tmet3, 6 methods): edge-car link share 0.759 with flow-warped IoU vs 0.656 signed (5,112 edge cars); W valid-pixel share 0.912. Ordering still holds (key video ID 0 / W 11.6; VAE ID -41% / W -53%; fresh noise ID +54% / W +37%; warm 0.7 ID -17% [-0.025, -0.009], W +0.04 [-1.2, +3.6], road -0.058; warm 0.5 collapses). 0.759 < the agreed 0.80 bar -> amendment 3 NOT adopted; keep the signed linking and state its coverage (both reviewers' rule).
- Hand check of the SIGNED links on the two traffic-bearing dev turns (pulled/linkcheck.jpg; 24 random of 68 edge-car links on scene-0234 t95-140 and scene-0285 t90-135): 24 / 24 link the same car, no identity switches; some masks bleed onto an occluding car (#7, #16). So signed links are precise; the limit is coverage (66% of edge cars).
- Arm C cache from B@1k (data/rollcache_1000, tmpC-cache1000 3753630, A6000) submitted; tmpB at 3,300; tmpA done (4k).
- bevF-otherval and bevF-final done (fixed calibration); P3 re-fit next.
- Next: ask both reviewers (hourly check) to sign the frozen scorer (signed linking + fixes) so the held decision metric jobs can run.

## 2026-11-26 23:35 — drift check 3 swap results
- Converged: lag threshold = 0.8 x the donor's cold car F1 on the SAME five donor ticks (same seed, 0.3 matcher), zero-reference windows skipped, full windows incl. the last, no eligible window -> run excluded and counted (both); gradient clip 1.0 kept in R (as E3 / X1), share of clipped updates reported (both). Implemented (temporal_metrics.py lag block; --lag_ref is now the cold rollout folder; train_carmap2.py logs clipped_share).
- Not converged after one swap: lag gating position, Astra = window end (no future ticks), Fable = window centre (end makes a perfect render read 4). Scorer reports start / centre / end; gating position goes back to the next hourly check. No lag test has been run yet, so nothing waits on it.

## 2026-11-26 23:25 — drift check 3 outcome (both answers in); swap on 3 points
- Agreed by both (applied / running): decision metric jobs 3753219-24 HELD (they would have run today's scorer); amendment-3 candidate = key tracks linked by flow-warped Hungarian IoU (t-1 key masks warped to t with RAFT key flow, >= 0.5; render-to-key stays 0.3), accepted only if on dev: linked edge share >= 80%, >= 18 / 20 hand-checked new links right on two traffic-bearing turns (entries / exits / dropouts), ordering check passes again; else keep the signed linking and state coverage. temporal_metrics.py --link flow added (both link rates, W valid-pixel share and new-link examples always reported). Dev re-scored with --link flow: tmet3-* 3753609-14.
- P3 on the seeded 30 train vs the 84 (both; Fable withdrew the 12); the 12 only as a report-only check of degraded train vs raw unseen boxes.
- Rasterizer: z-buffered faces accepted by both subject to a picture + old-vs-new on dev frames: pulled/rastercheck.jpg (4 dev keyframes with 43-45 boxes): boxes sit on the key's cars; parked rows now get per-box edges; per-pixel inverse depth. Frozen from here for R. Hull judge (bev_box_judge.py) unchanged so 0.397 / 0.437 stay comparable.
- Swap sent (council/drift-20261126-2255-swap.md): lag tick (window end vs centre), lag reference on the same ticks, grad clip in train_carmap2.

## 2026-11-26 23:10 — drift check 3 (Astra in; Fable pending): face-level depth in the rasterizer
- Astra (council/drift-20261126-2255-astra.md): keep temporal ranking paused; approve testing flow-warped Hungarian IoU linking at 0.5 (render-to-key matching stays 0.3); hand-label links on two traffic-bearing dev turns before opening decision scores; P3 val side = the 84; lag counts to the window END (and include the last full window); face-level depth is required by PLAN-carmap v3 ("far-to-near, depth-tested faces"); fixed-raw timing OK; R must start fresh, assert all 144 drives / tokens.
- Applied now (plan text, both reviewers had the plan): mrv/carmap.rasterize rewritten with a per-pixel z-buffer over the 6 faces (12 triangles) of every box; nearest surface owns each pixel; inverse depth per pixel; instance boundary from the id map. vs the old mean-depth hulls on 30 decision frames: coverage IoU 0.927, per-pixel depth changed on 28% of covered pixels; 132 vs 45 ms / frame -> train_carmap2.py rasterises every training map once (parallel) and caches it (deterministic: fixed degradation, no box dropout); asserts all 144 drives and zero missing keyframes; refuses to resume an existing ckpt without --resume.
- Waiting for Fable on: flow-warped linking, the 84 vs 12, lag window end.

## 2026-11-26 22:50 — temporal metric check: ordering passes (fresh noise worst); car map R timing
- Fresh noise every tick vs independent (dev, fixed scorer): ID 0.157 vs 0.102 (+54% [+0.048, +0.063]); W 41.1 vs 30.1 (+37% [+6.5, +15.9]); flips 0.240 vs 0.191 (+0.049 [+0.038, +0.063]); car F1 0.470 vs 0.474; road 0.851 vs 0.852. With the key video (ID 0, W 11.4) and VAE round trip (ID -42%, W -54%) the ordering required by the stop rule holds: keys / VAE best, fresh noise worst. Open: track reliability (66% of edge cars linked at IoU >= 0.5) -> council next hour. Decision metrics still unread.
- Fixed-calibration train + dev box caches done (bevF-train-0..3); other-val and final released. Arm R 100-update timing on fixed-calibration raw train boxes (cmR-timing 3753600, A6000).
- tmpA at 3,100 / 4,000 (0.54 s/update).

## 2026-11-26 22:20 — car map: camera-only P2 passes with fixed calibration
- Fixed calibration (camera->lidar from the rig only; max change vs the pose-using converter 0.560 m) + can_bus zeroed, yaw fixed, decision drives, hull F1 vs game-key cars (score >= 0.3, 120 frames): with previous BEV 0.437, without 0.437; GT boxes 0.459; gate 0.397 -> PASS (above Fable's 0.431 council band). Mode kept: previous BEV on.
- Released the fixed-calibration train / dev box caches (bevF-train-0..3); other-val (P3 re-fit) and final follow as slots free.

## 2026-11-26 22:30 — temporal metric check: turning clips + track reliability
- Turning clips (pulled/turncheck.jpg; strongest 2 s of ego yaw per dev drive: scene-0393 51 deg from tick 40 - no cars in view, uninformative; scene-0234 31 deg from tick 105): key masks mostly right, but on scene-0234 whole ticks lose nearly all masks (t106, t112) and small far cars get a new track id at almost every tick.
- Measured over all 6 dev drives (969 ticks, tmp/trackstats.py): mask-count dropout ticks (<= half the neighbours' median, >= 2 cars) 19 = 2.0%; edge-car masks linked to the previous tick at IoU >= 0.5: 1,676 / 2,556 = 66% (per drive 0.50-0.90); best IoU to the previous tick median 0.64, >= 0.3 for 78%. So about a third of edge-car transitions never enter ID / W, mostly fast-moving near cars at the edges. Same keys for every method (coverage loss, not a method bias), but the stop rule names "unreliable masks / tracks". Ranking stays paused; question goes to the council next hour (e.g. flow-warped IoU linking).

## 2026-11-26 22:15 (real clock) — drift check 2 outcome (council/drift-20261126-2156-*.md)
- NOTE: PROGRESS headings "21:05" ... "22:45" above ran ahead of the real clock (estimates); order and content are right.
- JOB CAP BREACH found by Astra: 16-17 project jobs running at ~21:03 (I submitted the dev re-score, P5 check and 2k re-renders without re-counting; the name filter also missed some job names). Fixed: slurm/release_upto.sh counts project jobs by working directory (~/mini-reality-view), active = running + pending-not-held; tmpB held so at most 12 can run (11 now).
- Both reviewers, applied: history checkpoints (B / C) now always run through the trained history conv with the no-history input (zeros + plane 1) for cold / independent renders; lag: full five-tick windows, unrecovered switches kept (inf in the median), median reported; --compare requires seeds 0 and 1 on every drive; arm C must restore B's optimiser (no silent restart); C cache history noise on its own stream; carmap_check asserts full box coverage; --fixed_calib implies run-time can_bus zeroing (the dataset rebuilds can_bus from ego pose, so the info-level zeroing did nothing; all P2/bevF jobs had --zero_canbus anyway); comma-separated box globs.
- Rulings: temporal cut to 4k approved by both (C@4k identical to the signed C@4k; Fable notes batching cache clips could restore 6k - optional, not done); gate at both t (both); degraded boxes for the 6 dev drives in BOTH the single-step and rendered checks, raw for the 12 reserved, 12 reported alone beside the pooled 18 (both); Mask2Former key-mask misses accepted and to be disclosed, SegFormer stays as the independent check (both).
- Hull gate: keep 0.397 as signed; a fixed-calibration result between 0.397 and 0.431 (yaw-fixed baseline 0.451 - 0.02) goes back to council (Fable).
- Open (to swap next hour): P3 hull F1 val side = the 84 calibration-fit drives (Astra; all have keys) vs the 12 reserved (Fable); lag window definition (Fable: start-of-window lag reads up to 4 ticks low); rasterizer mean-depth hull order vs face-level depth (Astra); arm R timing with fixed-calibration inputs only (Astra) vs timing-only on old boxes (Fable). Decision metrics stay unread until fresh-noise + turning-clip checks pass (both).

## 2026-11-26 22:45 — car map R signed (PLAN-carmap v4); temporal arms cut equally to 4k
- Car map round 3: Astra and Fable both signed (council/carmap-r3-astra.md, -fable.md); written into council/PLAN-carmap.md as the v4 amendment (arm R: bias-free zero-init 3x3 map conv after conv_in on frozen E3; 50/50 whole + key-car-cell loss; cubic-t; camera dropout 50% to 1k then 25%; single-step dev gate at 1k, rendered 18-drive gate at 2k; fixed-calibration caches + yaw fix + P2/P3 redo before adoption).
- 12 reserved calibration drives drawn by seed (tmp/carmap_dev12.txt: scene-0331, 0346, 0632, 0771, 0783, 0904, 0925, 0928, 0929, 0967, 0972, 1071); the degradation re-fit uses the other 84 (tmp/bev_otherval84.txt). Their game keyframe keys already exist (data/sim/keys_tpv), so no rendering needed.
- Yaw-fixed hull F1 on the old (pose-using, can_bus-zeroed) decision boxes: detector 0.451 (was 0.444 mirrored), GT 0.459.
- Code: scripts/train_carmap2.py (arm R), scripts/carmap_check.py (single-step dev gate), scripts/key_carmasks.py (key car cells; keycar-0/1 3753531/32 running, dev 3753533 held), sd_render.py renders map-conv checkpoints; mrv/carmap.MapConvIn.
- Fixed-calibration caches queued (held for the cap): bevF-train-0..3 3753525-28, bevF-otherval 3753529, bevF-final 3753530; P2 with fixed calibration running (3753521/22).
- Temporal timing (plan: 180 slice-minutes per arm incl. caches; cut all arms equally): training ~0.54 s/update on A6000; one C cache ~ 22 s per clip -> ~47 min + ~4 min load per 128-clip cache. C at 6k with caches at 1k/2.5k/4k ~ 54 + 153 = 207 min > 180. Cut all arms equally to 4k updates (equal comparisons at 2k / 4k as planned); C caches at 1k (from B) and 2.5k only (~36 + 102 = 138 min). A / B resubmitted at 4k: tmpA 3753534, tmpB 3753535.

## 2026-11-26 22:10 — car map round 2: answers in, swap sent; yaw bug confirmed + fixed; camera-only fix started
- Round 2 answers: council/carmap-r2-astra.md, council/carmap-r2-fable.md; swap brief council/carmap-r2-swap.md (both agree: loss gives little signal where the map matters, cameras make it redundant -> heavy early camera dropout; early checks on dev; decision drives only for one locked scoring; camera-only fix + repeat hull gate / P3 before adoptable training. Differ on injection point, loss weighting, timestep sampling, arms; swap round running).
- Fable's yaw finding CONFIRMED (tmp/yawcheck.py, 1,272 detector-GT car pairs on 250 decision frames): as coded, footprint IoU 0.620 and heading within 15 deg of GT (full circle) for 19%; with yaw -> -yaw - pi: IoU 0.667, 96% within 15 deg (median 2.3 deg). mmdet3d 0.17 boxes were mirrored and reversed in mrv/carmap.box_corners_ego and scripts/bev_box_judge.py. Fixed (frames carry yaw_conv; gt_frame = "carmap"). X1/X2 were trained on mirrored hulls / reversed heading channels; the 0.417 / 0.444 / 0.441 hull numbers and P3 were computed with mirrored hulls.
- Camera-only (Astra): BEVFormer's camera->lidar transforms came from the converter, which chains per-camera timestamped ego poses (lidar-based localisation). bev_extract.py --fixed_calib builds them from the rig calibration only (the same formula with the two ego poses equal) and zeroes can_bus (also still zeroed at run time).
- Submitted: P2 again with fixed calibration + zeroed can_bus, with (bevP2-fixed 3753521) and without previous BEV (bevP2-fixednoprev 3753522); hull judge with the yaw fix on the old camera-only boxes (bevbox-yawfix 3753523). Gate 0.397.

## 2026-11-26 21:45 — car map 2k (fixed renders): stop rule fires validly; round 2 council
- 2k, fixed renders (30 decision drives x 4 keyframes x seeds 0/1, Mask2Former, paired drive bootstrap):
  X1 car F1 0.315 vs E3 0.318 (-0.002 [-0.030, +0.025]); map-only swap 0.323 (correct - swapped -0.007 [-0.024, +0.009]); road IoU 0.856 vs 0.848 (+0.008 [+0.003, +0.013]).
  X2 0.322 (+0.004 [-0.024, +0.035]); map-only swap 0.334 (correct - swapped -0.013 [-0.034, +0.009]); road 0.855 (+0.007 [+0.002, +0.013]).
  SegFormer: X1 0.307 vs 0.302, X2 0.302 vs 0.302.
- Map-only swap moves car F1 < 0.02 -> both arms stopped (valid now). Picture pulled/cm2k_sheet_fixed.jpg: the branch now changes road edges and car shapes, but the right and wrong maps give near-identical changes.
- Back to council: council/carmap-r2-brief.md sent to Astra and Fable (round 1, independent answers -> carmap-r2-astra.md / -fable.md), incl. the open camera-only point.
- Temporal timing: arm A 100 updates on an A6000 in ~0.9 min (~0.54 s/update, 6k ~ 55 min); B timing running; C cache shard timing (3753245) next.

## 2026-11-26 21:25 (cluster clock; earlier headings tonight ran ~25 min ahead) — fixes verified
- Car map render fix verified (P5-style, X1 2k, scene-0003 frames, same noise as E3): mean |render - E3| 3.9-32.3 /255 with the fix vs 0.3-2.7 with the bug. 2k dec + mapswap re-renders running (3753241-44).
- Dev re-scored with the fixed scorer (equal drive weights, shared transitions, valid backward warp, contiguous drift; 6 drives, 5 with >= 170 contiguous ticks):
  - warm 0.7 vs independent: ID 0.083 vs 0.101 (-18%, [-0.027, -0.010]); W 30.8 vs 29.8 (+0.99 [-1.24, +3.31]); recall -0.032 [-0.085, +0.017]; car F1 -0.045 [-0.102, +0.006]; road IoU -0.058 [-0.106, -0.012]; late LPIPS gap +0.086 [+0.038, +0.137]; late car-F1 gap -0.064 [-0.134, +0.004].
  - VAE round trip vs independent: ID -42%, W 13.7 vs 29.4 (-54%), late LPIPS gap -0.147, late F1 gap +0.188. Key video: ID 0, W 11.4 vs 30.0. Ordering as required; fresh noise still pending (3753158, behind the decision rollouts for the cap).
- --compare: a method's gap vs its own cold reference counts the reference as 0 (independent rollouts are the cold reference).

## 2026-11-26 21:50 — car map diagnostic (partial, before it was cancelled)
- scripts/carmap_diag.py ran on the 120 decision keyframes (reviewers: should have been dev; cancelled after X1 2k/4k and X2 2k; numbers kept for information only, not used for any decision). Single-step noise prediction is NOT affected by the render bug, so this speaks to training: at every t (100-900) eps-MSE with the right map equals the swapped map to within 1e-4 (e.g. X1 4k t=300: right 0.10658, swapped 0.10669, off 0.10627; X2 2k t=700: 0.01880 / 0.01882 / 0.01891); on map-covered cells right vs off differs by < 1% (X1 4k t=300: 0.21911 vs 0.22037). The branch's change to the prediction is 1-4% RMS and almost the same for right and swapped maps. Expectation: the fixed renders will still show a small map effect; the 2k rule decides.

## 2026-11-26 21:40 — drift check 1 outcome (Fable): car map renders were broken; arm C draw fixed
- Fable (council/drift-20261126-2055-fable.md) found a real bug, confirmed on the cluster: diffusers 0.43 pops the down_intrablock_additional_residuals list it is given (unet_2d_condition.py:1141/1155/1190). sd_render.py built RES once per image and passed the same list to all 25 DDIM steps, so the car map acted on step 1 only and steps 2-25 were plain E3. Every car-map render so far is invalid: X1/X2 2k "dec" and "mapswap", pulled/cm2k_sheet.jpg, and therefore the 2k stop decision. Training and carmap_diag.py rebuild the list per call (unaffected).
  - Fix: a fresh list per step; sd_render.py also now refuses to render a frame with no boxes (was silently "map unavailable").
  - Old renders / scores moved to runs/judge/old_popbug/. Re-run: P5-style check (X1 2k, scene-0003 + scene-0012, 3753240: must differ from E3 on the same noise), then 2k dec + mapswap for X1 (3753241/42) and X2 (3753243/44). The unchanged 2k rule is re-applied to these; 4k stays closed unless 2k passes; the redesign council is on hold until then (Fable).
  - Car-map diagnostic on decision data cancelled (both reviewers: diagnostics belong on dev drives).
- Arm C (both reviewers): own-history rows were ~2% (uniform dense draws, own frame only on cache hits) instead of half. Fixed in train_temporal.py for ALL arms (identical targets): 128 non-overlapping eight-tick train clips preselected once (seeded, data/temporal_cclips.json); every batch draws 6 of its 12 dense rows from those clips' history rows and 6 from all dense rows; C replaces exactly those 6 rows' history with its cached own frames (asserts the cache hit; logs own_share). Checkpoint 2,500 added (C's second cache). New scripts/temporal_ccache.py (DDIM 25 / CFG 2, s = 0.3, first clip tick without history). Timing: one cache shard (1/16) after the B timing run (3753245, held for the cap).
- Both reviewers: do not read decision metrics until the dev metric check passes (fresh noise, turning clips, flow share). Agreed.
- Open, Astra only (to put to Fable next hour): BEVFormer camera-only - CAN bus is zeroed, but its projection matrices are built from per-camera timestamped ego poses; P2 / the hull gate may need repeating with fixed calibration. Also P3 hull F1 omitted; rasterizer orders hulls by mean depth.

## 2026-11-26 21:20 — drift check 1: Astra's review applied to the temporal scorer
- Astra (council/drift-20261126-2055-astra.md): splits/seeds/stop rules/cap OK; P3 used 96 calibration drives (3,862 frames), not 150 as I wrote. Main finding: the temporal scorer did not match the signed plan -> fix before ranking. Applied (plan compliance, not plan changes):
  - --compare: equal drive weights (seeds pooled inside a drive), paired seeds only (unpaired drives listed), ID / W per drive on shared transitions, drives with eligible but no shared transitions listed and the comparison marked can_pass = false. (Old pooled-transition W for warm 0.7 vs independent 25.7 / 25.7; Astra's equal-weight recompute 31.1 / 29.8.)
  - W: backward warp with flow i -> j on the keys (was forward flow read at destination coordinates), valid = in-bounds and forward-backward consistent; valid / total car-pixel counts stored per transition.
  - Drift: only rollouts whose history chain is unbroken through tick 169 (no restart in ticks 1-169); late car-F1 gap added beside the late LPIPS gap (and early gaps).
  - Lag: rollouts now store ok / hard / tick of the bundle actually fed (the donor's after the swap); metrics use the donor tick for keys.
- Old dev metric JSONs moved to runs/troll/v1_old/; dev re-scored with the fixed scorer: tmet2-indep_dev 3753234, warm07 3753235, keyvae 3753236, keys 3753237, warm05 3753238 (held, cap). Decision metrics (3753219-24, pending) will run the fixed code.
- Still open from Astra: camera-only for BEVFormer (zeroed CAN bus but per-camera ego poses still enter its projection matrices), P3 hull F1 omitted, P5 same-noise branch-off render skipped, rasterizer paints mean-depth hulls, the car-map diagnostic used decision data (move to dev), arm C fills fewer than 50% own-history rows and C's cache shard must be timed before A/B 6k. Waiting for Fable before acting on these.

## 2026-11-26 21:05 — temporal metric check: hand mask check; lag reference fixed
- Hand check of 20 random cached key masks (dev drives; pulled/maskcheck.jpg): no false car masks seen; the red/blue/purple/yellow sports-car models and parked rows are masked well; MISSED consistently: the white/silver sports car (3), boxy SUVs / jeeps (~5, likely read as truck by the Cityscapes model), one white car beside a masked car. Rough key-mask recall ~80-85% of visible cars, precision ~100%. Same segmenter on render and key, so missed models simply do not enter ID / W / recall (coverage loss, not bias toward a method). Noted for the council.
- W coverage: transitions with a valid W (car pixels in the outer 20% columns) are ~80% of the ID transitions (e.g. 867 vs 1031 shared, 1149 vs 1396).
- temporal_metrics.py lag now uses the plan's reference: 80% of the DONOR drive's cold car F1 (same seed) from --lag_ref (independent rollouts' metrics); refuses to run without it; zero reference -> no lag value. (Was a fixed 0.3.)
- Hourly drift check 1 sent to Astra and Fable (council/drift-20261126-2055.md).

## 2026-11-26 20:50 — temporal: metric check part 2; warm 0.5 out; decision rollouts
- Metric check (dev): VAE round trip vs independent: ID 0.058 vs 0.097 (-41%), W 14.6 vs 27.7, recall 0.72 vs 0.43, car F1 0.75 vs 0.47; flips 0.21 vs 0.19 [-0.07, +0.13] (small blurred cars toggle at IoU 0.3, not significant). Key video and VAE round trip both rank above the model on the co-primaries as required; fresh noise pending (3753158).
- Warm 0.5 on dev: car F1 0.304 vs 0.474 (-0.170), road IoU 0.693 vs 0.852, late LPIPS gap +0.281, recall -0.10. Collapses (matches the striped frames seen by eye): dropped. Warm 0.7 is the dev-picked warm setting.
- Decision set (30 drives x seeds 0/1) rollouts: independent 3753213/15/17 and warm 0.7 3753214/16/18 (3 shards each, 2080 Ti); metrics sharded (temporal_metrics.py --shard; --compare merges a glob): 3753219-24.
- Car map diagnostic running on an A6000 (3753196); duplicates cancelled.

## 2026-11-26 20:40 — temporal dev numbers; lit singles; timing runs
- Frozen methods on the 6 dev drives (12 rollouts, E3 10k), warm 0.7 vs independent (paired, 6 drives): edge-car ID 0.068 vs 0.085 (-20%, [-0.030, -0.007]); W 25.7 vs 25.7 (no change, [-2.4, +4.2]); recall 0.40 vs 0.43; car F1 0.429 vs 0.474 (-0.045 [-0.102, +0.006]); road IoU 0.795 vs 0.852 (-0.058 [-0.106, -0.012]); late LPIPS gap +0.086 [+0.038, +0.137]. Dev only (settings), not a decision; but warm 0.7 already misses the road and drift gates here.
- Metric check, part 1: key video scored as the render: ID 0 vs 0.099, W 13.2 vs 28.2, recall 1.0, flips 0, car F1 1.0, road 1.0 -> ranks best as required. VAE round trip and fresh noise pending (3753157, 3753158).
- tmet-warm05_dev failed: a key-mask cache .npz read while another metric job was writing it (BadZipFile). Fixed: atomic cache writes (tmp + os.replace) and corrupt entries recomputed; rerun 3753192.
- Lit keyframe keys verified against dense 10 Hz frames: old keys differ by 29-31/255 on scene-0005 / 0007, lit keys by 1.7-2.1 (scene-0026 already matched). train_temporal.py --singles now defaults to data/lat512_tpv_lit (600 train drives incl. all 144 + 6 dev; dev excluded in code).
- Plan's timing step for the trained arms: 101-update runs for A and B (tmpT-A 3753202, tmpT-B 3753203) on A6000 (H200 slices: node damnii10 draining).
- Car map diagnostic resubmitted on three GPU types (3753195-97); first to start wins.

## 2026-11-26 20:15 — temporal: metric check added (plan stop rule, had not been run)
- PLAN-temporal stop rule: before ranking, the key video and the VAE round trip must score best and new-noise-every-frame worst, plus a hand check of ~20 key masks and two turning clips. Added: temporal_rollout.py --method keyvae (the keys' own VAE latents) and --method fresh (new seeded noise every tick); temporal_metrics.py --render_keys (key video scored as the render).
- Submitted (dev drives): troll-keyvae-dev 3753154, troll-fresh-dev 3753155 -> tmet-keys_dev 3753156, tmet-keyvae_dev 3753157, tmet-fresh_dev 3753158. Frozen-method metrics (tmet indep / warm05 / warm07 on dev) running.
- Car map diagnostic 3753126 still queued for an H200 slice.
- Owner asleep until ~06:00; hourly Astra + Fable drift check scheduled (council/drift-*.md).

## 2026-11-26 19:50 — car map: 2k stop rule fired (both arms)
- X1 (boxes) 2k: car F1 0.325 vs E3 0.318 (d +0.008 [-0.005, +0.023]); map-only swap 0.321 (correct - swapped +0.004 [-0.003, +0.010]). Road IoU 0.848 = E3.
- X2 (boxes + grid) 2k: 0.322 (d +0.004 [-0.007, +0.016]); map-only swap 0.319 (+0.002 [-0.012, +0.015]).
- Plan stop rule: "At 2k, map-only swap moves car F1 by < 0.02: stop that arm, back to council" -> both arms stopped; 4k checkpoints exist (training had already finished) but are not scored under the plan.
- Pictures (pulled/cm2k_sheet.jpg): detector boxes land on the right cars, but E3 / X1 right map / X1 wrong map renders are nearly identical. The branch is ignored. Training grad norms stayed tiny (0.003-0.01).
- Diagnostic for the council: scripts/carmap_diag.py (3753126) - eps-MSE with right / swapped / no map, branch change RMS vs prediction RMS, residual RMS per scale, error on map-covered cells, at t = 100..900, for both arms at 2k and 4k.
- Train-drive yardstick: E3 10k on the 30 fixed train drives (tmp/bev_leak_train30.txt, 29 keyed) scored (runs/judge/tpvE3_10000_train30_*.json).

## 2026-11-26 18:40
- P5 timing (3752928): 100 X1 updates fine on an H200 1g.18gb slice, ~1.2 s/update after a 2.2 min first step (the job sat ~20 min in shared-disk reads first); host RSS 13 GB; branch is zero-initialised so "branch off = E3" holds by construction.
- P3 detector leak (runs/judge/bev_leak.json, camera-only boxes, 30 train drives vs 150 unused val drives): car recall 0.848 vs 0.724, precision 0.786 vs 0.723, centre error 0.53 vs 0.62 m -> LEAK FLAG. By distance: recall 0.98/0.88/0.70 vs 0.95/0.79/0.51.
- Per the plan, training maps degraded to val error rates (scripts/bev_degrade.py -> tmp/bev_boxes_trainD.json): drop 3.5/10.2/28.1%, centre jitter sigma 0.16/0.28/0.44 m, phantoms 0/0.05/0.24 per kept box (shifted vehicle copies). Re-check (runs/judge/bev_leak_degraded.json): recall 0.94/0.79/0.51 vs 0.95/0.79/0.51, precision 0.94/0.81/0.49 vs 0.96/0.81/0.48, centre error 0.62 vs 0.62 m, flag off. Note: X2's BEV grid features cannot be degraded the same way; the train-vs-held-out gap gate (<= 0.05) covers it.
- Submitted X1 3752959 (boxes) and X2 3752960 (boxes + grid), both on the degraded boxes, 4k updates.
- Temporal: encode 203/204 done; lit re-render 600/600 -> lit-lat 3752700 released; dev rollouts (troll indep / w05 / w07) running, metrics queued.

## 2026-11-26 17:45
- Progress: temporal encode 165/204 drives; lit keyframe re-render 551/600 scene folders (lit-5 running); camera-only BEVFormer caches: train+dev boxes done (tmp/bev_boxes_train_*.json), 195 grid files; other-val and decision running.
- Car map code: mrv/carmap.py (rasterizer, 4-scale zero-init branch, shared grid sampling, GT oracle frames), scripts/train_carmap.py (X1/X2 on frozen E3), scripts/sd_render.py --carmap_boxes / --carmap_oracle / --map_swap / --carmap_grid.
- P5 timing: 100-update X1 run on an H200 slice (3752928).
- Live page plan picture (SVG decision tree) added by the side chat; ETA labels to be kept current.

## 2026-11-26 18:30 — Car map P2 passed: camera-only BEVFormer
- P2 (decision drives, hull F1 vs game-key cars, score >= 0.3; gate 0.397): standard (can_bus ego motion) 0.417; can_bus zeroed 0.444; zeroed + no previous grid 0.441; GT boxes 0.459. Camera-only mode (can_bus zeroed, previous grid kept) is used for every cache, so the car map needs no non-camera input.
- P1 caches restarted in camera-only mode (old standard-mode outputs moved to tmp/old_std, data/bev_tpv cleared): train shards 3752812-15, decision 3752816, other val 3752817, final 3752818 (held; released one at a time within the 12-job cap).
- Released held temporal jobs as slots freed: t10-enc-6/7, lit-2/3/4 (lit-0/1 finished). lit-5 still held.

## 2026-11-26 17:50 — Job cap: 12 running
- Owner: at most 12 running jobs from this project (other slots for other projects). Cancelled + re-queued on hold t10-enc-6/7 (now 3752770/71) and lit-2..5 (now 3752772-75; they first remove half-done STARTED-without-DONE folders, so they must be released only after lit-0/1 end); held all other pending jobs (bevP1-train x4, bevP1-decision, troll x3, tmet x3, lit-lat). Dependencies retargeted (troll -> new encoders, lit-lat -> new lit jobs). Running now: 6 encoders, 2 lit, 2 P2, P1 other-val, P1 final.

## 2026-11-26 17:30 — Car map preflights started
- bev_extract.py: --zero_canbus, --no_prev (P2), shard split over --scenes_file lists (fixed: the list used to override the shard). P2: decision drives with can_bus zeroed 3752749, zeroed + no previous grid 3752750 (gate 0.397). P1 caches (boxes + grids -> data/bev_tpv): train+dev 4 shards 3752758-61, other val (96 drives, for P3) 3752755, final 24 3752756 (stored, unviewed), decision 3752757.
- scripts/bev_leak.py (P3): predicted vs GT car boxes within 50 m, centre match <= 2 m: recall / precision / centre error by distance, 30 seeded train drives vs 96 unused val drives; flag rule as in the plan.

## 2026-11-26 17:10 — Car map plan signed
- council/PLAN-carmap.md FINAL (v3) signed by Astra and Fable after 3 rounds + sign-off (s3 objections: camera-only motion audit with renderer gates repeated after any motion-source change, X3 = frozen temporal winner + map branch vs the same winner with the branch off, paired predicted-vs-annotated oracle, winner-selection rule, joint final-24 confirmation, cache coverage, X3 placement on non-refresh ticks).
- Plan: preflights P1 cache, P2 can_bus-zeroed hull test (>= 0.397), P3 detector train-leak check, P4 10 Hz feed, P5 timing; X1 boxes (E3 frozen, branch at 4 encoder scales, 4k), X2 boxes + grid, X3 map + memory winner; adopt at car F1 >= 0.382 and >= +0.05 over E3 (lower bound > 0), road not down, swap controls, train-vs-held-out gap <= 0.05.

## 2026-11-26 16:30
- Car map council round 1 (council/carmap-s1-astra.md, carmap-s1-fable.md): both: init E3 10k; BEVFormer detections drawn into the chase view through a zero-initialised pixel-aligned branch; bar car F1 >= ~0.38 (half the 0.763 ceiling) with a confirmed gain. Fable adds a box-length channel (MetaDrive picks car models from length), constant-velocity roll-forward of 2 Hz boxes to 10 Hz with a small tracker, a GT-box arm, and a preflight on BEVFormer's train-drive leak. Round 2 running.
- Temporal: frozen methods on the 6 dev drives queued after the 8 encode shards: independent 3752735, warm 0.5 3752736, warm 0.7 3752737 (E3 10k, seeds 0/1) -> metrics 3752738-40 (cold = independent).

## 2026-11-26 16:10 — Owner: both tracks in parallel
- Car map: council brief council/carmap-brief.md (updating PLAN-bev.md for the third-person target: input form, init E3 vs E1, BEVFormer at 2 Hz vs 10 Hz, interaction with memory, 3 experiments with rules vs the 0.763 ceiling). Astra + Fable round 1 running.
- Temporal (role: pick the memory method; final model = car map + winning memory method with a short retrain): temporal_encode fixed per amendment 2 (offset-rejected ticks encoded from their nearest bundle; hard flag) and restarted on 8 shards (3752726-33; the 2-shard run did 6 drives in 30 min). Lit keyframe re-render (3752694-99) + latents (3752700) queued.
- Written: scripts/temporal_rollout.py (independent / warm / history; one sampling-noise tensor per drive and seed; separate history stream; through offset rejects; restart at hard breaks; --swap lag test; --shard) and scripts/temporal_metrics.py (placement every 5th accepted tick; key tracks; edge-car ID and W per transition for shared-transition comparisons; recall, flips, ghosts; late / early LPIPS gap vs the cold rollout; lag; --compare = drive-level paired bootstrap with seeds together). Lag threshold still uses a fixed 0.3 F1 reference (TODO: the other drive's cold F1).

## 2026-11-26 15:35 — Placement diagnostic (owner review item 4)
- BEVFormer-base (camera-only, train-only checkpoint) run on the 30 decision drives (1,191 frames, boxes only; tmp/bev_boxes_decision.json). scripts/bev_box_judge.py: car boxes projected as hulls into the third-person view, matched to the cached game-key car masks (Mask2Former, IoU >= 0.3), 120 judge frames (seed 0):
  GT nuScenes car boxes 0.459 | BEVFormer boxes (score >= 0.3) 0.417 | our renderer E3 10k 0.318 (its own ceiling, true keys through the VAE: 0.763).
- Overlays (pulled/bevov_sheet.jpg -> docs/figs/bevbox_diag.jpg): GT (yellow) and BEVFormer (cyan) hulls sit on the game cars; the hull method's ceiling is low (0.46) because box hulls are fatter than car silhouettes, so compare BEVFormer with GT boxes: 91% of GT, vs our renderer at 42% of its ceiling.
- Reading: the cameras do contain where the cars are (a camera-only detector recovers it almost fully); our renderer loses much of it. Spatial conditioning from BEVFormer (council/PLAN-bev.md, written for the old drone keys) is motivated; it needs a council update for the third-person target before any build.

## 2026-11-26 15:20 — Temporal amendments 1-2 approved; data prep resumes
- Pairing: at 45 ms only 54.1% of ticks were usable (max-over-cameras offset median 44.5 ms: half a camera period); 50 ms -> 94.5%. Amendment 1 (50 ms) approved by Astra and Fable. Even so no drive had an unbroken >= 19 s run (replays end ~18.4 s; ~5% rejects scattered). Amendment 2 approved: rollouts run through offset rejects (rejected ticks unscored), hard break at repeated bundle / render gap / > 100 ms, drift window ticks 150-169 on >= 170-tick rollouts, cold frames from the same bundles. Measured qualifying rollouts: decision 22/30, final 18/24, dev 5/6, train 85/144.
- Owner agreed: no 40k long run now; temporal arms from E3 10k + BEVFormer placement diagnostic (bev-boxes 3752688 running on the 30 decision drives; seaborn added to envs/bev; scripts/bev_box_judge.py scores GT / BEVFormer / renderer cars vs the game keys).
- I/O better (8 MB fsync 1.5-5.6 s on the slow OSTs at 15:15). Manifest at 50 ms -> temporal_encode (2 shards) queued.

## 2026-11-26 13:55
- Dense 10 Hz render finished fast (~25 min): 204 drive folders in data/sim/keys_tpv10 (150 train/dev + 54 val). Baseline (no warm start) clips re-rendered: tmp/xwarm0_scene-0563/0915.
- I/O degraded again at ~13:50 (8 MB fsync: OST4 289 s, others 12-29 s); VPN dropped ~13:50, reconnected via FortiTray. Pairing manifest submitted as a batch job (3752595). Held until writes recover: temporal_encode (~14 GB of token caches) and the --lit keyframe re-render (~12k PNGs).
- Waiting for the owner: 40k long run (go rule not met: 42% of ceiling, plan silent on 25-50%) vs proposed temporal arms from E3 10k + BEVFormer box diagnostic.

## 2026-11-26 13:40 — Experiment T verdict at 10k
- Judge vs exact game keys, 30 held-out drives (Mask2Former / SegFormer car F1; road IoU; LPIPS): E1 10k 0.288 / 0.279, 0.846, 0.312; E3 10k 0.318 / 0.302, 0.848, 0.316. E1 - E3 = -0.030 [-0.077, +0.013] / -0.024 [-0.060, +0.014]; road -0.002 [-0.014, +0.011]. (5k: E1 0.276, E3 0.294; E2 0.044.)
- Decision rules (council/PLAN-tpv.md): init -> our trained renderer (E2 drew almost no cars: outright). Data -> E1 - E3 < 0.02: the ~2.4k-frame quarter is already enough at this budget; no more drives needed. Go rule -> best arm 0.318 = 42% of the VAE ceiling 0.763: below the 50% needed for the 40k long run, above the 25% "back to council" line; the plan has no rule for this band -> owner decision. Road IoU rising (E3 0.836 -> 0.848).
- Placement climbs slowly (E3 +0.024 from 5k to 10k). Starting point for the temporal arms: E3 10k (best car score; its training drives are the temporal train list).
- E1 5k train-frame judge (3752154) produced no score file (likely lost in the I/O stall); E3 5k train 0.359 vs val 0.294.

## 2026-11-26 13:25 — Dense 10 Hz render started (reduced width)
- Walking-RL session reports storage recovered at 13:11; my per-OST probe (8 MB fsync) at ~13:20: OST3 0.5 s, OST2 3.5 s, OST0/1/4/5 11-22 s (better, not normal). Started the dense render with 6 jobs: train+dev 150 drives (tmp/t10_train.txt, scn_8k) 3752564-67, decision+final 54 val drives (tmp/t10_val.txt, scn_val) 3752568-69 -> data/sim/keys_tpv10. Lit re-render of keyframe keys after this.

## 2026-11-26 13:15 — Lighting mismatch explained
- Keyframe render drawn twice (keep the second) matches the dense frame (mean abs diff 1.7 vs 31 for the old keyframe key; pulled/pilot/light_cmp.jpg). MetaDrive's first render after several unrendered ticks has no shadows / flat lighting. The dense look (shadows) is the correct render; the 12,111 existing TPV keys have the flat look. E1/E2/E3 results stay valid (one look throughout).
- For the temporal arms (dense rows + keyframe singles in one batch) the singles must have the same look: re-render all TPV keyframe keys with --lit (new flag: render twice) into data/sim/keys_tpv_lit, with the dense render, once the slow OSTs recover. slurm/sim_tpv.sh takes a keys folder name and extra args.

## 2026-11-26 12:55 — E1/E3 finished 10k; ceiling and controls; warm start first look; lighting mismatch
- E1, E3 finished 10k (12:34 / 12:38); 10k judges running (3751991 / 3751995).
- VAE ceiling (true game keys through the SD VAE, same judge): car F1 0.763 (Mask2Former) / 0.624 (SegFormer), road IoU 0.94, LPIPS 0.040. E1 5k = 0.276 = 36% of ceiling (go rule needs >= 50% at 10k).
- Wrong-street control on the TPV target (E1 5k, swap_scene): car F1 0.019 vs 0.276 -> the judge validates for the new target.
- Per-OST write probe (8 MB fsync) at 12:25: OST1 2 s, OST3 1 s, OST2 22 s, OST0/4/5 timeouts (71-353 s). Big dense render stays held.
- Dense pilot (scene-0005/0007, 162/182 ticks rendered, both ended before the last keyframe): object positions match the keyframe keys, but the LIGHTING differs (dense frames have car shadows and brighter grass; keyframe keys rendered every 5 ticks have none; mean abs diff ~30/255). Likely the shadow pass needs consecutive frames. Diagnostic: keyframe render with a double perceive (MRV_DOUBLE_RENDER, 3752466). Mixing two looks under one style would be invalid; must be resolved before the temporal arms.
- Warm start first look (E1 2.5k, scene-0563, crude mean consecutive-frame change: none 5.95, warm 0.5 3.24, warm 0.7 2.25): warm 0.5 COLLAPSES into a stripy black-and-white feedback artifact (its low change = a frozen broken picture); warm 0.7 looks as clean as independent frames and steadier (pulled/warm_strip.jpg). Baseline clips re-rendered (3752480/81; the side chat's xvid runs were cut short).
- scripts/train_temporal.py written (arms A/B/C per the signed plan).

## 2026-11-26 12:00 — E1 vs E3 at 5k: tie
- E1 5k (all drives) car F1 0.276 (Mask2Former) vs E3 5k (quarter) 0.294: E1 - E3 = -0.018 [-0.066, +0.021]; road IoU 0.845 vs 0.836 (+0.009 [-0.001, +0.020]); LPIPS 0.335 vs 0.326. Within the data rule's "< 0.02 -> enough" band so far; decision at 10k. E1 swap control and VAE ceiling still running (I/O).
- I/O still patchy: 50 MB fsync timed out (217 s) at ~11:45 after 1.8 s at 11:38 (slow OSTs, as the walking-RL session reported). Big dense render stays held.
- scripts/temporal_encode.py written (tokens + key latents per dense tick).

## 2026-11-26 11:45
- Cluster writes partly recovered (our 10 MB fsync <1 s, 50 MB 1.8 s at 11:38); the walking-RL session still sees slow OSTs (OST1/2/4/5) and slow reads, and asked to hold the big dense render ~1 h. Released small held jobs (xwarm 3752205-08 running; tj-E1/E3-10000 waiting for training). Before the big render: per-OST write probe (lfs setstripe -i).
- Temporal lists frozen: data/temporal_lists.json (train 144, dev 6 = scene-0468/0393/0396/0234/0285/0043 over density x turning, decision 30, final 24). Dense render pilot on scene-0005/0007: 3752363 (keys_tpv10).

## 2026-11-26 11:25 — First 5k scores (E1 still scoring); temporal plan signed
- Judge vs exact game keys, 30 held-out drives x 4 frames x 2 seeds (Mask2Former / SegFormer car F1, road IoU, LPIPS):
  E2 5k 0.044 / 0.041, road 0.74, LPIPS 0.525 (confirms stopping E2); E3 5k 0.294 / 0.253, road 0.84, LPIPS 0.326;
  E3 5k on its own training drives 0.359 / 0.380, road 0.87, LPIPS 0.247 (train-val gap ~0.07 F1). E1 5k, swap control, VAE ceiling still running (slow I/O).
- Cluster writes still stalled (50 MB fsync timed out after 355 s at 11:17); holds kept.
- PLAN-temporal.md FINAL (v5) signed by Astra and Fable after 5 rounds (s3-s5 objections: abs per-transition ID, non-negative placement, shared noise tensors, no DiT drift comparison, shared-transition ID/W, paired recall lower bound, A+warm-start comparator, confirm rule on 24 untouched drives).
- Built: sim_render.py --dense (every 10 Hz tick, t<tick>.jpg + ticks.json), temporal_lists.py (frozen train 144 / dev 6 / decision 30 / final 24), temporal_manifest.py (sweep pairing <= 45 ms, sequence breaks).

## 2026-11-26 11:05 — Cluster I/O stall (Lustre home writes) — load shed
- Since ~10:23 writes to /home stall (another project's 50 MB fsync > 10 s; reads fine). On request: cancelled tpvE2 3751988 (plan already leaned to stopping E2; its 5k checkpoint and 5k scoring 3752151 are kept) and its 10k judge 3751993; held pending xwarm 3752205-08 and tj-E1/E3-10000 3751991/95 (release with scontrol release when writes recover). E1/E3 keep training (one 515 MB ckpt each ~15 min). f28-p3g is not ours.

## 2026-11-26 10:55 — Next experiment: temporal consistency (owner request)
- Owner saw a 10 s clip (real sweeps ~12 Hz through E1 ckpt_2500, fixed noise; side chat, tmp/xvid_*): mostly stable, but edge cars change model/shape between frames, and no memory of objects that leave view. Temporal consistency / memory is the next experiment once E1/E2/E3 pick the starting point.
- Council (no improvising): brief council/temporal-brief.md (options: inference warm start, Tesla-style previous-frame conditioning with self-rollout training, temporal input memory incl. BEVFormer, video/temporal attention modules; metrics: flicker/stability, 20 s drift, placement must not drop). Astra (gpt-6.2-astra, xhigh) and Fable (claude-fable-5-2, high) round 1 running via delegate_task.
- Free baseline now: scripts/video_render.py (in repo) gets a warm-start option (previous output latent re-noised so a fraction s of the 25 DDIM steps remains, same fixed noise). Runs on the same two clips at s = 0.5 and 0.7: 3752205-08 -> tmp/xwarm<s>_<scene>.

## 2026-11-26 10:40 — 5k batch submitted
- All three arms stalled ~12 min at step 4,800 (10:17-10:29, same node saxa; shared filesystem / node slowness, ssh also slow), then continued; no loss.
- 5k batch (after ckpt_5000 + step >= 5,100): tj-E1-5000 (+ VAE ceiling) 3752150, tj-E2-5000 3752151, tj-E3-5000 3752152, tj-E1-5000-swap 3752153, tj-E1/E3-5000-train 3752154/55, preview 5000 3752156.
- Live page: https://taruntomar122.github.io/mini-reality-view/ (docs/index.html via scripts/build_site.py; scripts/update_site.sh pulls logs/scores/previews, rebuilds, pushes).

## 2026-11-26 09:55 — Experiment T preview at 2.5k (pictures only)
- pulled/prev2500_scene-{0915,0563,0632}.jpg (6 real photos | game key | E1 | E2 | E3 | E1 wrong street), same noise per column.
- E1 and E3 already match the game keys closely after 2.5k steps: road shape, crossings, rows of parked cars, trucks ahead (moving-traffic drive: both trucks and the near white car in place), the red barrier on the quiet street. E3 (quarter data) looks about as good as E1.
- E2 (plain SD start) has the game look but invents cones / barriers and puts cars in the wrong places.
- Wrong-street control (E1 with another drive's cameras): clearly different layouts and traffic, so the cars come from the cameras.
- Small glitch: own car slightly mangled in one E1 frame (quiet street, last row).

## 2026-11-26 09:20 — Owner review folded in (phone work deferred)
- Owner: phone demo is low priority; focus on how accurately six real cameras place traffic in the game-style chase view. scripts/try_my_photos.py and phone_test.py still hardcode style 1 (TPV is style 2) and a single rotating phone has different camera origins than the car rig: fix and test both paths on one matched val example when phone work resumes.
- Earlier visual feedback: the queued 5k judges waited for the whole 10k job -> cancelled (3751990/92/94); they are submitted by hand once ckpt_5000 is written. Preview job (slurm/tpv_preview.sh <step>): fixed held-out drives scene-0915 (busy junction, rain, 26 vehicles), scene-0563 (moving traffic at a light, 10/20 moving), scene-0632 (quiet industrial street) x 3 keyed frames x seed 0 (same noise for every checkpoint), E1/E2/E3 + E1 with another drive's cameras (wrong street). Real camera frames and keys copied to tmp/prev_cams.
- judge_pair.py: key file must match frame index AND sample token (missing count reported); key masks cached per judge in runs/judge/keycache_<judge> (240 renders use only 120 distinct key frames).
- sd_render.py: --scenes, --split train|val, --scene_list (for the train-frame vs held-out check).
- Still to do: judge validation on game keys (hand-count ~20 varied keys with detections overlaid; swap_scene control with matched noise + paired CIs on the TPV target); fixed train-frame vs held-out comparison (E3's quarter is in both E1's and E3's training data); BEVFormer box diagnostic only if appearance is good but moving cars are misplaced.
- Note: E3 at 10k x 16 sees ~68 passes over 2,363 frames vs ~17 for E1 over 9,587: it tests how many NEW game-view examples the warm start needs at a fixed update budget. A near tie with a wide CI leaves data sufficiency open.

## 2026-11-26 08:55 — Experiment T arms training
- Keys done: 12,111 third-person frames (9,587 train / 2,524 val) in data/lat512_tpv. E1 (init sd_run1, style rows 2->3), E2 (plain SD), E3 (init sd_run1, quarter: 2,363 train frames) all running since ~08:42; judges queued at 5k / 10k.

## 2026-11-26 07:40
- Third-person keys: camera now shows the whole own car (pulled/tpv_first.png). Shards 0/1/2/7 + both val shards done: 5,338 train + 2,524 val keys. Shards 3-6 landed on damnii08, which (like damnii07) hangs MetaDrive on every scene: cancelled and resubmitted excluding both nodes (3752033-36); latents 3751984 retargeted to them; E1-E3 still queued behind the latents.
- README rewritten short with a "Where we are" table; the long write-up moved to docs/results.md.

## 2026-11-26 07:50 — Experiment T plan + arms queued
- Fable round 1 (council/tpv-s1-fable.md) -> merged plan council/PLAN-tpv.md: E1 init sd_run1, E2 plain SD, E3 E1 on a quarter of the drives; 10k steps each; judge vs exact game keys (scripts/judge_pair.py: same segmenter on render and key, own car masked, car F1 / road IoU / L1 / LPIPS), VAE ceiling; decision rules as in the plan.
- Code: train_sd.py --init expands style rows (2 -> 3) and --scene_list; sd_render.py --keys_like (val drives / frames that have game keys); judge_pair.py; slurm/tpv_judge.sh, slurm/tpv_experiments.sh.
- Queued after the latents (3751984): E1 3751987, E2 3751988, E3 3751989; judges at 5k/10k 3751990-95 (E1 10k also runs the VAE ceiling).

## 2026-11-26 07:30 — Experiment T (third-person game keys) started
- Owner: research first, then 3 experiments, start now. Council brief council/tpv-brief.md (init A/B/C, data size, eval, 3 experiments); Fable running (Astra unavailable in this session).
- Camera moved back so the whole own car is in frame: tpv = 7.5 m behind the car centre, 2.4 m up, pitch -8 deg (old preview keys deleted).
- Data jobs: 8 train shards over data/scn_8k (keys only, 2080 Ti, damnii07 excluded) 3751973-80; val: scn_convert_subset (150 val scenes -> data/scn_val) 3751981 -> 2 shards 3751982/83; latents data/lat512_tpv 3751984 (afterany). Keys -> data/sim/keys_tpv (train + val in one folder; train_sd splits by the official val list). Inputs = real tokens data/enc_full (+ phone copies data/enc_aug_0 for train).

## 2026-11-26 07:00 — Owner's new direction: real photos in, game-style third-person view out
- Owner: outputs still messy because the model copies messy keys -> idea: keep real camera photos as input but use the perfect game render as the answer key; and a third-person chase view (own car visible) instead of the drone.
- sim_render.py: --view tpv (6 m behind the car centre, 2 m up, pitch -6 deg, own car shown), --keys_name, --keys_only, --only. Preview on 4 train scenes (tpv-preview 3751968, 2080 Ti, excluding damnii07): 51 keys in data/sim/keys_tpv. Sheets pulled/tpv_scene-*.jpg (6 real photos | third-person game key | old drone game key).
- Possible tweak: the own car is cut off at the bottom edge -> move the camera ~1 m further back / a bit higher.
- Waiting for the owner's go before rendering ~8k third-person keys.

## 2026-11-26 06:15 — PAUSED by the owner
- Owner: "we stop here, pause all the jobs and let's chat". Cancelled all of this project's jobs (bev-try 3751843 running; sdB-sim 3751715, sdB-real 3751716, sd-run1-45k 3751718, phone-test 3751879 + judges 3751880/81 pending). Other projects' jobs untouched.
- State kept on the cluster: runs/sd_run1/ckpt.pt (final 40k), runs/sd_run1_45k (copy, untrained), sim pilot arms (runs/simA|simB|simC), judge renders/scores (runs/judge), sim data (data/sim, 7,891 pairs; enc_sim / lat512_sim), BEV env (envs/bev, built) + infos pkl; phone_test.py ready, not run.
- To resume: phone test (scripts/phone_test.py + judges), BEV trial (slurm/bev_extract.sh --split val --limit_scenes 2), the scaled sim run only if still wanted (slurm/sim8k_chain.sh steps sdB-sim / sdB-real / sd-run1-45k).

## 2026-11-26 06:00 — SD 40k placement
- Judge, SD 40k vs 20k (240 renders): Mask2Former F1 0.286 vs 0.284 (+0.002 [-0.044, +0.043]); SegFormer 0.162 vs 0.153 (+0.009 [-0.038, +0.060]); kerb error 1.46 vs 1.40 m (+0.058 [+0.019, +0.098]) / 1.42 vs 1.39 (+0.030 [+0.004, +0.057]). Placement plateaued after 20k (kerb slightly worse); 10k -> 20k -> 40k car F1: 0.260 -> 0.284 -> 0.286.
- Queue: our account is at its job limit (AssocMaxJobsLimit; 20 running jobs on the account incl. other projects): phone-test, sdB-sim, sd-run1-45k waiting. sim8k encoders done (508 scenes); sim8k-lat finished?

## 2026-11-26 05:20
- scripts/phone_test.py: the 120 judge frames (seed 0) through the phone path: real camera pictures -> mrv/aug.augment (crop/zoom, colour, gamma, hue, blur, JPEG) -> try_my_photos' 16:9 crop -> DINOv2 -> DEFAULT rig rays -> SD 40k (style 1, DDIM 25, CFG 2, same noise as sd_render seed 0). Queued after judge-sd40k: 3751879 -> m2f 3751880 / SegFormer 3751881; compare with the calibrated clean renders (paired on seed 0).

## 2026-11-26 05:10 — sd_run1 finished (40k)
- Final built-in eval, 8 val scenes vs trusted splat keys (style 1): masked L1 27.32 / LPIPS 0.307 (5k preview on 4 scenes: 33.5 / 0.390). Pictures (pulled/sd40k_final.jpg -> docs/figs): clearly better than the 45M DiT on the same scenes; near field behind the ego still smeared. README updated.
- Queue is busy: judge-sd40k 3751798, sd45-copy 3751717, sim8k encoders 3751712-14, bev-try 3751843 all waiting for GPUs.

## 2026-11-26 04:40
- BEV trial failed in detectron2 v0.6 import: Image.LINEAR (removed in Pillow 10) -> pillow 9.5.0 in envs/bev. Trial rerun 3751843 (A6000).
- sd_run1 39.9k (final render next). Sim encoders waiting for GPUs.

## 2026-11-26 04:10
- Sim 8k: shards 6/7 both ran on damnii07 and finished zero scenes in 1.5 h (MetaDrive hangs on every scene there; the pilot's hanging shard was also on damnii07). Cancelled; 7,891 drone pairs in data/sim/keys (pilot + 6 shards), close to the ~8,000 target. Encoders (afterany) start now.
- BEV trial 3751639 moved to an A6000 (2080 Tis busy). sd_run1 38.5k.

## 2026-11-26 03:45
- BEV infos written (nuscenes_infos_temporal_train.pkl 506 MB, val 105 MB, 03:34); create_data then started the 2D-annotation export (not needed, ~8.5 h) -> prep cancelled, trial 3751639 released.
- Sim 8k renders: 6/8 shards done; shards 6/7 keep hitting MetaDrive hangs (attempt 8 each, watchdog working).
- sd_run1 36.9k -> judge at 40k queued (3751798, sim_judge.sh sd40k).

## 2026-11-26 02:45 — SIM VERDICT
- Arm C (real + 2 sim + 2 twin rows) vs A: C5k F1 0.000 vs 0.033, -0.033 [-0.052, -0.015] (m2f) / -0.029 [-0.057, -0.004] (SegFormer); C3k = A3k. Pictures (pulled/sim_arms.jpg): C draws realistic streets with almost no cars. C FAILS.
- B passes the go rule at equal real steps (see 02:10) but not at equal updates (B5k vs A7k -0.026 / -0.015, CI incl. 0; m2f kerb +0.068 worse).
- Decision rule (owner, 25 Nov): B passes, C does not -> render ~8,000 pairs and a new main run (5k sim, then real with sd_run1's settings), replacing sd_run1 only if it beats it at equal real steps. Followed, plus the plan's equal-update reference (sd_run1 continued to 45k). Flag for the owner: the equal-update result suggests a head start only; the scaled run (~17 h on an H200 slice) can still be cancelled before it starts.
- Launched (slurm/sim8k_chain.sh): scn-8k convert 600 train scenes 3751703 -> 8 render shards 3751704-11 (2080 Ti, watchdog, 11 h attempts) -> sim-enc 3751712/13 + lat 3751714 -> sdB-sim 5k 3751715 -> sdB-real 40k 3751716 (--init sdB_sim, n_styles 3, cover_min, phone tokens; snapshots 10/20/30/40k). sd_run1 -> copy -> 45k: 3751717/18.
- BEV prep running (create_data at ~1 sample/s for 34k samples -> ~9 h; 8 h limit risk, watch).
- README: sim result (learned) + C (did not work) + figure.

## 2026-11-26 02:10 — Sim pilot: arm B scored
- VPN dropped ~01:50, reconnected with FortiTray (no job lost).
- BEV prep failed again: Python prefers a regular package over a namespace one anywhere on sys.path, and detectron2/tools has __init__.py while BEVFormer/tools has none. Added empty BEVFormer/tools/__init__.py (projects/ already had one); import check OK. Prep resubmitted 3751693 (trial 3751639 after it).
- Arm B (2k sim-only, then real) vs A, paired scene bootstrap (240 renders, 30 val scenes), Mask2Former / SegFormer:
  - B5k vs A5k (equal real steps): F1 0.096 vs 0.033, +0.063 [+0.021, +0.098] / 0.076 vs 0.033, +0.043 [+0.004, +0.084]; kerb -0.003 [-0.048, +0.040] / -0.084 [-0.112, -0.054].
  - B3k vs A3k: F1 +0.027 [+0.006, +0.052] / +0.020 [-0.001, +0.042]; kerb -0.040 [-0.082, +0.002] / -0.037 [-0.061, -0.010].
  - B5k vs A7k (equal total updates): F1 -0.026 [-0.067, +0.013] / -0.015 [-0.051, +0.018]; kerb +0.068 [+0.006, +0.129] / -0.036 [-0.073, -0.001].
  - Reading: at equal real steps B passes the go rule as written (Mask2Former +0.063 >= 0.05, CI excludes 0, SegFormer agrees on sign, no kerb regression). At equal total updates B is no better than 2k more real steps: the sim phase gives a head start, not better placement. C scoring running.

## 2026-11-26 01:30
- Arm A 7k: F1 0.123 (m2f) / 0.092 (SegFormer), kerb 1.48 / 1.43 m -> cars appear between 5k and 7k.
- BEV prep failed: detectron2's editable install exposes its own `tools` package first on sys.path, so BEVFormer's tools.data_converter was not found. bev_prep.sh now runs create_data via runpy with BEVFormer's root first. Resubmitted 3751669; trial 3751639 retargeted.
- B2 / C still training (~4k/5k).

## 2026-11-26 01:10
- BEV env built (fix 2 worked): BEV_BUILD_OK torch 1.10.2+cu113, CUDA on 2080 Ti, mmdet3d 0.17.1, detectron2 v0.6.
- BEV data prep 3751638 (create_data -> nuscenes_infos_temporal_{train,val}.pkl; symlinks bev/BEVFormer/data/{nuscenes,can_bus}). scripts/bev_extract.py: runs BEVFormer recurrently per scene in time order, takes the BEV embedding after each frame, resamples it onto our ego grid (x -8..41 m, y +-16.4 m, 96x64 bilinear -> 2x2 pool -> 48x32 x 256 fp16) via lidar2ego; --eval runs the official nuScenes detection metrics (= sanity gate). Trial on 2 val scenes queued (3751639).
- Sim pilot, first scores: arm A 3k F1 0.006 / 0.004 (m2f / SegFormer), 5k 0.033 / 0.033; kerb 1.61 / 1.45 and 1.55 / 1.48 m. At 5k the pilot renderer draws road layout and barriers but almost no cars (pulled/simA5k_check.jpg) - the main run only placed cars at 10k (0.26). The pilot therefore measures whether sim makes cars appear sooner; go rule unchanged.
- sd_run1 ~27k.

## 2026-11-26 00:35
- BEV build (fix 2 of 3): mmdet3d 0.17.1 now compiles and imports (sysroot pin worked); detectron2 master failed (needs ATen/cuda/Atomic.cuh, torch >= 1.11). Pinned detectron2 v0.6 (torch 1.10 era); bev_build.sh skips mmdet3d when importable. Rebuild 3751589.
- Sim pilot scoring queued (slurm/sim_judge.sh: render judge set + Mask2Former + SegFormer per snapshot): A 3k/5k (now) 3751590/91, A 7k 3751592 (after A), B 3k/5k 3751593/94 (after B2), C 3k/5k 3751595/96 (after C). Comparisons: B5k and C5k vs A5k (primary), B5k vs A7k (equal updates), 3k as a trend check.
- Arms: A 6.3k/7k, B2 1.6k/5k, C 1.6k/5k (~37 steps/min, ends ~02:00). sd_run1 25.7k.

## 2026-11-25 23:55
- BEV build 3751357 failed (fix 1 of 3 for this stage): mmdet3d spconv CUDA files do not compile: nvcc 11.3 cannot parse `__s128` in the conda sysroot's Linux 6.12 kernel headers (sysroot_linux-64 2.39 came with gcc 10). Fix: pin sysroot_linux-64=2.17 in envs/bev (login node, logs/bev_sysroot.log), then rebuild with a clean build dir on a 2080 Ti.
- Arms: A 3.9k/7k, B1 done -> B2 1.9k/5k, C running. sd_run1 23.4k.

## 2026-11-25 23:25 — SD 20k judge
- SD 20k vs 10k (judge set, 240 renders): Mask2Former F1 0.284 vs 0.260 (diff +0.023, 95% CI -0.026..+0.075); SegFormer 0.153 vs 0.124 (+0.030, CI -0.002..+0.063); kerb error 1.40 vs 1.47 m (-0.064, CI -0.103..-0.027) / 1.39 vs 1.43 (-0.036, CI -0.064..-0.005).
- Pictures (pulled/sd10k_vs_20k.jpg -> docs/figs): fewer dark speckles on roads, sharper buildings, more drawn cars inside the true outlines; the near field right behind the ego is still smeared (weakest part of the keys). README: placement judge table + this figure.
- Arms: A 2.2k/7k, B1 1.3k/2k, C running (15 min), B2 queued. BEV build still waiting for an A6000.

## 2026-11-25 23:10
- Sim data final for the pilot: 91 scenes, 1,388 clean pairs (2,776 rows with the phone-style copies); shard 3 used up its 6 attempts on hanging scenes, the rest never started. Plan asked for ~2,000 pairs: we have ~70% (stated in the write-up).
- Arms: A 3751356 at 700/7k; B1 3751311 running (sim-only 2k); B2 3751312 and C 3751313 queued.
- sd_run1 20.2k: checkpoint copied to runs/judge_ckpt_sd20k.pt -> judge renders 3751479 -> m2f 3751480 + SegFormer 3751481 (compare with 10k: m2f F1 0.260, SegFormer 0.124).
- BEV build 3751357 still waiting for an A6000.

## 2026-11-25 22:30
- Sim watchdog works (shard 3: two scenes timed out at 5 min, skipped). 90 scene folders. Encoders 3751308-10 switched to afterany on the shards (the running shard 3 wrapper still has only 6 attempts; if it runs out, encode what is there rather than block the arms).
- Arm A 3751356 loading (25 min in). BEV build 3751357 queued for an A6000. sd_run1 18.4k.

## 2026-11-25 22:00
- Arm A (3751138) was OOM-killed while loading (MaxRSS = 120 GB limit; sd_run1 had peaked at the same limit and survived by luck). train_sd.py loading rewritten: load_set splits train / val scenes as it reads and builds each tensor by copying chunks into a preallocated buffer (freeing each chunk), so token rows are never held twice. Tested on a tiny set (3 lidar scenes incl. one val + 2 splat scenes, 30 steps + render): OK. Arm A resubmitted: 3751356.
- Sim shard 3 keeps hitting hanging scenes (scene-0378 hung 25 min). sim_render.py: per-scene C-level watchdog (faulthandler.dump_traceback_later(300, exit=True)) -> exit after 5 min, wrapper restarts, scene skipped; wrapper now allows 20 attempts (applies to newly submitted shards; the running one picks up the watchdog at its next attempt). 85/100 scene folders.
- BEV env login part done (BEV_LOGIN_OK, torch 1.10.2+cu113, mmcv-full 1.4.0, checkpoint 277 MB, can_bus 7,832 files). Build on A6000: 3751357.
- sd_run1 ~17k.

## 2026-11-25 21:20 — Sim chain repaired
- The resubmitted render shards (3751244/45) had no GPU and landed on saxa: MetaDrive could not open /dev/dri (EGL permission denied) and crashed at env.reset on every attempt. Each crash left an empty scene folder with a STARTED marker (12 of them; removed), and the wrapper ended with exit 0, so sim-enc / sim-lat ran on the 64 finished scenes and arms A/B1/C started early.
- Fixes: STARTED is written only after env.reset succeeds; sim_render.sh exits with the last attempt's code; shards run on a 2080 Ti (the first run had --gres=gpu:1).
- Kept arm A (3751138, real data only, unaffected). Cancelled B1/B2/C (minutes in, folders removed). New chain: render 3751306/07 (2080 Ti) -> sim-enc 3751308/09 + sim-lat 3751310 (incremental, skip finished scenes) -> B1 3751311 -> B2 3751312; C 3751313.
- sd_run1 14.4k. BEV env still downloading pip packages (slow network on the login node).

## 2026-11-25 20:45
- Sim render shards 2 and 3 hung since ~18:52 (MetaDrive stuck inside env.step on scene-1046 / scene-1082, one pair each; logs empty because grep was block-buffered). 64/100 scenes. Fix: sim_render.py resumes (DONE / STARTED markers; marker-less folders from the first run count as done) + --skip; slurm/sim_render.sh retries each shard up to 6x with a 25 min timeout per attempt (a hung scene keeps STARTED and is skipped next time), grep --line-buffered. Encoders skip empty scenes / frames without all 6 cameras. Resubmitted shards 2/3 as 3751244/45; sim-enc-0/1 and sim-lat retargeted to them (scontrol), arms follow as before.
- BEV env: pip step (mmdet/mmseg/numba/nuscenes-devkit...) sat 40 min at 0.6% CPU (network stall) -> killed, rerun with --timeout 60 --retries 5, not quiet (logs/bev_setup_login2.log).
- sd_run1 12.5k (~48 steps/min).

## 2026-11-25 20:10 — Judge validation result
- Controls on SD-10k renders (240 renders each): car F1 normal / tokens rolled (rays kept) / wrong street:
  Mask2Former 0.260 / 0.026 / 0.028; SegFormer 0.124 / 0.016 / 0.012. Kerb error 1.47 / 1.53 / 1.58 m (m2f), 1.43 / 1.55 / 1.52 m (SegFormer).
- Verdict (written before the sim results): car F1 is valid for whether cars come out in the right places (10x drop with wrong inputs, both judges), with coarse resolution (-0.04 at a 1 m label shift, -0.09 at 2 m; the pre-set 1 m criterion of 0.15 fails). Kerb error barely moves (+4-10% with the wrong street) -> treat as weak; a sim "go" must come from car F1 (rule unchanged: >= 0.05 F1, CI excludes zero, both judges agree), and the write-up says placement differences under ~1-2 m are below the judge's resolution.
- Picture: pulled/judge_controls.jpg (docs/figs/judge_controls.jpg): yellow = projected nuScenes cars; with the right cameras, drawn cars sit in most outlines.
- sd_run1 resumed at 10000 (STOP removed in time), 11k at 19 min (~53 steps/min -> 40k in ~9 h); RSS 76 GB. Sim renders 60/100 (2 of 4 shards done). BEV env: conda part done, pip installs running.

## 2026-11-25 19:40
- The 19:02 OOM happened during the STOP save (ckpt.pt.tmp half written at 19:02): memory near the limit + save spike. The STOP file was still there when the restart started -> removed (with the partial .tmp) before the restart reached its first step; otherwise it would have paused at once. sd_run1 3751192: cover_min keeps 0.639 (lidar) / 0.658 (splat) of latent cells; 56,260 + 4,606 train rows; peak RSS 126 GB while loading, 78 GB after the duplicate rows are freed.
- Judge on the lidar answer keys (120 judge frames): Mask2Former F1 0.02, SegFormer 0.00. The segmenters do not see cars in dotted lidar keys, so keys cannot serve as the ceiling / resolution check. Remaining validation = roll_tokens + swap_scene controls (renders 3751193/96 moved to any GPU).
- Sim renders 58/100 scenes. (Note: timestamps of the last few entries above ran ~45 min ahead of the clock.)

## 2026-11-25 20:10 — Experiment W plan converged; judge ceiling check
- Council W: Fable round 1 (council/bev-s1-fable.md) + my second view -> council/PLAN-bev.md. BEVFormer-base BEV grid (train-only checkpoint) as an extra input next to DINOv2, cached offline (48x32 x 256 in ego metres); first arm W2 = pixel-aligned warp of the grid into the drone latent (ground + 1 m height samples) added after conv_in; phone rows get a BEV null; 15% dropout; control = sim arm A; same go rule. Fallbacks StreamPETR, SparseBEV; later candidate Depth Anything 3 (no leakage, phone-friendly).
- Env: slurm/bev_setup_login.sh running on the login node (conda py3.8, torch 1.10.2+cu113, mmcv-full 1.4.0, CUDA 11.3 + gcc 10, repos, checkpoint, can_bus) -> then slurm/bev_build.sh on an A6000 (mmdet3d 0.17.1 + detectron2 ops).
- Judge ceiling/resolution: scripts/judge_keys.py turns the lidar answer keys of the 120 judge frames into a renders folder (runs/judge/keys_lidar); scored by m2f + SegFormer at shifts 0/1/2 m (3751199-204).

## 2026-11-25 19:55 — Judge validation round 1; sd_run1 OOM
- sd_run1 (3751069) was OOM-killed at 19:02 (MaxRSS 126 GB of 120 GB, ~13.6k steps) before the STOP: steps 10k-13.6k lost (checkpoint only every 5k). Cause: training rows held twice in RAM (sets + train copies, ~61k fp16 token rows ~40 GB each) plus slow allocator growth. Fix in train_sd.py: after the split keep only val rows in `sets`; checkpoint every 1k; MALLOC_ARENA_MAX=2 in py.sbatch. The restart with the hole fix (3751176) was cancelled while still loading and resubmitted with the memory fix: 3751192 (resumes from 10k, --hole_w 0 --cover_min 0.8,0.3).
- Judge validation on SD-10k renders (30 val scenes x 4 frames x 2 seeds = 240):
  m2f car F1 at label shifts 0 / 0.5 / 1 / 2 m: 0.260 / 0.243 / 0.223 / 0.166 (monotone, but drop at 1 m 0.037 < pre-set 0.15). SegFormer: 0.124 at 0, 0.076 at 2 m.
  Kerb error 1.47 m (cap 2) at every shift: expected, the shift moves only car labels, not the road -> kerb check needs a wrong-street control.
  Camera shuffle (tokens AND rays rolled by 3): F1 0.257 vs 0.260, pictures nearly identical (pulled/shuffle_check.jpg). The renderer places content by each token's ray, not by camera slot, so rolling both keeps the geometry consistent: not a control. (Good sign that it uses the rays.)
- Proper controls queued with sd_render.py --control: roll_tokens (tokens rolled, rays kept: front content claimed to be behind) 3751193 and swap_scene (tokens + rays from the judge scene 15 positions away) 3751196, each scored by m2f + SegFormer (3751194/95, 3751197/98).
- Reading so far: the judge is monotone but coarse at this model quality (F1 0.26). The 1 m criterion assumed sharper renders; the go rule uses paired bootstrap CIs, so a coarse judge errs towards "no difference" (= drop sim), which is the conservative side.

## 2026-11-25 19:35 — New experiment W: 3D-aware "world model" encoder (owner approved, runs in parallel)
- Gap vs Tesla: their renderer sits on a driving-trained vision encoder (3D, agents, memory); ours on generic per-frame DINOv2. Idea: features from a public camera-only nuScenes 3D model (BEV / occupancy) as renderer input, cached offline, judged with the geometry judge. Brief: council/bev-brief.md. Council round 1: Fable running (Astra unavailable in this session: no delegate_task; my own take will be the second view).

## 2026-11-25 19:25 — Hole fix for the main SD run
- SD 10k judge renders (pulled/sd10k_grid.jpg): no big black hole under the car (SD fills it), but dark speckles all over the road = lidar-key gaps (lidar keys are premultiplied, black between points) and splat floaters leaking through partly covered latent cells, which still had weight cover (and holes 0.05).
- Key coverage per latent cell (sample of 15 scenes): lidar mean 0.75, >=0.8 in 65% of cells, 0<c<0.5 in 14%; splat-trust mean 0.52, >=0.3 in ~65%, >=0.9 only 11% (trust is soft).
- train_sd.py --cover_min (one threshold per key set): cells below get zero weight. sd_run1 paused with STOP at ~13k and resubmitted (3751176, afterany 3751069) with --hole_w 0 --cover_min 0.8,0.3; everything else unchanged. SD's prior should fill the zero-weight areas instead of copying black.

## 2026-11-25 19:10 — Trust-mask comparison (valid version)
- 12 unseen val scenes, scored against true-colour trusted splat keys, trusted pixels only, low drone, 45M DiT 40k:
  low_mix_nomask L1 27.2 / LPIPS 0.399; low_mix_trust L1 27.8 / LPIPS 0.364 (mean image 33.1 / 0.524).
- Pictures (pulled/trust2_sheet.jpg -> docs/figs/trust_masks.jpg): the trust model copies the key's black hole under/behind the ego (never seen by a camera, trust low but not zero); the no-mask model fills it with mush. Verdict: helps on trusted pixels, but needs exact zero weight there + a prior to fill (SD). README updated.

## 2026-11-25 19:50
- scn-subset 3751130 done (100 pilot scenes). Sim renders 3751131-34 running: 38/100 scenes in 8 min. Checked scene-0001 by eye (pulled/sim_pairs_check.jpg): the trucks and parked car sit in the same places in the 6 cameras and in the drone key; our own car is hidden; the look is game-like (allowed by the owner, the go rule is placement only).
- Judge truth 3751146 and SD-10k judge renders 3751144/45 running; 10 validation scorers queued behind them.
- sd_run1 12.7k.

## 2026-11-25 19:00
- judge-gt crashed: shapely 1.8.5 (pinned by nuscenes-devkit) has a numpy-1 build of shapely.vectorized that breaks under numpy 2.4 -> point-in-polygon now via matplotlib.path (exterior minus holes). Rerun 3751146.
- Judge models downloaded (Mask2Former Swin-L Cityscapes panoptic, SegFormer-B5).
- Judge validation queued (after truth + renders of SD 10k checkpoint 3751144/45): both judges x label shifts 0/0.5/1/2 m on normal renders, and shift 0 on shuffled-camera renders (10 jobs 3751147-56) -> runs/judge/sd10k_*.json. Pass = F1 falls at each shift (>= 0.15 at 1 m), kerb error rises, shuffled cameras score worse.

## 2026-11-25 18:50 — Geometry judge built (council/PLAN-sim.md step 4)
- Session moved to another account after a usage limit; T3 orchestration tools (scheduled-task / delegate) are not available in this session, cluster work continues as before.
- scripts/sd_render.py: renders a checkpoint on the judge set = val scenes (sorted, with tokens) 30-59, 4 evenly spaced keyframes x 2 seeds, style 1, DDIM 25, CFG 2; --shuffle_cams rolls the six cameras by 3 (front<->back) for the validation control.
- scripts/judge_gt.py: exact truth in the low-drone view: nuScenes vehicle.car boxes within 30 m (>= 1 lidar pt) -> projected 3D corners -> convex hulls painted far->near, kept if visible part >= 8 px tall and >= 50% of the hull; drivable_area from the map expansion on a 0.2 m ground grid -> road mask + boundary points in ego metres; label shifts 0/0.5/1/2 m forward for validation (job 3751142 -> data/judge_gt).
- scripts/judge_score.py: Mask2Former Swin-L Cityscapes panoptic (cars score >= 0.5, road) or SegFormer-B5 (car connected components >= 30 px); predictions whose bottom pixel is on the ground within 30 m; one-to-one Hungarian matching at IoU >= 0.3 -> pooled F1; kerb error = symmetric mean nearest distance (ground metres, capped 2 m) between predicted road boundary and map boundary within 30 m in view; --compare = paired scene bootstrap 95% CIs. Mask2Former + SegFormer weights downloading on the login node (logs/dl_judge.log).
- Judge validation will use SD's 10k checkpoint (copied to runs/judge_ckpt_sd10k.pt): renders normal + shuffled cameras -> scores at shifts 0/0.5/1/2 m and shuffled; both judges.

## 2026-11-25 18:40
- VPN offline 18:15-18:30 at the user's request (not reconnected); back fine, no job lost.
- Full ScenarioNet conversion is slow and memory-bound (~250/850 scenes in 75 min, MaxRSS 50 GB at a 48 GB request: one full nuScenes index per worker). The pilot needs only 100 train scenes -> scripts/scn_convert_subset.py (ScenarioNet's own write_to_directory, 100 evenly spread train scenes, 2 workers) -> data/scn_pilot (job 3751130). The full conversion keeps running for scaling later.
- Because Slurm copies batch scripts at submission, the queued chain was resubmitted on scn_pilot: render 3751131-34 -> sim-enc 3751135/36 + sim-lat 3751137 -> arms A 3751138, B1 3751139 -> B2 3751140, C 3751141.
- sd_run1 11.7k.

## 2026-11-25 18:20
- sd_run1 resume confirmed: "resumed at 10000", 56,260 lidar train rows (clean + phone copy) + 4,606 splat rows; 10.8k steps.
- train_sd.py: --set keydir:tokdir:style:rows[:aug+aug] (fixed rows per set, each set with its own seeded generator for indices/timesteps/noise/dropout -> paired real draws across arms), --n_styles 3 (style 2 = sim), --seed, --save_at; tokens files may carry their own rays (sim). CameraAdapter(n_styles). Old path (no --set) unchanged for sd_run1.
- Pilot arms queued (slurm/sim_arms.sh) after the sim encodes: A 3751099 (real 11 lidar + 5 splat, 7k, snapshots 3k/5k/7k); B1 3751100 (2k sim-only, 16 sim rows) -> B2 3751101 (init B1, real 5k, snapshots 3k/5k); C 3751102 (real + 2 sim rows + 2 twin rows, 5k). All: seed 7, phone copies, H200 18 GB slice.
- low_mix_trust retrain at 36.8k/40k; sim conversion still running.
- Next: the geometry judge (renders of the judge set + Mask2Former/SegFormer vs projected nuScenes boxes/map + validation checks).

## 2026-11-25 17:50
- SD restart done by the watcher: sd_run1 stopped at ckpt step 10000 and resumed as job 3751069 (17:32) with --tokens_aug data/enc_aug_0, mem 120G (loading data).
- Corrected trust comparison, arm 1 (true-colour trusted splat keys, 12 val scenes): low_mix_nomask L1 27.2 / LPIPS 0.399 (mean image 33.1 / 0.524). low_mix_trust retrain at 8.2k/40k.
- Sim: shared phone augmentation moved to mrv/aug.py (encode_aug.py uses it). scripts/sim_encode.py: sim rig tokens clean + phone-aug (rays recomputed for the crop; sim renders are 448x252 so crop boxes are scaled to 1600x900 for the calibration). Queued after the 4 render jobs: sim-enc x2 (-> data/enc_sim, data/enc_sim_aug) and sim-lat (encode_latents -> data/lat512_sim, full cover).

## 2026-11-25 18:00 — Sim step 2 queued
- scripts/sim_render.py + slurm/sim_render.sh: 100 official train scenes (spread over the list), every other keyframe (~2,000 pairs), 6 calibrated rig cameras 448x252 (jpg) + ego-hidden low drone 512x288 (png, alpha 255) -> data/sim/{cams,keys}/<scene>/. 4 shards (3751064-67, 2080 Ti) after scn-convert (3751036).
- Next: encode (DINOv2 tokens clean + phone-aug with rays; sim keys -> lat512 with full cover), geometry judge (Mask2Former + SegFormer, projected nuScenes boxes/map), train_sd changes for arms B/C (style 2, sim-only phase, twin-key rows).

## 2026-11-25 17:50 — Sim step 1 (replay verification) PASSED
- scripts/sim_verify.py on 5 mini scenes: replayed car centres vs nuScenes annotations 0.0 m (after one constant map offset per scene; spread 0.0); sim ego origin = logged ego (rear axle, 0.0 m), so cameras placed from calibrated_sensor are right; per-camera HFOV from intrinsics ~65 deg (back ~90 deg); hiding all children of agent.origin except the camera hides our car (~2.5% of drone pixels change).
- Lens vs K: first 18.6 px median (fail) -> fix 1: standalone Panda3D lens matches K exactly, so not the lens maths; fix 2: MetaDrive's lens is a standard PerspectiveLens; the principal-point film offset had the wrong sign (local test: +sign 22 px, -sign 0 px). After the fix: 0.00001 px. Pictures: pulled/sim_verify.jpg / docs/figs/sim_verify.jpg.
- Next: step 2 (render ~2,000 ego-hidden replay pairs from official train scenes once scn-convert finishes; encode clean + phone-aug tokens, 512x256 sim keys) and the geometry judge.

## 2026-11-25 17:35 — Owner: game-like look is fine if accurate
Go rule for sim now rests only on placement (car F1 >= +0.05 or kerb error -10%, CI excluding zero, no regression on the other); LPIPS and "sim look" are reported, not used to reject (council/PLAN-sim.md updated).

## 2026-11-25 17:30 — Decision rules for the sim verdict agreed with the owner
Written into council/PLAN-sim.md ("Decision rules"): alignment fails -> 3 fixes then drop; judge fails -> backup judge then inconclusive/skip; C wins -> fold sim rows into sd_run1 at its next checkpoint; B wins -> 8k pairs + new main run (replaces sd_run1 only if better at equal real steps); both -> larger gain, tie -> C; neither -> drop sim; game-like look -> one fix then drop. Then always: final eval, optional Difix, distil, try_my_photos + how-to, final write-up.

## 2026-11-25 17:20 — Sim plan final (council/PLAN-sim.md)
- Round 4: Astra accepts Fable's two points: B's sim phase trains adapter + LoRA; placement check = graded ground shifts 0.5/1/2 m (F1 must fall at each step, >= 0.15 at 1 m). Final plan written to council/PLAN-sim.md.
- Step 1 started: ScenarioNet conversion of v1.0-trainval (scn-convert, 8 CPUs) -> data/scn_full. Next: replay verification script (intrinsics/FOV per camera, ego origin + hide, overlays).

## 2026-11-25 17:15 — Sim council converged (rounds 1-3); SD restart armed
- Agreed plan (council/s3-astra.md, council/s3-fable.md): first verify the ScenarioNet replay (per-camera intrinsics/FOV, ego origin + hide, overlays; stop on systematic offsets). Render ~2,000 ego-hidden replay pairs from official train scenes (512x288 -> crop 32 rows, low drone), clean + phone-augmented inputs. Three arms from identical init with paired real batches (16 real rows: 11 lidar + 5 splat): A real only; B 2k sim updates then 5k real (A extended to 7k for equal updates); C real + 2 sim-camera rows + 2 real-camera rows targeting the replay key. Style rows 0 lidar / 1 splat / 2 sim, shared learned null. Score every arm at 3k and 5k real steps.
  Geometry judge: frozen Mask2Former Swin-L Cityscapes panoptic (cars conf >= 0.5, road), one-to-one matches IoU >= 0.3 to projected nuScenes vehicle hulls within 30 m (>= 8 px), F1; road-edge distance in ground metres vs projected drivable-area boundaries (capped 2 m); 4 frames x 30 val scenes outside the final 30, 2 paired seeds; second judge SegFormer-B5 must agree on the sign; validation checks must pass first (graded shifts, camera shuffle).
  Go rule: >= 0.05 F1 gain or >= 10% kerb-error reduction over A with paired scene-bootstrap 95% CI excluding zero, no regression on the other, masked splat LPIPS worse by <= 0.01, no sim look in pictures. Budget ~8 GPU-h.
  Open (round 4 to Astra): B's sim phase trains adapter+LoRA (Fable) or adapter only (Astra); label-shift check as graded 0.5/1/2 m ground shifts (Fable) vs 24 px.
- SD restart armed: watcher job 3751035 waits for sd_run1's next checkpoint (10k), scancels 3750658 and resumes runs/sd_run1 with --tokens_aug data/enc_aug_0, --mem 120G (same keys: lat512 lidar-low + splat-low-trust; SD path unaffected by the encode.py colour bug).

## 2026-11-25 17:25 — ScenarioNet replay works
- Needed: setuptools<81 (pkg_resources) in envs/sim and the nuScenes map expansion v1.3 (downloaded to data/nuscenes/maps and data/nuscenes_full/maps). nuScenes-mini converted to data/scn_mini.
- Pictures (pulled/sim_scn.jpg, docs/figs/sim_scenarionet.jpg): real layouts with logged cars, trucks, pedestrians, barriers, crosswalks at their recorded places; scenery is still grass hills (no buildings). A silver car at the bottom of some drone views = probably our own car (hide failed) or the rear-axle origin offset -> part of the replay verification step. Cameras still at 70 deg (real: ~65 front/side, ~89 back).
- Council round 3: Astra = include C now, ~8 GPU-h (B: 2k sim then 5k real; A to 7k for equal updates), automatic scoring with frozen Mask2Former Swin-L Cityscapes (cars IoU>=0.3 one-to-one vs projected nuScenes hulls, road-edge distance in metres), validation checks. Waiting for Fable round 3.

## 2026-11-25 17:15 — trust-mask comparison INVALID (bug found in the pictures), fixed and rerunning
- First numbers looked like a big win (low_mix_trust vs trusted splat keys: L1 25.7 / LPIPS 0.321; low_mix_nomask 34.1 / 0.405; mean 34.4 / 0.497), but the pictures (pulled/trust_result.jpg) showed the keys darkened wherever trust is low and the trust model drawing dark patches.
- Cause: scripts/encode.py premultiplied RGB by alpha for every key. Right for lidar keys (alpha = covered), wrong for trust-masked splat keys (alpha = loss weight only). The truth used for scoring was also darkened, favouring the model trained on it. The SD path is NOT affected (encode_latents.py keeps splat colours).
- Fix: encode.py keeps splat colours ("splat" in --targets). Old encodes kept as data/enc_splat_tv_low_trust_BAD_premult, old run as runs/low_mix_trust_bad. Re-encode (3751033) -> retrain low_mix_trust (3751034) -> re-score both arms against the true-colour trusted keys (eval-lownm2, eval-lowtr2). Minor: earlier splat encodes without trust (opacity alpha ~1) were premultiplied too; effect small (sky only).
- Sim council round 3 running (council/sim2real-round3.md).

## 2026-11-25 17:25 — Sim council round 1 done, round 2 running
- Both seats: ScenarioNet nuScenes replays first; skip MetaDrive procedural maps; CARLA only after a positive pilot (+ Vulkan check); cheap pilot before any big run. Astra: ~3.6 GPU-h pilot, go only if sim pretrain improves car-placement F1 by >= 5 points on held-out real scenes while meeting kerb/appearance thresholds; controls for equal real exposure and equal total updates. Fable: use sim for layout not looks; keep ~20% sim rows in the mix with their own style token instead of only pretraining; "twin keys" (ScenarioNet-rendered complete key of the same real moment paired with the real cameras); layout metric = nuScenes map + 3D boxes projected into the drone view vs a segmenter on our renders, validated first; ~8 GPU-h pilot.
- Fable's checks: nuScenes intrinsics give ~65 deg HFOV for five cameras and ~89 deg for the back camera (our sim test used 70 for all); ScenarioNet ego origin may be the rear axle with length/width swapped -> alignment overlay first.
- low_lidar (low-drone lidar keys, 45M, rays) on 30 val: L1 23.9 / LPIPS 0.348 (mean image 30.1 / 0.615); A_rays (high) 26.3 / 0.377 (mean 30.6 / 0.655) -> low at least as good relative to baseline, with cleaner keys.
- Round 2: council/sim2real-round2.md -> council/s2-astra.md, council/s2-fable.md.

## 2026-11-25 17:15 — Sim council round 1 (user asked: research, not winging it)
- Council: Astra (codex gpt-6.2-astra, xhigh) and Fable (claude-fable-5-2, high), same brief council/sim2real-brief.md: literature on mixing sim and real for conditional generation and driving view synthesis, simulator choice, data recipe, training recipe for SD+LoRA+adapter, geometry-aware evaluation, cheap go/no-go pilot. Outputs: council/s1-astra.md, council/s1-fable.md.
- Round 2 (after both): swap plans so each critiques the other and they converge (council/s2-*.md); then build the agreed plan. No sim training until then; data-generation tooling (ScenarioNet test) continues.

## 2026-11-25 17:05 — SD 5k preview
- sd_run1 at 5k (eval_only copy runs/sd_run1_5k, DDIM 25, CFG 2, style 1) on 4 val scenes vs low splat keys with trust: masked L1 33.5, LPIPS 0.390 (low_mix_nomask DiT 45M on 12 scenes: 34.1 / 0.405; different res, rough comparison).
  Pictures (pulled/sd5k.jpg): photo-like streets (trunks, kerbs, lane lines, poles, trucks) roughly where the key has them (truck ahead on tree-lined road matches); one scene is still leafy noise; positions not precise yet. Big qualitative jump over the DiTs.
- ScenarioNet (nuScenes-mini replay in MetaDrive) installing; test job follows (sim-scn).

## 2026-11-25 17:00 — Experiment S feasibility, step 1: MetaDrive works
- envs/sim (Python 3.11, metadrive-simulator via uv on the login node; the compute node timed out downloading a Python build). Headless render on a 2080 Ti: env up 60 s; 6 rig cameras + low drone camera in 0.16 s per frame (448x252).
- Pictures (pulled/sim_test, docs/figs/sim_metadrive_*.jpg): rig directions correct; drone key complete and sharp (every lane line/kerb). But the procedural world is an empty highway in grassy hills: no buildings, trees, parked cars; traffic distant -> big domain gap to nuScenes city streets. Ego car visible in the drone shot (real keys have none) -> hide it in sim drone shots.
- Cluster has apptainer/singularity (CARLA container possible); login node has no Vulkan ICD dir (check on a GPU node).
- Next: ScenarioNet nuScenes replay in MetaDrive (real layouts + real traffic, simple graphics) and, in parallel, a CARLA apptainer Vulkan test.

## 2026-11-25 16:35 — plan: fold side-job lessons into sd_run1 (user approved)
- train_sd.py can now pause safely: `touch runs/<out>/STOP` (checked every 50 steps) or scancel/SIGTERM -> saves ckpt.pt (atomic) and exits; rerunning with the same --out resumes (step, optimizer, LoRA, adapter). Data/args may change on resume; model shape may not.
- Both phone-augmented copies done (data/enc_aug_0, data/enc_aug_1, 850 scenes each).
- RESTART PLAN for sd_run1 (old code, no pause handler): wait for its 10k checkpoint (log shows step 10000 and ckpt.pt mtime updates), scancel it, then resubmit with the SAME --out runs/sd_run1 plus `--tokens_aug data/enc_aug_0` and `--mem 120G` (one copy to keep RAM down), on gpu:h200_1g.18gb:1, time 2 days. Splat keys: keep data/lat512_splat_tv_low_trust if low_mix_trust beats low_mix_nomask on trusted pixels (eval-lowtr), else switch --keys2 to a no-trust 512 encode (encode_latents.py on data/targets_splat_tv_low).

## 2026-11-25 16:20 — NEW EXPERIMENT S (user, side chat): sim pretraining -> real fine-tune
In a simulator the answer key is perfect (every pixel known, no holes/smears). Pretrain the SD renderer (same camera adapter + LoRA, rays, DINOv2 tokens, low drone 9 m back / 4.5 m up / look 15 m ahead at 0.5 m, 60 deg HFOV) on sim pairs: 6 cameras placed like the nuScenes rig (positions, yaw, FOV) -> drone view. Then fine-tune on real keys and compare with real-only at equal real steps. Do NOT stop sd_run1 or the low_* runs.
Simulator (cheapest working wins; findings logged here): 1) MetaDrive + ScenarioNet (pip, Panda3D, headless EGL; ScenarioNet can replay nuScenes layouts); 2) CARLA 0.9.x -RenderOffScreen (better graphics, ~20 GB, Vulkan, apptainer?) if 1 is too cartoonish or easy; 3) fallback SHIFT-like synthetic data only if both fail.
Target: a few thousand pairs (300-500 short episodes x 2 Hz), mixed maps/weather/time, traffic + pedestrians; also through the phone augmentation.
Comparison on the 30 real val scenes (low-drone keys with trust masks): A) sd_run1 real-only; B) sim-pretrain -> real fine-tune, same real steps; optional C) real + sim mixed with a new style id. Masked LPIPS/L1 + pictures; say whether geometry (road edges, car positions) improves, not just texture.

## 2026-11-25 15:15
- sd_run1 2k/40k (loss 0.19). low_mix_nomask 39.8k/40k, low_lidar 13k, low_mix_trust 12k. Evals queued after each: low_mix_{nomask,trust} vs low splat keys with trust masks, style 1 (3750831/32); low_lidar vs low lidar keys (3750833). Aug copies 1213/1700.

## 2026-11-25 14:45
- sd_run1 RUNNING: 34,149 lidar-low frames (28,130 train) + 2,783 splat-low-trust frames (2,303 train); ~1.8 min/100 steps -> 40k ~ 12 h. low_lidar 3.6k, low_mix_nomask 11.6k, low_mix_trust 4.2k (of 40k). Aug copies running.

## 2026-11-25 14:15
- Video fix 3 WORKED: full_c_self (fine-tuned 20k with --self_prev 0.5), eval s_prev 0.3, 30 val scenes: LPIPS 0.368 (best of all models; full_b 0.374), L1 29.2, first5->last5 28.5->30.3 (+1.8; original +7.8, fix 1 +3.4). Pictures (pulled/selfprev_fix.jpg): last frames keep road texture instead of greying out.
- sd_run1 RUNNING (loading data). lat512_full_low 850/850. One enc-low shard still running -> low_lidar / low_mix_* then start. Aug copies 691/1700.

## 2026-11-25 13:55
- VPN dropped (gateway unreachable, "Connection closed by UNKNOWN port 65535") -> FortiTray AppleScript reconnect, gateway back after 15 s.
- Style-mix eval: vs high splat keys on 12 val scenes: full_b L1 37.5 / LPIPS 0.675; mix_high style 1: 28.9 / 0.639 (mean image 34.5 / 0.873). vs lidar keys (30 val, style 0): mix_high 28.4 / 0.372 (full_b 27.8 / 0.374) -> lidar skill kept.
  BUT pictures (pulled/stylemix.jpg): the val high-drone 15k-iter splat keys are very messy, and mix_high's splat style is soft mush; the lidar-trained model looks sharper and more road-like. The L1 win is blur matching a noisy key. Verdict: did not work with high-drone splat keys; low-drone + trust-mask keys (low_mix_* runs) are the real test.
- c-selfprev DONE -> eval with s_prev 0.3 (3750778). lat512_full_low 671/850; then sd_run1 + low_* runs start.

## 2026-11-25 13:15
- mix_high DONE (init full_b; lidar style 0 + trainval high splats style 1, mix 0.5, 30k, final loss ~0.14). Evals: full_b vs splat keys (3750742), mix_high style 1 vs splat keys (3750743) on the val scenes that have splats; mix_high style 0 vs lidar keys on 30 val scenes (3750744).
- Low lidar targets 850/850 DONE; enc-low and lat512-low running -> then low_lidar / low_mix_{nomask,trust} and sd_run1 start.
- c-selfprev 18.8k/20k. Augmented copies running (copy 0: 89/850).

## 2026-11-25 12:50
- Aug test OK: augmented tokens align with clean ones (same sample order), mean cosine to clean tokens 0.81. Launched 2 full augmented copies of all 850 scenes (8 jobs) -> data/enc_aug_{0,1}.
- train_sd.py: --tokens_aug (phone-like copies as extra training rows, train scenes only, own rays) and --init. CameraAdapter + LORA_TARGETS moved to mrv/sd.py.
- scripts/try_my_photos.py: 6 photos -> EXIF-rotate, centre-crop 16:9, 448x252 -> DINOv2 -> pooled tokens; default rig rays assets/default_rays.pt (mean over 850 scenes); renders with the SD checkpoint (CFG) or a DiT checkpoint; CUDA / MPS / CPU.
- mix_high 25.4k/30k; c-selfprev 14.8k/20k; low lidar targets 824/850.

## 2026-11-25 12:20
- mix_high 20k/30k (loss 0.16); c-selfprev 10.6k/20k; low lidar targets 777/850 (then enc-low -> low_lidar / low_mix_* and lat512-low -> sd_run1).
- P (phone photos) started: scripts/encode_aug.py — per camera independently: zoom-crop z in [1,1.3] at a random offset (narrower FOV + small yaw/pitch jitter), brightness/contrast/saturation x0.75-1.25, hue +-0.03, gamma 0.8-1.25, blur (50%, sigma <= 1.2), JPEG q 35-95; rays recomputed for the patches inside the crop. Small test job 3750677 (2 scenes) before the full copies.

## 2026-11-25 11:50
- mix_high 14.6k/30k (loss 0.19); c-selfprev 6.6k/20k; low lidar targets 613/850; low splat keys encoded with/without trust (enc_splat_tv_low, enc_splat_tv_low_trust).
- Queued after enc-low (all 45M + rays, 40k, bs 64): low_lidar (3750659, step-2 model comparison vs A_rays at the high drone), low_mix_nomask (3750660) vs low_mix_trust (3750661) (step-1 trust-mask comparison; splat mix 0.3).

## 2026-11-25 11:50 — Step 3 started early (user asked): SD 1.5 renderer
- scripts/train_sd.py: frozen SD 1.5 UNet (base weights bf16/fp16) + LoRA r=64 on attention/proj/ff + CameraAdapter (DINOv2 tokens 384->768, + ray MLP, per-camera + 2D positions, style token, 2-layer transformer -> 865 tokens replacing the CLIP text tokens) + learned null tokens (10% drop) for CFG. SD's own DDPM schedule, epsilon prediction, coverage/trust-weighted loss (hole_w 0.05), 512x256 keys (latent 4x32x64). Render: DDIM 25 steps, CFG 2.0. 42.9M trainable.
- Smoke test (300 steps, 70 trainval splat keys low+trust, H200 1g.18gb, bf16): runs, ~1.7 min / 100 steps at bs 16. Pictures (pulled/sd_smoke.jpg): photo-like street textures already (trees, road, buildings), not yet following the cameras (as expected at 300 steps). 2 val scenes LPIPS 0.428.
- Queued sd_run1: keys = low-drone lidar (style 0, all 700 train scenes) + low-drone splat with trust masks (style 1, 70 scenes), mix 0.3, 40k steps (~11 h), after 4 encode_latents shards that wait on the low lidar targets (608/850 done).

## 2026-11-25 11:20
- mix_high 9k/30k (loss 0.22). c-selfprev 2.4k/20k (its "min" column is wrong: the self-prev loop reused the timer variable t0; renamed to ta/tb for future runs, training itself unaffected).
- Trust masks for the 70 trainval low splats done. Encoding low splat keys without / with trust masks (tokens reused from enc_full): 3750618 / 3750619 -> data/enc_splat_tv_low{,_trust}.
- Low lidar targets 430/850; enc-low (4 shards) waits on them.
- One rsync dropped mid-transfer (unexpected EOF); retried fine.

## 2026-11-25 10:45
- Video drift fix 1+2 combined (s_prev 0.3 + reset every 8): L1 26.76, LPIPS 0.376, first5->last5 27.5->30.4 (~same as s_prev 0.3 alone). Single-frame full_b stays flat (29.4->29.6), so the remaining +3 is real drift.
- Fix 3 started: train.py --self_prev 0.5 — for half the samples the previous frame is drawn by the model itself (EMA, 2 steps, single-frame mode, from the previous keyframe's camera tokens) instead of the perfect key; fine-tune full_c for 20k steps, lr 1e-4 (c-selfprev 3750591, H200 18 GB slice).
- mix_high running (3.6k/30k). Low lidar targets 266/850. Trainval low splats 70/70 rendered; trust masks running (16 done).

## 2026-11-25 10:20
- Experiment A on 30 val scenes, 4 steps (45M, 700 scenes, 40k): no rays L1 26.22 / LPIPS 0.384; rays 26.27 / 0.377. Small gain for rays; keep them (also needed for phone photos). Pictures: pulled/rays_ablation.jpg (kerbs/road edges slightly better placed).
  Note: these 45M runs (L1 26.2) beat full_b (110M, 27.8) on L1 and are close on LPIPS (0.377-0.384 vs 0.374).
- Video drift fixes on full_c (30 val scenes, 4 steps): s_prev 0.1: LPIPS 0.389, L1 first5->last5 27.4->35.2 | fix 1 s_prev 0.3: 0.374, 27.4->30.8 | s_prev 0.5: 0.408, 28.5->34.2 (beyond the 0-0.3 training range) | fix 2 reset every 8: 0.385, 27.3->30.1. Best: s_prev 0.3. Drift reduced by more than half, not gone. Next: s0.3 + reset 8 (3750559); fix 3 (train on own rollouts) if needed.
- mix_high OOM'd on an 11 GB 2080 Ti (110M, bs 64) -> resubmitted on gpu:h200_1g.18gb (3750558). Teaching partition gres names: nvidia_geforce_rtx_2080_ti, nvidia_rtx_a6000, h200, h200_3g.71gb, h200_1g.18gb.
- Full low-drone lidar targets: 74/850 scenes so far (8 shards running).

## 2026-11-25 10:00
- ssh master had gone stale ("Connection closed by UNKNOWN port 65535"): ssh -O exit mlp, reconnect -> fine.
- Stage 5 video model full_c (512x16, 80k) on 30 val scenes, whole-scene rollouts, 4 steps, s_prev 0.1: masked L1 29.95, LPIPS 0.389 (single-frame full_b 27.8 / 0.374). DRIFT: L1 first 5 frames 27.4 -> last 5 frames 35.2. Stage counts as failed; rescues:
  fix 1: more noise on the previous frame at run time (s_prev 0.3, 0.5: 3750530/31); fix 2: restart from pure noise every 8 frames (3750532); fix 3 if needed: retrain on the model's own rollouts.
- Experiment A finished (40k each): evals 3750528 (no rays) / 3750529 (rays). evaluate.py now handles rays (+ --reset_every).
- enc_splat_tv done (70 scenes). Style-mix run mix_high: init full_b, lidar keys (style 0) + trainval splat keys high drone (style 1), mix 0.5, lr 1e-4, 30k steps (3750533).
- Low-drone lidar keys for all 850 scenes: 8 target shards -> 4 encode shards with --tokens_from data/enc_full (no DINOv2 rerun) -> data/enc_full_low. Trust masks for the 70 trainval low splats queued after the last re-render (4 shards).

## 2026-11-25 09:15
- Trust masks v2 (opposite views count half): mean trust per scene low 0.38-0.54, high 0.30-0.63. The dark band at the bottom of the low-drone masks is unchanged: it is the ground under/right around our own car (bottom of the low-drone frame ~ the ego position), which no camera sees -> correctly untrusted (the final view draws the car there). v1 diagnosis was wrong; the angle change is kept as reasonable.
- Experiment A smoothed train loss (no rays / rays): 10-15k 0.841/0.826, 20-25k 0.746/0.753, 28-34k 0.681/0.667. Small edge for rays so far; single-batch losses were misleading.
- full-train-c (video, 512x16, 80k) DONE; in-training rollout masked L1 on 6 val scenes 24.7-34.4. Full eval with drift: eval-full-c 3750514 (30 val scenes, 4 steps).
- enc_splat_tv 67/70; trainval low re-renders 44/70.

## 2026-11-25 08:45
- Trust masks v1 done on mini (low drone mean trust 0.30-0.51 per scene, high drone 0.13-0.50 -> low drone keys more trustworthy, matches the pictures). Pictures (pulled/trust_masks.jpg): street, cars and facades trusted, sky not; but the road right behind the car was untrusted although the back camera sees it — the angle weight clip(cos,0,1) gave 0 when the real and drone views point in opposite directions. Fix 1: weight 0.5 + 0.5*clip(cos) (flat ground looks the same from both sides). Rerun 3750477/78.
- Experiment A at 24k/40k: train loss with rays 0.658 vs 0.738 without (clear gap).
- enc_splat_tv 28/70 encoded; low re-renders of the 70 trainval splats running (7 jobs); full-train-c 75.8k/80k; mini scene 9 splat finally running on the A6000.

## 2026-11-25 08:15
- All 70 trainval splat scenes done -> encoding them (enc-splat-tv 3750446); low-drone re-renders of the 70 queued (7 jobs x 10 scenes, 3750447-53).
- Step 2 keys: mini splat re-render at the LOW drone done. Pictures (pulled/splat_high_vs_low.jpg): far fewer floaters and colour junk, sharper cars/facades/excavator, sky now visible; remaining smears at the bottom corners (right behind the car, unseen by any camera) and one sky glitch (scene-0655). Low drone is clearly the better key; model comparison at both heights still to do.
- Experiment A at ~14.5k/40k: loss with rays 0.779 vs 0.790 without (gap narrowing). full-train-c 70k/80k.

## 2026-11-25 07:45
- Stage 7 distillation (1-step student of full_b's 4-step output, 15k steps, final loss 0.011): val masked L1 27.4 (vs 28.9 plain 1-step) but LPIPS 0.394 (vs 0.381 plain 1-step, 0.374 plain 4-step). Pictures (pulled/distill_sheet.jpg): very close to the 4-step teacher, a bit smoother. Verdict: no gain over plain 1-step sampling; keep plain few-step sampling for now and revisit with a perceptual/adversarial loss for the SD1.5 model.
- Experiment A at 5.2k/40k: train loss with rays 0.894 vs 0.952 without (early).
- Trainval splats 67/70; mini re-renders running (12 scene dirs written so far); full-train-c 65k/80k.

## 2026-11-25 07:20
- calib_rays.pt done (850 scenes, 6x144 Plücker rays each). Lidar keys at the low drone done for mini (coverage 0.21-0.81, about the same as high).
- Experiment A launched: 45M DiT, all 700 train scenes, 40k steps, bs 64, identical except --rays: runs/A_norays vs runs/A_rays.

## 2026-11-25 07:15
- Stage 7 step baselines, full_b on 30 val scenes: 1 step L1 28.9 / LPIPS 0.381; 2 steps 27.5 / 0.383; 4 steps 27.8 / 0.374. One step is already within 0.007 LPIPS of four -> flow-matching paths are nearly straight here; distillation (10.6k/15k, loss 0.031) can only close a small gap.
- full-train-c 59.8k/80k. Trainval splats 59/70. Mini low/high re-renders running; trust masks queued behind them.

## 2026-11-25 06:50
- Trainval splats 51/70; full-train-c 54k/80k; distillation + 1/2-step baselines running; SD1.5 UNet/VAE + Difix + peft downloaded (DL_DONE). CPU jobs (calib rays, lidar-low) and the mini low/high re-renders still queued.
- Step 1 code: scripts/trust_masks.py — lift each drone pixel with the splat depth, project into the 6 real cameras of that keyframe, visible = in image, in front, < 60 m, not behind a nearer lidar surface (per-camera lidar depth map at 1/4 res, 5x5 min-spread); weight by cos(angle between drone and real view) x exp(-dist/60); trust = opacity x sqrt(best). Written as key alpha. Queued after the mini re-renders (3750396 low, 3750397 high).

## 2026-11-25 06:50 — camera-only, and a phone-photo goal (user, side chat)
- The model stays camera-only (no lidar input at run time). Experiment A (calibration rays) stays.
- New goal: work from just 6 phone photos taken in a street (front, front-left, front-right, back-left, back, back-right) and draw the drone view.
  - When no calibration is given, assume the standard nuScenes rig directions and FOV (mean rays over all scenes).
  - Training robustness so phone photos are in distribution: random crop/scale (FOV ±15%), centre-crop to 16:9, colour/exposure jitter, small per-camera yaw jitter, JPEG/blur. Ray inputs must follow the same crop/scale/yaw so geometry stays consistent.
  - Deliver scripts/try_my_photos.py --front a.jpg --front_left b.jpg ... --out drone.png (cluster GPU, or CPU/MPS on the Mac for a small model) and a README how-to for taking the 6 photos.

## 2026-11-25 06:40
- Public repo created (user request): https://github.com/TarunTomar122/mini-reality-view — README in the style of the user's spiking-vsr write-up, figures in docs/figs, explainer moved to docs/explainer.html, Slurm log paths made relative (no usernames in the repo). Kept updated as results come in.
- Experiment A code: scripts/calib.py (per-scene K + cam->ego -> Plücker rays per pooled patch, data/calib_rays.pt, job 3750368), Renderer(rays=True) adds an MLP ray embedding in context(), train.py --rays. SD1.5 UNet/VAE downloaded; Difix downloading.

## 2026-11-25 06:30 — experiment A approved (user, side chat)
A) Camera calibration as input (camera-only, Tesla-like): add a per-patch ray embedding (ego-frame direction from K + cam->ego, plus camera origin; Plücker or dir+origin -> MLP) to the DINOv2 tokens in Renderer.context(); store K and cam->ego (or the 16x9 ray grid) at encode time. Compare with vs without rays.
Placement: finish runs -> 2 (lower drone) -> 1 (trust masks) -> A (quick ablation on the from-scratch DiT) -> 3 (SD1.5 with the best camera-only inputs, incl. rays) -> distil -> final eval -> write-up. Rays go into the SD1.5 camera adapter too. Scheduled task prompt updated.

## 2026-11-25 06:20
- Queue busy: lidar-low, mini splat re-renders (low/high), distillation, step baselines all pending. Trainval splats 39/70; full-train-c 49k/80k.
- Step 3 research: Difix is public (code nv-tlabs/difix3d, weights HF nvidia/difix, research/non-commercial licence; SD-Turbo based, single step). SD 1.5 weights via HF mirror sd-legacy/stable-diffusion-v1-5 (CreativeML OpenRAIL-M). Full fine-tune of the 860M UNet won't fit 11 GB (AdamW states ~14 GB) -> plan: frozen SD1.5 UNet + LoRA + a trainable camera-token adapter replacing the CLIP text tokens; 512x256 (latent 32x64, divisible by 8).
  Downloading SD1.5 UNet/VAE, Difix, sd-turbo licence + installing peft on the login node (logs/dl_sd.log). scripts/encode_latents.py: 512x256 latents of any key set (crop top 32 rows; alpha = weight).

## 2026-11-25 06:10
- Step 2 (lower drone) started: mrv/nusc.py CHASES {"high": 13 m back / 8.5 m up / look 12 m ahead, "low": 9 m back / 4.5 m up / look 15 m ahead at 0.5 m}; make_targets.py and ds_render_chase.py take --chase. ds_render_chase now also saves depth + camera (…_depth.npz) for trust masks.
  slurm/ds_render.sh re-renders trained splats with no retraining. Jobs: mini splats low (3750359) and high-with-depth (new id) -> data/targets_splat_mini_{low,high}; lidar keys low on mini (3750360) -> data/targets_lidar_low.

## 2026-11-25 06:00 — NEW PLAN (user, via side chat; project is now multi-day)
Why: even splat keys aren't clean (floaters, edge smears); the model can only chase its answer key. Finish the running work (full_c video, 70 trainval splats, distillation + step baselines, style-mix run), then:
1) Trust masks: per-pixel confidence for splat keys (opacity x seen-by-a-real-camera via depth reprojection into the 6 cameras at nearby frames / depth consistency), stored as target alpha so the loss weights it; low weight elsewhere. Compare with vs without on held-out scenes.
2) Lower drone: second chase camera ~4-5 m up, ~8-10 m back (configurable in mrv/nusc.py, ds_render_chase.py, make_targets.py); re-render splat + lidar keys from existing splat checkpoints; compare keys and models at both heights; pick one or train on both with a pose input.
3) Pretrained generator: fine-tune an open pretrained diffusion model (start: SD1.5 UNet with camera-token cross-attention, same SD VAE latents; also SD-Turbo/SDXS, PixArt/Sana; Difix/Difix3D+ to clean splat keys; Cosmos/Vista/OpenDWM as possible teachers). Check weights, licence, GPU fit. Then distil to 1-4 steps.
Order: finish runs -> 2 -> 1 -> 3 -> distil -> final eval -> README + plain HTML results page. Scheduled 30-min task prompt updated with this plan.

## 2026-11-25 05:45
- Trainval splats 28/70; full-train-c 43.8k/80k (loss 0.45). Distillation + step baselines still queued (GPU queue busy); mini scene 9 waiting for the A6000.

## 2026-11-25 05:20
- Trainval splats 19/70 rendered, 19 running, 32 queued, no failures. full-train-c 38.4k/80k.
- Stage 7: scripts/distill.py — 1-step student copies full_b's 4-step output from the same noise (latent MSE), 15k steps (3750324) -> eval at 1 step (3750327).
  Baselines for full_b at 1 and 2 steps on 30 val scenes (3750325/6); 4 steps already measured (LPIPS 0.374).

## 2026-11-25 04:50
- Trainval splats: 2/70 rendered, 19 running (expect ~4 h for all). Mini splats 9/10 (scene 9 waiting for the A6000). full-train-c 33k/80k.
- Prepared stage-6 training: Renderer(n_styles=2) adds a learned "target style" embedding (0 = lidar target, 1 = splat target);
  train.py --enc2 <splat enc> --mix 0.5 --init <ckpt> trains on both sets (lidar teaches layout from ~700 scenes, splat teaches complete appearance from ~57), renders with --render_style 1.
  evaluate.py --truth_enc <splat enc> --style 1 scores against splat targets. Smoke-tested on CPU (old checkpoints load with only style.weight missing).

## 2026-11-25 04:15
- Stage 8 eval, 30 val scenes, 4 steps:
  | run | data | model | masked L1 | LPIPS |
  | mean image | - | - | 30.6 | 0.655 |
  | full_a | 340 scenes | 45M, 40k | 28.1 | 0.412 |
  | full_b | 700 train scenes | 110M (512x16), 80k | 27.8 | 0.374 |
  Pictures (pulled/full_b_sheet.jpg): sharper and more detailed than A; road shape follows the truth (curves match). It also copies the lidar-ring stripes and holes of the targets -> splat targets should fix that.
- Sky masks for 70 trainval scenes done; 19 splat jobs running, 51 queued. full-train-c (video) 27.8k/80k. Mini splat scene 9 queued for the A6000.

## 2026-11-25 03:45
- full-train-b DONE (512x16, 80k steps, final loss ~0.24). In-training val render (25 steps) masked L1: 0003 30.6, 0012 26.4, 0013 27.2, 0014 29.5, 0015 28.4, 0016 29.6 (~same as run A on L1). Proper eval (30 val scenes, LPIPS) job 3750251.
- full-train-c (video) at ~20k/80k.
- Trainval per-scene preprocessing: all 64 finished; sky masks 22 done / 13 running / 35 queued; 70 splat jobs queued behind them.
- Mini splats 9/10: scene 9 (scene-1100) had OOM'd on an 11 GB 2080 Ti (4.5 GB reserved-unallocated, fragmentation) -> rerun on A6000 with expandable_segments (3750252).

## 2026-11-25 03:15
- Cleaned up: the 70 old ds-tv jobs (depending on the cancelled bulk sky jobs, DependencyNeverSatisfied) were still queued -> cancelled. Now 70 per-scene chains (18 preprocessing running, 14 queued).
- full-train-b 67.2k/80k (loss 0.24) -> done in ~20 min; full-train-c 17k/80k (loss 0.71). Mini splats 9/10.

## 2026-11-25 02:50
- full-train-b step 46.6k/80k (loss 0.32); full-train-c (video) 11.8k (loss 0.76). Splat mini: 8/10 done, scene 7 running fine after the clamp fix.
- Trainval drivestudio preprocessing was thrashing: 2 jobs x 8 workers each holding the full nuScenes index (MaxRSS 67 GB at a 64 GB limit, 32 min CPU in 2 h), only ~6/70 scenes complete after 2 h.
  -> cancelled; new slurm/ds_tv_chain.sh submits per scene: preprocess (1 worker, 2 CPU, 24 GB) -> sky masks -> splat 15k -> chase render. Scenes 006/032/034/042/067/069 were complete and skip preprocessing.

## 2026-11-25 02:15
- Scene 7 fix worked: no NaN, 15k/30k iters.
- full-train-b step 25.8k/80k after 37 min (fast GPU), loss 0.52; full-train-c (video) step 6.4k (slower slice), loss 0.82.
- Trainval drivestudio preprocessing running (8 scenes in parallel per job, ~600-1200 images written per scene so far); sky masks + 70 splat jobs queued behind it.

## 2026-11-25 01:50
- Encode DONE: 850/850 scenes in data/enc_full. full-train-b (512x16 single-frame) and full-train-c (512x16 video --prev) running: ~6 min/1k steps -> 80k in ~8 h (ETA ~09:45).
- Splat targets: 8/10 mini scenes done (scene 4 finished). Scene 7: rescue 2 (sharp_shape_reg w=0) also NaN — the NaN is 0/0 in the scale ratio: the car is parked, the lidar has exact duplicate points, kNN distance 0 -> log(0) = -inf initial scale. Rescue 3 (root cause): patched drivestudio models/gaussians/vanilla.py + models/nodes/rigid.py to clamp the initial kNN distance at 1e-4 m; rerun 3749884.
- Queued trainval splat pipeline: sky masks x4 (3749885-88, after preprocessing) -> 70 scenes (idx 0-69) at 15k iters on 2080 Ti, project tv -> data/targets_splat_tv.

## 2026-11-25 01:15
- Splat scene 7 (night, scene-1077): rescue 1 (seed=1) also NaN at step 0 -> rescue 2: sharp_shape_reg off for Background + RigidNodes (3749868).
- Encode 762/850; trainval drivestudio preprocessing (2 jobs) running; splat scene 4 still training.

## 2026-11-25 00:55
- Stage 8 eval of full-train-a (30 val scenes, 4 steps): masked L1 28.1 (mean-image baseline 30.6), LPIPS 0.41 (baseline 0.65). No drift for a single-frame model (first-5 vs last-5 frames 29.7 / 29.7).
- Stage 6: 30k-iter Street-Gaussian targets for 7 mini scenes are far better than lidar targets (pulled/splat_vs_lidar.jpg): complete scenes, parked cars, excavator, lane paint, facades, no holes; streaks at the frame edges, night scene noisier.
  scene 4 still training; scene 7 (scene-1077, night) NaN at step 0 in Background sharp_shape_reg -> rescue 1: seed=1 (3749832). ds_scene.sh now takes extra overrides.
- Splats cost ~2 GPU-h/scene at 30k, so not for all 850. Plan: splat ~70 trainval scenes (idx 0-69, incl. some val scenes) at fewer iterations, then fine-tune the full lidar-trained model on splat targets. drivestudio trainval preprocessing jobs 3749841/3749842 (idx 0-34, 35-69). ds_scene.sh takes DSROOT/NUSC_ROOT/NUSC_VER env vars for trainval.
- Encode 585/850; full-train-b/c queued after it.

## 2026-11-25 00:20
- Targets 850/850 done. Encode round 2 running (400/850 encoded). full-train-b/c queued after it.
- 8 splat scenes training (~1 h in).
- Stage 8 script: scripts/evaluate.py (val scenes; masked L1, LPIPS on covered pixels, error per frame for drift, mean-image baseline; video models roll out the whole scene). LPIPS weights cached. eval-full-a job 3749819 (30 val scenes, 4 steps).

## 2026-11-24 23:45
- Targets 844/850 (7/8 shards done); encode round 2 waits on the last shard; full-train-b/c wait on encode.
- 9/10 splat scenes training (30k iters), 1 done or queued; no real errors in logs.

## 2026-11-24 23:15
- full-train-a DONE (340 scenes, 45M, 40k steps, final loss ~0.3). Held-out val scenes masked L1: 0003 28.7, 0012 26.0, 0013 27.3, 0014 28.0, 0015 28.5, 0016 29.3.
  Pictures (pulled/full_a_sheet.jpg): real road structure on unseen streets (lane lines, kerbs, trees roughly right), still blurry. Big step up from mini (which drew a generic blur on unseen scenes).
- Stage 6 test (scene 0, 2k iters, ~6 min train + ~18 min load/render on 2080 Ti): renders work and show things the lidar targets miss (trucks, excavator), but with colourful floaters (sky, near camera) — expected at 2k iters and with a viewpoint 7 m above the training cameras (pulled/splat_test_sheet.jpg).
  Launched all 10 mini scenes at 30k iters (3749741-50).
- Targets 757/850 scenes. Queued after encode round 2: full-train-b (512x16, ~110M, 80k steps) and full-train-c (same + --prev video) -> runs/full_b, runs/full_c.

## 2026-11-24 22:45
- Download COMPLETE (21/21). Targets: 570 scenes so far (round 2 running); encode round 2 pending on them.
- full-train-a at 38k/40k (loss ~0.29); rendering of val scenes next.
- Stage 6: sky masks done (mini, 10 scenes); ds-test0 (2k iters, scene 0, 2080 Ti) started.

## 2026-11-24 22:15
- full-train-a step 28.4k/40k (loss 0.34), ETA ~22:50 + render.
- Download 20/21 (only trainval10 camera left, streaming). full_round.sh round 2 submitted for parts 04-09 (targets skip the 340 done scenes; encode skips existing).
- Stage 6: drivestudio preprocessing of mini DONE (images, lidar, calib, masks, objects, 10 Hz). Sky masks running (~4-5 min/scene, 1206 images each). Then the 2k-iter test on scene 0.

## 2026-11-24 21:50
- full-train-a step 18.4k/40k (loss 0.46), ETA ~23:15. Download 18/21. Kerberos renewed (valid to 07:37).
- Stage 6 code: scripts/ds_render_chase.py (loads a drivestudio checkpoint, builds our chase camera from the CAM_FRONT pose + nuScenes cam->ego calibration, renders RGB + opacity at every keyframe -> data/targets_splat_<proj>/<scene>/<k>_<token>.png); slurm/ds_scene.sh (train streetgs.yaml on one mini scene, load_smpl False, then render).
- Chain: preprocessing 3749415 (was stuck on QOSMaxCpuPerJobLimit with 16 CPUs -> scontrol to 8) -> sky masks 3749506 -> 2k-iteration splat test on scene 0 (3749507, 2080 Ti) before launching all 10 scenes at 30k.

## 2026-11-24 21:15
- full-train-a: step 8.6k/40k after 27 min (loss ~0.8, not memorising like mini's 0.03); ETA ~23:15.
- Download 15/21.
- drivestudio env DONE (DS_ENV_OK, torch 2.1.2 cu118, gsplat 1.3.0). nvdiffrast only built for sm 7.5/8.6 (sm_90 ptxas crash) -> splat jobs must request --gres=gpu:nvidia_geforce_rtx_2080_ti:1 or gpu:nvidia_rtx_a6000:1, not H200 slices.
- drivestudio mini preprocessing 3749415 still queued (16-CPU job).

## 2026-11-24 20:50
- full-train-a running (since 20:41, H200 MIG slice, bf16) on 340 encoded scenes.
- Download 12/21.
- drivestudio env: gsplat 1.3.0 + pytorch3d built with conda gcc 11. nvdiffrast: ptxas segfault with 8 parallel jobs -> slurm/ds_finish.sh (MAX_JOBS=1, falls back to sm 7.5/8.6 only) job 3749414. drivestudio imports nvdiffrast in models/modules.py, so it's needed.
- drivestudio preprocessing of mini (10 Hz, interpolate_N 4) job 3749415 (slurm/ds_preprocess.sh). Plan: sky masks via scripts/sky_masks.py, then train configs/paper_legacy/streetgs.yaml with dataset=nuscenes/6cams, load_smpl=False, one job per scene; then write a chase-cam renderer over the trained splats.

## 2026-11-24 20:15
- Download 10/21 archives (parts 01-05 partly); targets done for 340 scenes; encode 4 shards running (144/340 after 23 min).
- drivestudio env: gsplat compile failed (system gcc 13 > CUDA 11.8 limit 11) -> conda-forge gcc/gxx 11 in the env, CC/CXX set; rebuild job 3749366.
- Queued full-train-a (3749367, after encode): data/enc_full (pooled tokens), official val held out, 45M DiT, 40k steps, bs 64 -> runs/full_a. First scaling point vs mini.

## 2026-11-24 19:45
- Stage 7 baseline (mini stage-4 model, H200 MIG 1g slice, batch 1 incl. VAE decode), masked L1 / fps:
  train scene-0061: 1 step 15.9 | 2 steps 1.5 / 20 fps | 4: 1.5 / 11 | 8: 2.0 / 5.8 | 16: 2.8 / 2.9 | 32: 3.4 / 1.5
  held-out scene-1077: ~20 at every step count (1 step 19.5 / 33 fps).
  -> 2 steps already matches or beats 32 here; ~20 fps at 2 steps before any distillation.
- Full targets: 8 shards running ~45 s/scene, ~290 scenes done (parts 01-03), coverage 0.5-0.83. Encode shards wait on them.
- Download 7/21 archives done (parts 01-03 + meta); 04/05 streaming.
- drivestudio env build still installing pip deps (30 min in).
- Next: when encode lands, train on what's there (val held out) for a first scaling point; full train when download completes.

## 2026-11-24 19:15
- Download: meta + parts 01, 02 (camera + lidar) done (~170 scenes); 03/04 streaming.
- full_round.sh submitted: 8 target shards (3749275-82) -> 4 encode shards (3749283-86, after targets). Rerun after more parts land.
- Fixes: node /tmp full (grep write error in env build; also earlier /tmp failure) -> TMPDIR=~/mini-reality-view/tmp in py.sbatch + setup_drivestudio.sh; env build resubmitted 3749273.
  steps_speed OOM (decoded a whole scene's GT at once on a shared 16 GB slice) -> chunked; resubmitted 3749274.
- Queue is busy (other project's jobs), all ours pending on Priority.

## 2026-11-24 18:46
- Download: no part finished since 18:23 (3 parallel streams still running). drivestudio env job 3749202 running (conda done, pip/compiles next).
- GPU was idle -> stage 7 baseline submitted: scripts/steps_speed.py on the mini stage-4 model (steps 1..32, masked L1 + fps at batch 1) job 3749229.

## 2026-11-24 18:40
- Training-scene render check: masked L1 3.2 (scene-0061) / 3.1 (scene-0916) vs 20-56 on held-out -> model uses the cameras; it just needs more data.
- slurm/full_round.sh: 8 CPU target shards + 4 GPU encode shards (pool 2) on data/nuscenes_full -> data/targets_full, data/enc_full. Run it when download parts land (idempotent).
- Stage 6 prep: drivestudio env build (slurm/setup_drivestudio.sh, conda-forge + nvidia channels only; torch 2.1.2 cu118, gsplat 1.3.0, pytorch3d, nvdiffrast) job 3749202. First try failed on Anaconda ToS (not accepted on the user's behalf).
  scripts/sky_masks.py replaces mmcv SegFormer with transformers SegFormer-B5 Cityscapes (weights cached in hf/).
- Queue: another project's ~12 GPU jobs are queued/running on Teaching, so ours may wait.
- Download: 3 parallel streams since 18:23, part01 camera ~6.7k front images in.

## 2026-11-24 18:30
- Stages 4+5 on mini (8 train scenes / 322 frames, 15k steps, 45M DiT, ~25 min each on an H200 MIG slice): train loss 1.8 -> 0.03 (memorised).
  Held-out masked L1 (0-255): scene-1077 20.4 (s4) / 19.8 (s5); scene-0553 53.4 / 56.3. Pictures (pulled/stage45_sheet.jpg): 1077 gets the rough layout (dark road, lit centre), 0553 a generic grey blur. Both test scenes are night scenes. Expected with 8 scenes; fix = full data.
- Sanity check submitted: render training scenes 0061/0916 (job 3749183) to confirm the model uses the cameras.
- Full download switched to streaming curl|tar, 3 parts in parallel (unpacking was the bottleneck). Done so far: meta, trainval01 lidar.
- Code for full data: make_targets skips undownloaded/finished scenes; encode --pool 2 stores 16x9 tokens (0.66 MB/frame); train --test_scenes val uses the official val split.

## 2026-11-24 18:00
- Stage 1 done: runs/stage1/cameras.jpg + bev.jpg look right (6 cams, lidar, camera cones).
- Stage 2 v1 (±3 keyframes, 0.08 m points): sparse, roughly 40% covered. v2 (static points from the whole scene, moving objects from current frame only, 0.12 m points): coverage 0.49-0.82 per scene (night/parked scene-0553 only 0.18). v2 is clearly better (pulled/compare_v1_v2.jpg) -> make_targets defaults now K=100, radius 0.12.
- Cancelled v1 training; re-encode v2 -> data/enc_v2 (job 3749135), then stage4 3749136 + stage5 3749137 train on it.
- Full download: meta done, trainval01 lidar in progress.

## 2026-11-24 17:45
- Env OK (torch 2.15 cu132 got pulled in by pip; works). Account = Teaching partition only; GPU jobs land on 2080 Ti / A6000 / H200 MIG slices. train.py uses fp16+GradScaler when no bf16.
- GPU check: H200 MIG 1g.18gb, CUDA OK. stage1_look first run crashed (np.Inf in old matplotlib) -> patched, rerun 3749120.
- Stage 2 targets job 3749113 running (~40 s/scene). Chained: encode 3749121 -> train stage4 3749122 + stage5 (--prev) 3749123, 15k steps each.
- Full trainval download started on login node (slurm/download_full.sh, logs/download-full.log, resumable via data/full_done.txt). No disk quota on home.

## 2026-11-24 ~17:15
- Code written: mrv/nusc.py, mrv/model.py (45M DiT), scripts/stage1_look.py, make_targets.py, encode.py, train.py; slurm/setup_env.sh, py.sbatch.
- Cluster ~/mini-reality-view: nuScenes-mini downloading/unpacking to data/nuscenes; .venv installing (logs/setup_env.log).
- Next: stage 1 look (runs/stage1), stage 2 targets on mini (data/targets_lidar), stage 3 encode (GPU), stage 4 train (test scenes scene-0553, scene-1077), stage 5 --prev.
