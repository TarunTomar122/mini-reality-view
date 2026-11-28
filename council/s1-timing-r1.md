# S1 timing round 1 — the 12k floor is not reachable as measured (28 Nov 10:22 UTC)
Pre-registered (PROGRESS 09:36): if the forecast says the 12k batch-8 floor is not reachable inside 24 slice-h after gates and
panels, no production start; the measured bottleneck goes to council, Fable's default candidate (2x2 token pooling + re-time)
first, restating the cap to the owner only second. Raw: runs/wan_timing_3756964/timing.json on the cluster (local copy
pulled/timing/timing.json is not committed); code: scripts/train_wan_teacher.py (load_split / clip / make_batch / start_producer),
scripts/wan_forecast.py, mrv/wan_plan.py (forecast_s1, panel_cost_h), scripts/wan_panel.py, slurm/wan_teacher.sh.

## Measured (H200 MIG 3g.71gb slice, batch 8, 48 S0c train drives)
- Gates PASS: token geometry (10.7 px vs 45 px wrong-camera control), init parity (stock vs wrapped 0.0; guide / tokens inert
  0.0), update-0 sensitivity (guide 0.16, tokens 0.005; control 0).
- GPU work per update: 3.63 s at micro 1 (peak 5.6 GB) ... 4.00 s at micro 8 (17.7 GB) — compute does not drop with micro.
- Production loop (8 updates, cold start): wall 10.69 s / update; producer assembly 10.67 s; queue wait 7.07 s -> GPU idle ~2/3.
  One Python thread builds each batch: per clip it fancy-indexes 5 ticks of the drive's token cache [T, 6, 777, 384] f16 + rays
  from an mmap'd torch.load on Lustre (~143 MB of tokens per batch); token file 0.40 GB / drive (60 GB for 150 drives), latent
  cache 16 MB / drive. saxa has ~1 TB free in /dev/shm (2 TB RAM); /tmp is full; jobs get 8 CPUs.
- Start-up 1,146 s for 48 drives (--verify_cache sha256 of cache + token + guide + enc files on Lustre, then Wan load), scaled
  by the forecast to 3,437 s for 144 drives — paid at every job start / resume (12 h jobs).
- Panel 13.9 slice-h each (forecast panel_cost_h): rollout 0.70 s / tick + 381 s start per rollout (model + data load);
  RGB scoring (signed scorer through score_rgb.py) 2.41 s / tick and dominates; dev 1,046 ticks x 2 seeds + swap renders.
- Forecast: overfit 7.5 h + resume test 3.0 h + two panels 27.9 h -> available -15.1 h of 24; floor_12k_reachable false.
  Ledger so far 0.71 slice-h (timing).
- Pending evidence: CPU profile prof-prod 3757040 (scripts/prof_producer.py; cold vs warm page cache, 4 / 8 reader threads,
  sequential MB/s) -> logs2/prof-prod-3757040.out on the cluster (read-only ssh allowed: `ssh mlp '...' < /dev/null`, <= 3 calls).

## Builder's candidate fixes (none applied; trainer edits are allowed now — the gate-key freeze starts at MODE=overfit)
F1 data path: stage the train drives' token + latent caches to /dev/shm on the node at job start (sequential copy, verified
   once), and/or N parallel producer workers; target: GPU-bound ~4 s / update -> 12k in ~13.3 h.
F2 start-up: verify sha256 once in a CPU job and write a verified manifest (path, size, mtime, sha); production checks the
   manifest (size / mtime) and samples a few hashes instead of re-hashing ~60 GB each start.
F3 panel: rollouts stay on the slice (model loaded ONCE per panel, not per rollout); RGB scoring moves to parallel 2080 Ti jobs
   (not slice time), i.e. the panel's slice cost ~1-2 h. This changes where signed panel hours are counted -> needs your sign.
F4 (Fable's pre-registered default) 2x2 pooling of raw-view tokens: 4x less token I/O and cross-attention compute; a recipe
   change (re-time, new gate key). Only worth it if F1 cannot make the loop GPU-bound.
Then re-time (MODE=timing again, ~1 slice-h) before overfit.

## Question (each seat <= 25 lines)
Astra -> council/s1-timing-r1-astra.md, Fable -> council/s1-timing-r1-fable.md: accept / reject / amend F1-F4 (and anything I
missed), in the order to apply; whether F3 (scoring off the slice) is acceptable under the signed S1 budget or the panel must
shrink instead (e.g. fewer seeds / placement ticks only — note the signed bars); the exact re-time pass rule before overfit;
and whether the 24 h cap needs restating to the owner. End with "PLAN: <ordered list>". Read-only: no code edits, no jobs, no commits.
