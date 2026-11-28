# Astra T3 · cc2b640 · hold the 144-drive timing submission
Python 3.14.7 locally returned `WAN_TESTS FAILS=0` with six trainer-import skips. The 3.11 pass, verified manifest (1,057 files, 132 GB, a0412b42f4cc), 170G request and fresh parity submissions are cluster reports. Parity PASS is pending.

- **8 OPEN — timing.** CPU repro through `sample_once` and `memory_final`: one valid read of 100 at t=100, rehearsal ends at 200, both reads fail at 300. With a request of 170, the job records `includes_rehearsal=True, pass=True`. Timestamp the last successful job-level read and require it after rehearsal. Retain the allocation-start coverage check. See `mrv/wan_stage.py:432` and `mrv/wan_plan.py:1031`.
- **Neon C OPEN — timing forecast.** The segment cache works. The rehearsal averages one miss and one hit, then prices eight ordinary renders per drive at that rate. Each segment actually incurs one miss and seven hits. With a 20 s encode and 100 s other work per pass, the ordinary renders cost 820 s and the forecast charges 880 s. Price cache fills separately from repeat work, including distinct swap/reference sequences. See `scripts/wan_panel.py:154,182` and `mrv/wan_plan.py:634`.
- **3 CLOSED.** The forecast checks the exact 144 train and six dev IDs before opening encodings and prices the canonical dev drives.
- **4 CLOSED.** The trainer rejects a different sweep before running it. Selection excludes extra cells and the recorded grid must match.
- **5 CLOSED.** The forecast preserves full-precision panel hours and requires both original allocation 3756964 and the distinct re-time job.
- **7 CLOSED.** Timing runs the production parity-binding check and includes its measured I/O in startup.
- **9 CLOSED.** The trainer rejects a forwarded manifest override and the geometry digest must equal the trainer’s digest.
- **11 CLOSED.** Refresh checks the timing ledger path and both timing charges. The ten-file evidence hash includes `mrv/guide.py`.
- **Rehearsal inputs CLOSED.** Both rollout paths hash the exact guide, encoding, token and sidecar bytes they parse against the trainer’s manifest. For comparable panel I/O, production render should pass the same manifest arguments. Today only rehearsal uses the in-memory verified loader (`scripts/wan_rollout.py:185–211`).
- **12 OPEN — panel scoring only.** Fresh parity requires resolved weights and binding before scoring. In a CPU repro `score_binding_matches` still accepted a panel shard explicitly marked `bound_before_scoring=False`. Require that flag in panel intake as parity intake already does (`mrv/wan_plan.py:570,587`).
- **14 CLOSED.** A fresh one-hour reservation after 30 minutes of setup holds the whole hour. Publication reconciles accepted receipt job IDs against finished charges and the cap.
- **15 CLOSED.** Render pins the checkpoint and donor-plan bytes it reads. Decision verifies the render digests and loads the checked checkpoint bytes.
- **Neon A CLOSED.** The staged-byte formula with 45 GB RSS and 80% occupancy yields 170G. Each copy advises eviction on its source descriptor. Actual peak remains a timing measurement.
- **Neon B CLOSED.** Reservations and receipt reconciliation match either ledger prefix by Slurm job ID. New scoring heartbeats use `alloc-`.

SIGNED (timing / overfit / production): no | no | no
