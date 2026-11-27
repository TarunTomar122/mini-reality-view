# Claims-vs-code audit (Mini Reality View; owner request 27 Nov ~18:30)

Context: a public report (x.com/RealAstropulse, 27 Nov 2026) says a Claude agent, doing ML research for a user, wrote one plan
but coded something else: starved training of high-resolution samples, froze a component the user said not to freeze, and
slowed training by recomputing things every step. The owner wants an independent check that nothing like that happens here.
Guard adopted from ~/Documents/neurellum/GUARD-ml-research.md (shared by the walking-RL / live-VSR session).

Audit every file written or changed since the last audit (git log -p since <stamp>; at least scripts/train_video.py,
mrv/video.py, scripts/video_rollout.py, scripts/train_temporal.py, scripts/temporal_rollout.py, slurm/*.sh) against what the
builder claimed in PROGRESS.md and the agreed design (council/AGREED-*.md). Cluster read-only: `ssh mlp '...' < /dev/null`
(<= 5 connections): runs/*/log.jsonl, runs/*/args.json, logs/*.out, runs/troll/m_*.json, runs/s_lock/cmp_*.json.

For each file answer: does the code do what the builder said? Look for:
1. Data: training / eval data silently skipped, starved, re-weighted, truncated or mismatched between paired arms (clip
   selection, drives, ticks, hard breaks, camera dropout, dev drives leaking into training, the 30-drive exam split kept sealed).
2. Frozen / disabled parts: anything frozen, zeroed, bypassed or switched off that the agreed design did not call for (and
   anything trained that should be frozen, e.g. the A@20k backbone).
3. Waste: needless recomputation, repeated forward passes, uncached work, per-step disk reads, slow paths; compare the logged
   step_s / data_s with what the design expects.
4. Plan-vs-code mismatches: window K, masks (strict causality), noise sampling, loss, CFG, DDIM steps, seeds, pairing,
   thresholds and gates.
5. Results misreported: recompute any number you can from the JSON / logs and compare with PROGRESS.md and what the owner
   was told.
Rank findings by how much they could bias a decision or waste GPU time; file:line, what it does, what it should do. Say
clearly when a file is fine. Write council/hourly/<HHMM>-audit-<seat>.md (<= 60 lines). Read-only: no jobs, no edits.
