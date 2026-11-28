# vpriv final signature request (28 Nov ~00:30 UTC) — Astra (code + A3 compliance)
Read-only. Write council/codereview-vpriv-final-astra.md (<= 25 lines), ending "SIGNED: yes | no | if <conditions>".

Context: council/AGREED-causal-r1.md (A1, A2, A3 CONVERGED + init note), your council/codereview-vpriv-astra.md (SIGN-IF).

## 1. A3 trigger evidence (signed tools 764dd3ec95db / fc7ee2b886a5; raw files in pulled/s_lock/ once synced)
- V8k vs A@20k: ID +51.1%, W +21.6%; F1 +0.0024; late +0.004 -> no pass, safety met.
- V12k vs A@20k: ID +48.0%, W +19.7%; F1 +0.0215; late +0.002 -> no pass, safety met.
- V16k vs A@20k: ID 0.1560/0.0989 rel +0.577 (both can_pass True), W 33.99/29.57 rel +0.149; car F1 diff -0.0105 (CI -0.035..0.017);
  late gap diff +0.0016 -> no pass, BOTH safety bars met.
- V16k vs own cold (Aoff_rho0.95): ID diff +0.0453, W diff +2.87 -> not strictly lower -> mixture = BOTH clips private (private_clips 2).
- LODO @16k vs A: ID +52.9..+64.7%, W +10.2..+19.5%, F1 -0.021..-0.003, late -0.0002..+0.0040.

## 2. Your SIGN-IF items
- (1) matched init: replaced by the accepted seeded init (A3 init note; vprivA 3755758: init 9f21e926a01825a8 identical in 3 processes).
- (2) resume fixture, (3) shared slot bit-identical to r1, (4) extra fixtures: vprivA 3755758 + vprivB 3755759 FAILS=0 on ad1b0aa
  (see PROGRESS.md 23:49 entry); T2 3755760 FAILS=0.
- Suite T1-T7 on deployed hashes: run 3755757 had T5 fail ONLY in the salt-only arm, from tick 0 with an empty history and equal
  noise keys / sources -> cross-process GPU nondeterminism. Fix 067cd12 (diff: `git show 067cd12`): scripts/det_run.py sets cudnn
  deterministic, benchmark off, TF32 off, then runpy-runs causal_rollout.py unchanged (its fingerprint binds the rho decision);
  slurm/causal_tests.sh exports CUBLAS_WORKSPACE_CONFIG=:4096:8 and routes T5 / T6 rollouts through it; det_run.py added to the
  sha256 manifest. Re-run 3755915 in progress (noise + rho fixtures PASS so far). Question: is running T5 / T6 under deterministic
  kernels an acceptable test of the causal property (training and production rollouts keep default kernels)?
- No change to train_causal.py / mrv/noise.py / causal_rollout.py / causal_video.py since ad1b0aa.

## 3. Launcher to sign: slurm/launch_vpriv.sh (not run yet)
Gates: suite / T2 / vprivA / vprivB logs FAILS=0; sha256 -c of the suite AND both vpriv fixture manifests (code unchanged since
tested); A3 trigger recomputed from the signed cmp files (no joint pass at 8k / 12k / 16k; 16k safety met; mixture rule);
rho decision still valid and == 0.95; one job on one saxa h200_3g slice; `train_causal.py --init runs/longA/ckpt_20000.pt
--out runs/vpriv --K 4 --rho 0.95 --rho_decision council/rho_decision.json --private_clips 2 --steps 16000` (default seed =
vlive's). It will be submitted with -d afterok:<suite>. Checks at 2k / 4k (informational) and 8k / 12k / 16k exactly as vlive,
plus the two-noise probe (scripts/noise_probe.py --noise ckpt / private) at each check.
Check especially: the trigger arithmetic (smaller reduction = min(-rel_ID, -rel_W), bar > 0.094; eligibility via can_pass),
anything in A3 the launcher omits, and whether the det_run fix weakens T5 / T6.
