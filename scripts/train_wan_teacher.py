"""PLAN-video S1: bidirectional Wan2.1-T2V-1.3B teacher, guide channels + camera tokens (mrv/wan_cond.py), council/PLAN-video.md;
revised for Astra's review council/codereview-wan-s1-astra.md (items 3-8, 11, 12) and the underlay guide (415c470).

Recipe
- data: 144 train drives (data/temporal_lists.json "train"; train / dev overlap asserted; sealed splits never read). Clips of 5
  latent frames at 512x256: with prob --p_start (0.15) a TRUE-START 17-tick clip, else a MID-STREAM clip (5 regular latents
  j0..j0+4, j0 >= 1, of a whole-segment stream encoded in 4 phases; 20 ticks). Streams run through soft rejects; every tick of
  a TRAINING clip is ok (mrv/wan_plan.py). Candidate counts per drive are printed and stored in args.json before sampling.
  Targets = Wan-VAE latents of the keys; guides encoded identically + source planes (depth / ground share, 2 x 4 folded);
  camera tokens + per-tick rays memory-mapped from the job-local stage dir (scripts/wan_tokens.py); every staged byte was hashed
  against the manifest while copied, and all caches carry ONE guide version (train and dev).
- objective: rectified flow (x_s = (1 - s) x0 + s eps, v = eps - x0, timestep 1000 s), MSE fp32; one sigma per clip,
  shifted-uniform with shift --shift 2.0 (Wan 480p uses 3.0 for 32.8k tokens; the SD3 sqrt(token) rule gives 1.7 per frame /
  0.84 per clip for our 2,560; kept above the rule because car placement is decided at high noise; median sigma 0.67).
- JOINT input dropout (Fable 07:30): one draw per clip -> both dropped --p_drop_both 0.10 (the unconditional branch CFG / S2
  need; independent 20 % / 10 % gave only 2 %), only the guide dropped --p_drop_guide_only 0.10, only the tokens dropped
  --p_drop_tokens_only 0.10 (marginals guide 20 %, tokens 20 %; recorded in args.json "dropout"); a kept guide gets holes w.p. 0.5. Updates 1 and 2 (absolute steps 0 / 1) use
  NO dropout / holes, so the first-update guard and the gate-open gradient check are deterministic.
- signals: SIGTERM / SIGUSR1 (slurm/wan_teacher.sh asks Slurm for USR1 300 s before the time limit and forwards it) -> atomic
  rolling checkpoint at the next update boundary, clean exit (WAN_TEACHER_SIGNAL_STOP), resume with --resume.
- trainable: LoRA r32 on the 10 targets x 30 blocks (asserted) + extra patch conv + projector + gates, fp32, seeded init; base
  bf16 + VAE frozen. AdamW 1e-4, no weight decay, 500-update warm-up, clip 1.0, bf16 autocast, per-block checkpointing,
  global batch 8 (--micro < 8 = accumulation).
- init diagnostics (item 4): stock parity with the same 512-row padded context and precision, 1-row shortcut, guide / token
  inertness, explicit finiteness (mrv.wan_cond.build(parity=True), check_parity).
- guard (item 5): after update 1 (fresh run): base transformer + VAE bit-identical; extra conv, all 30 gates (tokens on) and
  LoRA-B of 8 targets x 30 blocks moved (attn2 q / k LoRA cannot move yet: one text row, gate 0). Update 2: projector and
  attn2 q / k LoRA-B gradients non-zero in all 30 blocks (the gates are open).
- RNG: every update's draws from torch.Generator(seed * 100003 + step) (CPU) -> resumable by construction; the chained data hash
  pick_chain = sha1(prev | batch_sha) is saved and continued, so a resumed run must end on the uninterrupted run's chain.
- budget (item 11): ONE ledger (--ledger, jsonl heartbeats every 60 s) is debited by every S1 job: timing, overfit, resume test,
  training, panels (scripts/wan_panel.py); a job's charge = last beat - start, so preempted work is still debited. Before
  every update: stop (save) if used + this update + the panel reserve >= --budget_hours; the reserve is the MEASURED panel
  cost (timing.json forecast_final.panel_h_total: slice rendering only, Wan loaded once, incl. shared-mode references; scoring runs on the 2080 Ti) x --reserve_panels, never a fixed number. Verdicts: below 12k updates
  unresolved; 24 h read -> pass / one 12 h extension only for car F1 in [0.4156, 0.4456) with every other gate passed / stop;
  after the extension the bar stays 0.4456 (mrv.wan_plan.decide_s1, applied by scripts/wan_panel.py).
- saves: numbered ckpt_<k>.pt (--save_at) FIRST, then the rolling ckpt.pt every 500 updates; atomic; trainable weights,
  optimizer, step, RNG, pick chain, recipe, code / base / VAE / data / guide-version hashes.
- logs (log.jsonl): loss, grad norm, lr, sigma, step_s, data_s, wait_s, e2e, peak alloc / reserved GB, batch / chain / data sha,
  gate / extra norms, ledger hours; every --val_every a fixed dev probe (flow loss at sigma 0.3 / 0.6 / 0.9, inputs hybrid /
  guide-only / tokens-only / none / other-drive, + output sensitivity to guide and tokens).
Staging + trust (council/s1-timing-r1-CONVERGED.md items 2, 3, 5; mrv/wan_stage.py)
- --manifest (required in every mode): the read-only manifest of scripts/verify_manifest.py (CPU job: every cache / token /
  guide / enc file of train + dev hashed against its sidecar, + the Wan HF snapshot files). At start the token + latent caches
  (+ sidecars) of every drive this job opens and the Wan transformer / VAE folder are copied to --stage_dir (default
  /dev/shm/mrv_<SLURM_JOB_ID or pid>) with --stage_streams (8) parallel streams, hashing while copying; every hash must equal
  the manifest or the job refuses. Guide / enc files are never opened here: their size / mtime is checked against the manifest
  as a CHANGE ALARM only (mismatch -> refuse, "re-run verify_manifest"). Caches are memory-mapped from the stage dir. The
  manifest sha goes into the gate key and every run record beside DATA_SHA; the per-start re-hash (--verify_cache) is gone.
  Start-up (process start / allocation -> first update) and stage-in seconds / bytes are recorded (args.json "stage",
  log.jsonl "startup"). The cgroup limit must hold staged bytes + 48 GB (refused otherwise; slurm/wan_teacher.sh too).
- F5: --micro and --grad_ckpt are REQUIRED in every mode (slurm: MICRO / GRAD_CKPT), recorded in the recipe and the gate key;
  overfit / resume test / train / extend refuse unless they equal the newest timing.json "selected_setting" and its
  "retime_pass" (scripts/wan_forecast.py) passed every rule, and the timing ran with the same gate key.
Modes
- --timing (item 7 + CONVERGED item 7): sweep --timing_grad_ckpt {1, 0} x --timing_micro {1, 2, 4, 8} at global batch 8 (warmed
  real updates: wall / fwd / bwd / Adam / data / peak alloc + reserved; stops a grad_ckpt row at OOM); a MATCHED-UPDATE check
  (same params / inputs / noise / seed -> loss and grad norm within --match_rtol / --match_gn_rtol across settings) before the
  fastest matched fitting setting is selected; a --use_tokens 0 GPU-time reading (tokens dropped, diagnostic only); then
  --timing_prod_updates (>= 200) consecutive production updates from the staged data with the selected setting (per
  --timing_block (32)-update block: wall, gpu_s, queue wait, wait / wall, wall / gpu_s; forecast wall = slowest of the
  full-run mean and the last two block means); 8 batches built from the stage dir byte-equal to the same steps built from the
  original Lustre files; cgroup peak memory incl. staged files vs the request; probe (update-0 sensitivity on the selected
  setting), save, one 30-step rollout window (1 and 2 rows), Wan VAE encode (17 f) / decode (5 latents), SD-VAE encode ->
  timing.json (scripts/wan_forecast.py adds forecast_final + retime_pass).
- --overfit 8 (item 8): 8 DISTINCT fixed clips (6 mid-stream + 2 true starts) as every batch; fixed-noise / fixed-sigma eval of
  the SAME 8 every 100 updates; PASS = final eval loss <= --overfit_ratio (0.25) x the update-0 eval loss AND decoded samples
  (30 Euler steps, hybrid) reach car F1 >= --overfit_f1_frac (0.8) x the Wan-VAE round-trip car F1 against the keys (signed
  scorer's segmentation + 0.3 matcher), the round trip itself > 0; tokens-only and none samples of the same 8 clips / noise are
  reported as controls; pre-registered (Fable 07:30 / 08:30): FAIL if hybrid car F1 <= no-inputs car F1 + 0.02; a missed loss
  ratio with placement passing or a still-falling eval loss = NOT_REACHED (exit 6: rerun longer / lr 3e-4), allowed ONCE per gate
  key (code / model recipe / base / VAE / guide version), a second one = FAIL (exit 4); every verdict is appended to --gates and
  MODE=train refuses to start without PASS overfit + PASS resume-test records for its gate key; a stopped run exits 7 (PENDING); both gates read at update 2,000 only (--steps 2000 required;
  a shorter job prints OVERFIT_PENDING and resumes); the update-0 eval loss is stored in every checkpoint; sampling runs in the
  fitting microbatch -> overfit.json, exit 4 on FAIL.
  python scripts/train_wan_teacher.py --out runs/wan_s1 --steps 40000 --budget_hours 24 --ledger runs/wan_s1_ledger.jsonl --resume \
         --manifest data/wan_manifest_v1.json --micro <timing selection> --grad_ckpt <timing selection>
"""
import argparse, atexit, glob, hashlib, json, os, queue, random, shutil, threading, time
T_PROC0 = time.time()  # process start (start-up is measured from here and, when slurm exports MRV_ALLOC_T0, from the allocation)
import numpy as np
import torch
from mrv import wan_cond as WC
from mrv import wan_plan as WP

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--cache", default="data/wan_cache_v1")
p.add_argument("--tokens", default="data/wan_tok_v1")
p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--guides", default="data/guides_v1")
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--wan", default=WC.WAN_ID)
p.add_argument("--prompt_embed", default="")
p.add_argument("--rank", type=int, default=32)
p.add_argument("--tok_ticks", default="last", choices=list(WC.SLOTS))
p.add_argument("--use_tokens", type=int, default=1)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--micro", type=int, default=None, help="REQUIRED (F5): microbatch; global batch stays --batch (accumulation)")
p.add_argument("--steps", type=int, default=40000)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--shift", type=float, default=2.0)
p.add_argument("--sigma_dist", default="shifted_uniform", choices=["shifted_uniform", "logit_normal"])
p.add_argument("--p_start", type=float, default=0.15)
p.add_argument("--p_drop_both", type=float, default=0.10, help="joint dropout share: BOTH inputs dropped (unconditional branch for CFG / S2)")
p.add_argument("--p_drop_guide_only", type=float, default=0.10, help="joint dropout share: ONLY the guide dropped (tokens kept)")
p.add_argument("--p_drop_tokens_only", type=float, default=0.10, help="joint dropout share: ONLY the tokens dropped (guide kept)")
p.add_argument("--gates", default="runs/wan_gates/gates.jsonl", help="overfit / resume-test verdict registry (MODE=train needs PASS records)")
p.add_argument("--p_holes", type=float, default=0.5)
p.add_argument("--save_every", type=int, default=500)
p.add_argument("--save_at", default="1000,2000,4000,6000,8000,10000,12000,14000,16000,20000,24000,28000,32000,36000,40000")
p.add_argument("--val_every", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--resume", action="store_true")
p.add_argument("--max_steps_this_job", type=int, default=0)
p.add_argument("--budget_hours", type=float, default=24.0, help="S1 slice-hour cap over ALL S1 jobs in --ledger")
p.add_argument("--reserve_hours", type=float, default=None, help="override; default = measured panel cost (timing.json forecast_final."
               "panel_h_total, incl. shared-mode references) x --reserve_panels (Astra S1c item 4)")
p.add_argument("--reserve_panels", type=int, default=1, help="panels still to run inside this budget (train: the 24 h read; extend: the final); >= 1 in production")
p.add_argument("--panel_window_latents", type=int, default=5, help="rollout window size the S1 panels will use (the reserve must be priced for it)")
p.add_argument("--panel_swap_at", type=int, default=50, help="swap position the S1 panels will use (the reserve must be priced for it)")
p.add_argument("--timing_json", default="", help="default: newest runs/wan_timing_*/timing.json")
p.add_argument("--a_metrics", default="runs/troll/m_A20k_indep_dev.json")
p.add_argument("--a_roll", default="runs/troll/A20k_indep_dev")
p.add_argument("--parity_record", default="runs/wroll/score_parity_2080.json", help="MODE=train needs this bound 2080 Ti scorer-parity PASS "
               "(A@20k replay + sharded == unsharded; scripts/wan_score.py --parity_check)")
p.add_argument("--ledger", default="runs/wan_s1_ledger.jsonl")
p.add_argument("--kind", default="", help="ledger label (default: timing / overfit / train from the flags)")
p.add_argument("--overfit", type=int, default=0)
p.add_argument("--overfit_ratio", type=float, default=0.25)
p.add_argument("--overfit_f1_frac", type=float, default=0.8)
p.add_argument("--timing_micro", default="1,2,4,8")
p.add_argument("--timing_grad_ckpt", default="1,0", help="F5 sweep: gradient checkpointing on / off")
p.add_argument("--timing_prod_updates", type=int, default=200, help="consecutive production updates timed (re-time rule: >= 200)")
p.add_argument("--timing_block", type=int, default=32, help="production-loop block size for the per-block rules / forecast")
p.add_argument("--match_rtol", type=float, default=0.02, help="matched-update check: loss within this relative tolerance (bf16)")
p.add_argument("--match_gn_rtol", type=float, default=0.05, help="matched-update check: grad norm within this relative tolerance (bf16)")
p.add_argument("--lustre_check_batches", type=int, default=8, help="timing: batches built from the stage dir compared byte-for-byte with Lustre")
p.add_argument("--grad_ckpt", type=int, default=None, choices=[0, 1], help="REQUIRED (F5): per-block gradient checkpointing")
p.add_argument("--manifest", default="", help="REQUIRED: read-only data manifest of scripts/verify_manifest.py (F2)")
p.add_argument("--stage_dir", default="", help="job-local stage dir (default /dev/shm/mrv_<SLURM_JOB_ID or pid>)")
p.add_argument("--stage_streams", type=int, default=8, help="parallel copy + hash streams for the stage-in")
p.add_argument("--keep_stage", type=int, default=0, help="1: leave the stage dir for the next process of this allocation (slurm removes it in its EXIT trap)")
p.add_argument("--timing", action="store_true", help="timing / test run: dir must contain 'timing' or 'ctest'")
p.add_argument("--resume_test", action="store_true", help="the 16-update resume test (bounded exemption: <= 16 updates, *ctest* dir)")
p.add_argument("--production", action="store_true", help="set by slurm/wan_teacher.sh MODE=train / extend after any forwarded args")
a = p.parse_args()
MODE_ = WP.check_modes(a.timing, a.overfit, a.resume_test, a.production)  # Astra t1 item 10: exactly one run mode
TEST = a.timing or a.overfit
if TEST and not any(s in a.out for s in ("timing", "ctest")):
    raise SystemExit("--timing / --overfit runs must write to a *timing* / *ctest* directory")
if a.overfit and a.batch != a.overfit:
    raise SystemExit("--overfit N trains on exactly N clips per update: --batch N")
OVERFIT_STEPS = 2000  # declared: both overfit gates are read at update 2,000 (S1b item 6)
if a.overfit and a.steps != OVERFIT_STEPS:
    raise SystemExit(f"--overfit is declared at {OVERFIT_STEPS} updates: --steps {OVERFIT_STEPS}")
if a.overfit:  # Astra t1 item 10: only the signed gate (8 clips, 2,000 updates, ratio 0.25, F1 fraction 0.8) can register a verdict
    _ok_o, _why_o = WP.overfit_signed_ok(a.overfit, a.steps, a.overfit_ratio, a.overfit_f1_frac)
    if not _ok_o:
        raise SystemExit(_why_o)
os.makedirs(a.out, exist_ok=True)
dev, AMP = "cuda", torch.bfloat16
KIND = a.kind or ("timing" if a.timing and not a.overfit else ("overfit" if a.overfit else "train"))
# Astra S1f / S1g 4a: production enforcement (forecast reserve, budget stops, gates) follows the RUN MODE, never the --kind label
# or the output directory name: every run is production unless it is one of the three bounded test modes (timing: timed sweep
# then exit; overfit: exactly 2,000 updates on N clips; resume test: <= 16 updates); --production (MODE=train / extend) refuses
# any test flag, so a forwarded flag cannot turn a production job into an exempt one
if a.resume_test and (a.steps > 16 or "ctest" not in a.out):
    raise SystemExit("--resume_test is the bounded 16-update test: --steps <= 16, a *ctest* dir")
PROD = not (a.timing or a.overfit or a.resume_test)
# F5 (CONVERGED item 5): the microbatch and checkpointing choice are explicit in EVERY mode (timing, overfit, resume test, train,
# extend) and part of the recipe / gate key; a trainer default can no longer pick them silently (defaults were 8 and 1 while
# timing chose micro 1)
if a.micro is None or a.grad_ckpt is None:
    raise SystemExit("--micro and --grad_ckpt must be passed explicitly in every mode (slurm/wan_teacher.sh: MICRO=<n> GRAD_CKPT=<0|1>)")
if a.micro < 1 or a.batch % a.micro:
    raise SystemExit("--batch must be a multiple of --micro")
if not a.manifest:  # F2: no job opens a byte that was not hashed (the CPU manifest + hash-on-copy)
    raise SystemExit("--manifest is required in every mode: run scripts/verify_manifest.py (CPU job) and pass its output")
EVIDENCE_SHA = WP.evidence_sha()  # Astra t1 item 11: identity of every evidence-producing file (staging, forecast, rehearsal ...)
TJ_PATH = a.timing_json or (sorted(glob.glob("runs/wan_timing_*/timing.json"), key=os.path.getmtime) or [""])[-1]
TJ = None
LEDGER = WP.Ledger(a.ledger, KIND).start()
if not a.timing:  # CONVERGED item 7: no overfit / resume test / production unless the re-time passed with THIS setting
    try:
        TJ = json.load(open(TJ_PATH))
    except Exception as _e:  # noqa: BLE001
        raise SystemExit(f"{('overfit' if a.overfit else 'resume test' if a.resume_test else 'MODE=train')} needs a re-time timing.json "
                         f"({TJ_PATH!r}: {type(_e).__name__}); run MODE=timing")
    # Astra t1 item 11: the stored PASS is not enough: the forecast is refreshed against the ACTUAL ledger (every job since the
    # timing, this allocation included) and the gate registry before each stage is authorised
    _gk = TJ.get("gate_key")
    _done = {k for k in ("overfit", "resume_test") if _gk and any(r.get("kind") == k and r.get("status") == "PASS" and r.get("gate_key") == _gk
                                                                  for r in WP.gate_records(a.gates))}
    _start0 = 0
    if PROD and os.path.exists(f"{a.out}/ckpt.pt"):
        _start0 = int(torch.load(f"{a.out}/ckpt.pt", map_location="cpu", weights_only=False, mmap=True)["step"])
    ok_rt, why_rt = WP.retime_gate(TJ, a.micro, a.grad_ckpt, evidence=EVIDENCE_SHA, used_h=LEDGER.hours(), done_kinds=_done,
                                   start_step=_start0 if PROD else 0)
    if not ok_rt:
        LEDGER.stop("failed")
        raise SystemExit(f"refused by the re-time rule ({TJ_PATH}): {why_rt}")
CODE_SHA = WC.code_sha(__file__, "mrv/wan_cond.py", "mrv/wan_plan.py")
LISTS = json.load(open(a.lists))
CANON = json.load(open(WP.CANON_LISTS))
# Astra t1 item 3: timing / train / extend use EXACTLY the 144 distinct canonical train IDs and the 6 dev IDs; the gates may use a
# subset, each ID inside its original split
_ok_l, _why_l = WP.check_lists(list(LISTS["train"]), list(LISTS["dev"]), CANON, "gate" if (a.overfit or a.resume_test) else "full")
if not _ok_l:
    raise SystemExit(f"drive lists refused ({a.lists}): {_why_l}")
TRAIN, DEVL = sorted(LISTS["train"]), sorted(LISTS["dev"])
WP.check_split_membership(TRAIN, LISTS, allowed=("train",)); WP.check_split_membership(DEVL, LISTS, allowed=("dev",))
assert not set(TRAIN) & set(DEVL), "train / dev overlap"
assert not (set(TRAIN) | set(DEVL)) & (set(LISTS.get("decision", [])) | set(LISTS.get("final", []))), "sealed drives in train / dev"
NL = 5
S_SLOTS = WC.SLOTS[a.tok_ticks]
if PROD:  # production: ALWAYS the measured total for the panel's own settings; overrides may only raise it (S1e 4a / 4b)
    _tj = TJ_PATH
    try:
        _fc = json.load(open(_tj)).get("forecast_final")
    except Exception as _e:  # noqa: BLE001
        raise SystemExit(f"MODE=train needs the measured panel cost for its reserve ({_tj!r}: {type(_e).__name__}); run MODE=timing")
    a.reserve_hours = WP.production_reserve_h(_fc, a.reserve_panels, a.reserve_hours, a.panel_window_latents, a.panel_swap_at)
    print(f"panel reserve {a.reserve_hours:.2f} h (>= {a.reserve_panels} x measured panel cost for window {a.panel_window_latents}, "
          f"swap at {a.panel_swap_at}, from {_tj})", flush=True)
elif a.reserve_hours is None:  # timing / overfit / resume test: no panel reserve
    a.reserve_hours = 0.0
if PROD and LEDGER.hours() + a.reserve_hours >= a.budget_hours:
    raise SystemExit(f"S1 budget exhausted: {LEDGER.hours():.2f} h used + {a.reserve_hours} h reserve >= {a.budget_hours} h")


# ------------------------------------------------------------------------------------------------------------- data
# F1 + F2 (CONVERGED items 2-3): stage the caches this job opens (+ the Wan weights) to job-local /dev/shm, hashing while copying
from mrv import wan_stage as WS
MAN, MANIFEST_SHA = WS.load_manifest(a.manifest)
ROOTS = {"cache": WS.norm(a.cache), "tokens": WS.norm(a.tokens), "guides": WS.norm(a.guides), "enc": WS.norm(a.enc)}
if {k: (MAN.get("roots") or {}).get(k) for k in ROOTS} != ROOTS:
    raise SystemExit(f"manifest roots {MAN.get('roots')} != this job's data folders {ROOTS}: re-run verify_manifest for them")
if (MAN.get("split") or {}).get("train") != sorted(CANON["train"]) or (MAN.get("split") or {}).get("dev") != sorted(CANON["dev"]):
    raise SystemExit("the manifest was not written for the canonical 144 + 6 lists (data/temporal_lists.json): re-run verify_manifest")
if bool(a.use_tokens) and not MAN.get("use_tokens"):
    raise SystemExit("manifest was written without token caches (--use_tokens 0) but this job uses tokens")
USE_DEV = bool(a.val_every)  # the fixed dev probe opens the dev drives' caches: staged and hashed too
STAGE_DRIVES = TRAIN + (DEVL if USE_DEV else [])
_alarm = WS.change_alarm(MAN, [f"{a.guides}/{s_}.pt" for s_ in STAGE_DRIVES] + [f"{a.enc}/{s_}.pt" for s_ in STAGE_DRIVES])
if _alarm:  # size / mtime is a change alarm only: a match proves nothing, a mismatch refuses
    raise SystemExit(f"guide / enc files changed since the manifest was hashed ({len(_alarm)}, e.g. {_alarm[:3]}): re-run verify_manifest")
STAGE = a.stage_dir or f"/dev/shm/mrv_{os.environ.get('SLURM_JOB_ID') or os.getpid()}"
_RECEIPT = os.path.join(STAGE, "STAGED.json")
_created = not os.path.exists(STAGE)
if not _created and os.listdir(STAGE) and not os.path.exists(_RECEIPT):
    raise SystemExit(f"--stage_dir {STAGE} exists, is not empty and has no stage receipt: refusing to write into it")
os.makedirs(STAGE, exist_ok=True)
if _created and not a.keep_stage:
    atexit.register(shutil.rmtree, STAGE, True)
_jobs, LAYOUT = WS.stage_plan(MAN, STAGE_DRIVES, ROOTS, STAGE, bool(a.use_tokens), include_wan=True)
STAGE_BYTES = sum(j[3] for j in _jobs)
MEM_SAMPLER = WS.PeakSampler(1.0).start()  # in-process cgroup sampler from BEFORE the copy workers start (Astra t1 item 8)
_cg0 = WS.cgroup_mem()
MEM_REQUEST = WS.slurm_requested_bytes()  # Astra t1 item 8: the Slurm request, never an (possibly larger) ancestor cgroup limit
_ok_m, _msg_m = WS.mem_check(STAGE_BYTES, MEM_REQUEST if MEM_REQUEST else _cg0["limit"])
print(("MEMCHECK " + ("PASS " if _ok_m else "UNKNOWN " if _ok_m is None else "FAIL ")) + _msg_m, flush=True)
if _ok_m is False:  # files in /dev/shm count against the job's memory cgroup (ConstrainRAMSpace)
    raise SystemExit(_msg_m)
# Astra t1 item 9: files an earlier process of the SAME allocation staged are reused only after RE-HASHING each destination
# (a writable tmpfs file plus a receipt is not proof); any byte difference -> copied and hashed again
_jid = os.environ.get("SLURM_JOB_ID")
print(f"stage-in: {len(_jobs)} files ({STAGE_BYTES / 1e9:.1f} GB) -> {STAGE} with {a.stage_streams} streams", flush=True)
STAGE_REC = WS.stage_reuse(_jobs, _RECEIPT, _jid, MANIFEST_SHA, a.stage_streams, log=lambda m: print(m, flush=True))
STAGE_REC.update(dir=STAGE, total_files=len(_jobs), total_bytes=STAGE_BYTES, drives=len(STAGE_DRIVES), manifest_sha=MANIFEST_SHA,
                 mem_limit=_cg0["limit"], mem_request=MEM_REQUEST, t_done=time.time(), since_process_start_s=round(time.time() - T_PROC0, 1))
print("STAGED", json.dumps({k: v for k, v in STAGE_REC.items() if k != "mismatches"}), flush=True)
WAN_SRC = LAYOUT["wan"]  # weights load from the staged, hashed copy; the recipe / cfg keep the repo id (a.wan)


def sidecar(d, sc, root):
    """sidecar of a STAGED cache (hashed on copy); its recorded sha must be the manifest's (verify_manifest checked file == sidecar)"""
    j = json.load(open(f"{d}/{sc}.json"))
    if not j.get("complete"):
        raise SystemExit(f"{d}/{sc}.json not complete")
    if j.get("sha256") != WS.entry(MAN, f"{root}/{sc}.pt")["sha256"]:
        raise SystemExit(f"{root}/{sc}.json: sidecar sha differs from the manifest")
    return j


GUIDE_VERSION = None


def load_split(drives, cdir, tdir):
    """caches of `drives` from cdir / tdir (the stage dir normally; the Lustre originals only for the timing byte-equality check)"""
    global GUIDE_VERSION
    C, T, shas = {}, {}, []
    miss = [s for s in drives if not os.path.exists(f"{cdir}/{s}.pt") or (a.use_tokens and not os.path.exists(f"{tdir}/{s}.pt"))]
    if miss:
        raise SystemExit(f"missing cache / token files for {len(miss)} drives, e.g. {miss[:3]}")
    for sc in drives:
        jc = sidecar(cdir, sc, a.cache); shas.append(sc + ":" + jc["sha256"])
        jt = sidecar(tdir, sc, a.tokens) if a.use_tokens else None
        for j_ in [jc] + ([jt] if jt else []):
            if GUIDE_VERSION is None:
                GUIDE_VERSION = j_["guide_version"]
            if j_["guide_version"] != GUIDE_VERSION:
                raise SystemExit(f"{sc}: mixed guide versions across drives / caches: {j_['guide_version']} vs {GUIDE_VERSION}")
        # S1b item 2: the latent and token caches of a drive must come from the SAME guide and enc files, and those files must
        # be the ones the manifest hashed (the CPU job compared them; here the in-memory shas must agree)
        if jt is not None and (jt.get("guide_sha") != jc.get("guide_sha") or jt.get("enc_sha") != jc.get("enc_sha")):
            raise SystemExit(f"{sc}: latent cache and token cache were built from different guide / enc files")
        if jt is not None and not (jt.get("shared_pass_ok") and jt.get("weights_sha")):
            raise SystemExit(f"{sc}: token cache lacks the shared-depth / weight-hash evidence")
        if WS.entry(MAN, f"{a.guides}/{sc}.pt")["sha256"] != jc.get("guide_sha") or WS.entry(MAN, f"{a.enc}/{sc}.pt")["sha256"] != jc.get("enc_sha"):
            raise SystemExit(f"{sc}: guide / enc file in the manifest differs from the one the caches were built from")
        C[sc] = torch.load(f"{cdir}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True)
        if a.use_tokens:
            shas.append(sc + ":tok:" + jt["sha256"])
            T[sc] = torch.load(f"{tdir}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True)
            assert torch.equal(T[sc]["ticks"], C[sc]["ticks"]), f"{sc}: token ticks != cache ticks"
            assert tuple(T[sc]["tokens"].shape[2:]) == (WC.TOK_P, WC.TOK_DIM), f"{sc}: token grid {tuple(T[sc]['tokens'].shape)}"
    return C, T, shas


def candidates(C):
    mid, start, per = [], [], {}
    for sc, c in C.items():
        ok = c["ok"].tolist(); hard = c["hard"].tolist()
        st = [{"pos0": s_["pos0"], "L": s_["key"].shape[1]} for s_ in c["streams"]]
        m_ = [(sc, k, j0) for k, j0 in WP.mid_candidates(st, ok, hard, NL)]
        s_ = [(sc, k) for k in range(len(c["starts"]))]
        mid += m_; start += s_; per[sc] = (len(m_), len(s_))
    return mid, start, per


def clip(C, T, desc, kind):
    """-> key [16,5,32,64], guide [16,5,32,64], src [8,5,32,64], tok [5,S,6,P,384] f16 | None, ray [5,S,6,P,6] | None, ticks"""
    if kind == "mid":
        sc, si, j0 = desc; c = C[sc]; st = c["streams"][si]
        key, gd = st["key"][:, j0:j0 + NL].float(), st["guide"][:, j0:j0 + NL].float()
        spans = [[st["pos0"] + 4 * j - 3 + k for k in range(4)] for j in range(j0, j0 + NL)]
    else:
        sc, k_ = desc; c = C[sc]; s0 = int(c["starts"][k_])
        key, gd = c["start_key"][k_].float(), c["start_guide"][k_].float()
        spans = [[s0 + f for f in sp] for sp in WP.latent_spans(NL, first_single=True)]
    src = WC.fold_src(c["src_lr"].float(), spans)
    tok = ray = None
    if a.use_tokens:
        idx = torch.tensor([i for sp in spans for i in WP.token_slots(sp, a.tok_ticks)])
        tk, ry = T[sc]["tokens"][idx], T[sc]["rays"][idx]
        tok = tk.view(NL, S_SLOTS, *tk.shape[1:]); ray = ry.view(NL, S_SLOTS, *ry.shape[1:]).float()
    return key, gd, src, tok, ray, (sc, [sp for sp in spans])


C_TR, T_TR, SHAS = load_split(TRAIN, LAYOUT["cache"], LAYOUT["tokens"])
MID, START, PER_DRIVE = candidates(C_TR)
DATA_SHA = hashlib.sha256("\n".join(SHAS).encode()).hexdigest()[:16]
VAE_SHAS = {c["meta"]["vae_sha"] for c in C_TR.values()}
assert len(VAE_SHAS) == 1, f"cache built with several VAEs: {VAE_SHAS}"
no_clip = [sc for sc, (m_, s_) in PER_DRIVE.items() if m_ + s_ == 0]
print(f"clips: {len(MID)} mid-stream + {len(START)} true starts; contributing drives {len(TRAIN) - len(no_clip)}/{len(TRAIN)}"
      f" (none from {no_clip[:6]}{'...' if len(no_clip) > 6 else ''}); data_sha {DATA_SHA}; manifest {MANIFEST_SHA[:16]}; guide version {GUIDE_VERSION}", flush=True)
if not MID:
    raise SystemExit("no mid-stream training clips")
if a.overfit:
    g0 = torch.Generator().manual_seed(1234)
    n_s = min(len(START), a.overfit // 4)
    OF = [("mid", MID[i]) for i in torch.randperm(len(MID), generator=g0)[:a.overfit - n_s].tolist()] + \
         [("start", START[i]) for i in torch.randperm(len(START), generator=g0)[:n_s].tolist()]
    assert len(OF) == a.overfit and len(set(OF)) == a.overfit, "overfit clips must be distinct"


def sample_sigma(g, n):
    u = torch.sigmoid(torch.randn(n, generator=g)) if a.sigma_dist == "logit_normal" else torch.rand(n, generator=g)
    return WC.sigma_shift(u, a.shift).clamp(1e-4, 1.0)


def assemble(items, C=None, T=None):
    C = C_TR if C is None else C; T = T_TR if T is None else T
    cl = [clip(C, T, d, k) for k, d in items]
    st = lambda i: torch.stack([c[i] for c in cl]) if cl[0][i] is not None else None
    return st(0), st(1), st(2), st(3), st(4), [c[5] for c in cl]


def make_batch(step, C=None, T=None):
    g = torch.Generator().manual_seed(a.seed * 100003 + step)
    B = a.batch
    is_start = (torch.rand(B, generator=g) < a.p_start).tolist()
    pm = torch.randint(0, len(MID), (B,), generator=g).tolist(); ps = torch.randint(0, max(1, len(START)), (B,), generator=g).tolist()
    sig = sample_sigma(g, B)
    eps = torch.randn(B, 16, NL, 32, 64, generator=g)
    gd_, td_ = WP.drop_masks(torch.rand(B, generator=g).tolist(), a.p_drop_both, a.p_drop_guide_only, a.p_drop_tokens_only)  # joint dropout
    gdrop, tdrop = torch.tensor(gd_), torch.tensor(td_)
    holes = torch.rand(B, generator=g) < a.p_holes
    hole_boxes = [[(int(torch.randint(3, 13, (1,), generator=g)), int(torch.randint(4, 25, (1,), generator=g)),
                    float(torch.rand(1, generator=g)), float(torch.rand(1, generator=g))) for _ in range(3)] for _ in range(B)]
    n_holes = torch.randint(1, 4, (B,), generator=g).tolist()
    if step < 2:  # deterministic active inputs for the guard (update 1) and the gate-open gradient check (update 2)
        gdrop[:] = False; tdrop[:] = False; holes[:] = False
    if a.overfit:
        descs = list(OF)
    else:
        descs = [("start", START[ps[b]]) if (is_start[b] and START) else ("mid", MID[pm[b]]) for b in range(B)]
    K, G, M, TK, RY, _ = assemble(descs, C, T)
    for b in range(B):
        if gdrop[b]:
            G[b] = 0; M[b] = 0
        elif holes[b]:
            for hh, ww, fy, fx in hole_boxes[b][:n_holes[b]]:
                y0, x0 = int(fy * (32 - hh)), int(fx * (64 - ww))
                G[b, :, :, y0:y0 + hh, x0:x0 + ww] = 0; M[b, :, :, y0:y0 + hh, x0:x0 + ww] = 0
    sha = hashlib.sha1(str(descs).encode() + sig.numpy().tobytes() + eps[:, 0, 0, 0, :4].numpy().tobytes()
                       + gdrop.numpy().tobytes() + tdrop.numpy().tobytes() + holes.numpy().tobytes()).hexdigest()[:10]
    return {"key": K, "guide": G, "src": M, "tok": TK, "ray": RY, "sigma": sig, "eps": eps, "tkeep": ~tdrop, "sha": sha,
            "n_start": sum(1 for k, _ in descs if k == "start"), "gdrop": int(gdrop.sum()), "tdrop": int(tdrop.sum()),
            "guide_active": bool((~gdrop).any()), "tok_active": bool((~tdrop).any())}


# --------------------------------------------------------------------------------------------------------- model
prompt = torch.load(a.prompt_embed, map_location="cpu") if a.prompt_embed else None
PROMPT_NOTE = (f"text slot = cached embedding {a.prompt_embed}" if a.prompt_embed else
               "text slot = ONE all-zero 4096-d row (default; stock Wan never saw an all-zero prompt: Fable 07:30 suggests an "
               "off-cluster umT5 embedding of a plain scene caption via --prompt_embed before MODE=train)")
print("NOTE", PROMPT_NOTE, flush=True)
DROP = {"p_drop_both": a.p_drop_both, "p_drop_guide_only": a.p_drop_guide_only, "p_drop_tokens_only": a.p_drop_tokens_only,
        "guide_dropped_marginal": a.p_drop_both + a.p_drop_guide_only, "tokens_dropped_marginal": a.p_drop_both + a.p_drop_tokens_only,
        "holes_if_guide_kept": a.p_holes, "updates_1_2": "no dropout (guards)"}
WP.drop_masks([0.5], a.p_drop_both, a.p_drop_guide_only, a.p_drop_tokens_only)  # validates the shares
model, DIAG = WC.build(WAN_SRC, rank=a.rank, tok_ticks=a.tok_ticks, use_tokens=bool(a.use_tokens), dtype=AMP, device=dev, prompt=prompt,
                       grad_ckpt=bool(a.grad_ckpt), parity=True, seed=a.seed)
print("init diagnostics", DIAG, flush=True)
WC.check_parity(DIAG)
model.train()
TR = model.trainable(); params = [q for _, q in TR]
INIT_SHA = WC.tensor_sha(TR)
n_lora = sum(q.numel() for n, q in TR if "lora_" in n); n_new = sum(q.numel() for n, q in TR if "lora_" not in n)
print(f"trainable: LoRA {n_lora / 1e6:.1f}M, new {n_new / 1e6:.2f}M; init sha {INIT_SHA}", flush=True)
BASE_SHA = WC.base_sha(model)
from diffusers import AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained(WAN_SRC, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).eval()
vae.requires_grad_(False)
VAE_SHA = WC.vae_sha(vae)
if VAE_SHAS != {VAE_SHA}:
    raise SystemExit(f"cache latents were made with another VAE ({VAE_SHAS}) than the one on disk ({VAE_SHA})")
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
RECIPE = {k: getattr(a, k) for k in ("seed", "lists", "cache", "tokens", "rank", "tok_ticks", "use_tokens", "batch", "lr", "warmup", "shift",
                                     "sigma_dist", "p_start", "p_drop_both", "p_drop_guide_only", "p_drop_tokens_only", "p_holes", "overfit", "prompt_embed", "wan",
                                     "micro", "grad_ckpt")}  # F5: micro / grad_ckpt explicit and part of the recipe
CFG = {"rank": a.rank, "tok_ticks": a.tok_ticks, "use_tokens": bool(a.use_tokens), "shift": a.shift, "n_extra": WC.N_EXTRA,
       "prompt_embed": a.prompt_embed, "wan": a.wan, "clip_latents": NL, "tok_grid": WC.TOK_GRID, "seed": a.seed}
GATE_KEY = WP.gate_key(CODE_SHA, RECIPE, BASE_SHA, VAE_SHA, GUIDE_VERSION, MANIFEST_SHA)
if TJ is not None and TJ.get("gate_key") != GATE_KEY:  # the re-time must have run this code / recipe / data (CONVERGED item 7)
    raise SystemExit(f"re-time {TJ_PATH} was measured for gate key {TJ.get('gate_key')}, this run is {GATE_KEY} (code / recipe / data / "
                     f"micro / grad_ckpt changed since): re-run MODE=timing")
if PROD and a.budget_hours > 24.0 + 1e-9:  # Astra 08:30: more than 24 h only on the qualifying 24 h EXTEND verdict
    MK = f"{a.out}/S1_24H_DECISION.json"
    v_ = json.load(open(MK)).get("verdict", "") if os.path.exists(MK) else ""
    if not v_.startswith("EXTEND"):
        raise SystemExit(f"--budget_hours {a.budget_hours} > 24 refused: {MK} missing or not an EXTEND verdict ({v_!r})")
if PROD:  # Astra S1c item 6 + CONVERGED item 4: the scorer parity OFF the slice (2080 Ti: A@20k replay + sharded == unsharded),
    # bound to device / references / adapter / scorer / decoder / wrapper code, must have PASSED before production starts
    _ok_p, _why_p = WP.score_parity_ok(a.parity_record, WP.score_binding(a.a_metrics, a.a_roll))
    if not _ok_p:
        raise SystemExit(f"MODE=train refused: no bound 2080 Ti scorer-parity PASS ({a.parity_record}: {_why_p}); run slurm/wan_score.sh parity")
if PROD:  # Fable 08:30: production needs PASS overfit AND PASS resume-test records for this code / recipe
    ok_g, have_g = WP.gates_ok(a.gates, GATE_KEY)
    if not ok_g:
        raise SystemExit(f"MODE=train refused: gate records for key {GATE_KEY} in {a.gates}: {have_g} (need overfit PASS and resume_test PASS)")
    print(f"gates PASS for key {GATE_KEY}: {have_g}", flush=True)
start, ck, CHAIN, OF0, OF_HIST = 0, f"{a.out}/ckpt.pt", "", None, []
if os.path.exists(ck):
    if not a.resume:
        raise SystemExit(f"{ck} exists: pass --resume or use a fresh --out")
    s_ = torch.load(ck, map_location="cpu", weights_only=False)
    if s_.get("timing") and not TEST:
        raise SystemExit("refusing to resume production training from a timing / test checkpoint")
    for k_, v_ in RECIPE.items():
        if s_["recipe"].get(k_) != v_:
            raise SystemExit(f"incompatible resume: recipe {k_} {s_['recipe'].get(k_)} != {v_}")
    for k_, v_ in (("code_sha", CODE_SHA), ("base_sha", BASE_SHA), ("vae_sha", VAE_SHA), ("data_sha", DATA_SHA), ("guide_version", GUIDE_VERSION),
                   ("manifest_sha", MANIFEST_SHA)):
        if s_.get(k_) != v_:
            raise SystemExit(f"incompatible resume: {k_} {s_.get(k_)} != {v_}")
    model.load_trainable(s_["trainable"]); opt.load_state_dict(s_["opt"])
    start, CHAIN = s_["step"], s_["pick_chain"]
    OF0 = s_.get("overfit_eval0"); OF_HIST = list(s_.get("overfit_hist") or [])
    if a.overfit and OF0 is None:
        raise SystemExit("resumed overfit run without its update-0 eval loss in the checkpoint")
    torch.set_rng_state(s_["rng"]["torch"]); torch.cuda.set_rng_state(s_["rng"]["cuda"]); random.setstate(s_["rng"]["python"]); np.random.set_state(s_["rng"]["numpy"])
    print(f"resumed at update {start}; ledger {LEDGER.hours():.2f} h used", flush=True)
json.dump({**vars(a), "kind": KIND, "code_sha": CODE_SHA, "base_sha": BASE_SHA, "vae_sha": VAE_SHA, "data_sha": DATA_SHA, "manifest_sha": MANIFEST_SHA,
           "stage": STAGE_REC, "init_sha": INIT_SHA,
           "guide_version": GUIDE_VERSION, "mid_clips": len(MID), "start_clips": len(START), "clips_per_drive": PER_DRIVE,
           "drives_without_clips": no_clip, "trainable_lora_M": round(n_lora / 1e6, 2), "trainable_new_M": round(n_new / 1e6, 3),
           "init_diagnostics": DIAG, "cfg": CFG, "ledger_hours_at_start": LEDGER.hours(), "prompt_note": PROMPT_NOTE, "dropout": DROP, "gate_key": GATE_KEY}, open(f"{a.out}/args.json", "w"), indent=1)


def save(step, path):
    d = {"trainable": model.trainable_state(), "opt": opt.state_dict(), "step": step, "hours": LEDGER.hours(), "recipe": RECIPE, "cfg": CFG,
         "code_sha": CODE_SHA, "base_sha": BASE_SHA, "vae_sha": VAE_SHA, "data_sha": DATA_SHA, "manifest_sha": MANIFEST_SHA, "guide_version": GUIDE_VERSION,
         "init_sha": INIT_SHA, "pick_chain": CHAIN, "timing": bool(TEST), "args": vars(a), "overfit_eval0": OF0, "overfit_hist": OF_HIST, "gate_key": GATE_KEY,
         "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}}
    torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)


def to_dev(bt):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in bt.items()}


def flow_loss(bt, sl, reduce=True):
    x0, eps, sig = bt["key"][sl], bt["eps"][sl], bt["sigma"][sl]
    s5 = sig.view(-1, 1, 1, 1, 1)
    xt = (1 - s5) * x0 + s5 * eps
    extra = torch.cat([bt["guide"][sl], bt["src"][sl]], 1)
    tok = bt["tok"][sl] if bt["tok"] is not None else None
    ray = bt["ray"][sl] if bt["ray"] is not None else None
    with torch.autocast("cuda", dtype=AMP):
        v = model(xt, extra, sig * 1000.0, tok, ray, bt["tkeep"][sl])
    err = (v.float() - (eps - x0)) ** 2
    return err.mean() if reduce else err.mean(dim=(1, 2, 3, 4))


# ------------------------------------------------------------------------------------------- dev probe (fixed)
PROBE = None
if a.val_every:  # dev caches were staged + hashed with the train drives (USE_DEV)
    C_DV, T_DV, _ = load_split(DEVL, LAYOUT["cache"], LAYOUT["tokens"])  # also enforces ONE guide version for train and dev
    mid_d, start_d, _ = candidates(C_DV)
    gp = torch.Generator().manual_seed(4242)
    by_drive = {}
    for d in mid_d:
        by_drive.setdefault(d[0], []).append(d)
    drv = sorted(by_drive)
    sel = [("mid", by_drive[drv[i % len(drv)]][int(torch.randint(0, len(by_drive[drv[i % len(drv)]]), (1,), generator=gp))]) for i in range(8)]
    sel += [("start", start_d[int(i)]) for i in torch.randperm(len(start_d), generator=gp)[:2].tolist()] if start_d else []
    donors = []
    for k, d in sel:
        pool = [x for x in (mid_d if k == "mid" else start_d) if x[0] != d[0]]
        donors.append((k, pool[int(torch.randint(0, len(pool), (1,), generator=gp))]))
    def _stack(items):
        cl = [clip(C_DV, T_DV, d, k) for k, d in items]
        return [torch.stack([c[i] for c in cl]) if cl[0][i] is not None else None for i in range(5)]
    PK, PG, PM, PT, PR = _stack(sel); _, DG, DM, DT, DR = _stack(donors)
    PROBE = {"key": PK, "guide": PG, "src": PM, "tok": PT, "ray": PR, "dguide": DG, "dsrc": DM, "dtok": DT, "dray": DR,
             "eps": torch.randn(len(sel), 16, NL, 32, 64, generator=gp)}
    print(f"dev probe: {len(sel)} fixed clips from {len({d[0] for _, d in sel})} dev drives", flush=True)
    del C_DV, T_DV


@torch.no_grad()
def probe(step):
    model.eval(); res = {}
    n = PROBE["key"].shape[0]; on = torch.ones(n, dtype=torch.bool); off = ~on
    zg, zm = torch.zeros_like(PROBE["guide"]), torch.zeros_like(PROBE["src"])
    modes = {"hybrid": (PROBE["guide"], PROBE["src"], PROBE["tok"], PROBE["ray"], on),
             "guide_only": (PROBE["guide"], PROBE["src"], PROBE["tok"], PROBE["ray"], off),
             "tokens_only": (zg, zm, PROBE["tok"], PROBE["ray"], on), "none": (zg, zm, PROBE["tok"], PROBE["ray"], off),
             "other_drive": (PROBE["dguide"], PROBE["dsrc"], PROBE["dtok"], PROBE["dray"], on)}
    for sv in (0.3, 0.6, 0.9):
        res[str(sv)] = {}
        for name, (g_, m_, t_, r_, kp) in modes.items():
            bt = to_dev({"key": PROBE["key"], "eps": PROBE["eps"], "sigma": torch.full((n,), sv), "guide": g_, "src": m_, "tok": t_, "ray": r_, "tkeep": kp})
            ls = [flow_loss(bt, slice(i, i + a.micro), reduce=False).cpu() for i in range(0, n, a.micro)]
            res[str(sv)][name] = round(torch.cat(ls).mean().item(), 5)
    bt = to_dev({"key": PROBE["key"][:2], "eps": PROBE["eps"][:2], "guide": PROBE["guide"][:2], "src": PROBE["src"][:2],
                 "tok": PROBE["tok"][:2] if PROBE["tok"] is not None else None, "ray": PROBE["ray"][:2] if PROBE["ray"] is not None else None})
    xt = 0.4 * bt["key"] + 0.6 * bt["eps"]
    sens = WC.sensitivity(model, xt, torch.cat([bt["guide"], bt["src"]], 1), torch.full((2,), 600.0, device=dev), bt["tok"], bt["ray"], AMP)
    model.train()
    ok_s, why_s = WC.check_sensitivity(sens, bool(a.use_tokens))
    rec = {"update": step, "opt_step": opt_step(), "probe": res, "sensitivity_sigma0.6": sens, "sensitivity_pass": ok_s, "sensitivity_why": why_s,
           "code_sha": CODE_SHA}
    print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    return rec


log = open(f"{a.out}/log.jsonl", "a")
STARTUP = {}


def note_first_update():
    """start-up from process start (and from the allocation, MRV_ALLOC_T0 exported by slurm/wan_teacher.sh) to the end of the
    first update, with the stage-in seconds / bytes (CONVERGED item 2: priced from allocation, no extrapolation)"""
    if STARTUP:
        return
    t = time.time(); t_alloc = float(os.environ.get("MRV_ALLOC_T0") or 0) or None
    STARTUP.update(process_to_first_update_s=round(t - T_PROC0, 1), alloc_to_first_update_s=round(t - t_alloc, 1) if t_alloc else None,
                   stage_in_s=STAGE_REC["seconds"], stage_bytes=STAGE_REC["bytes"], stage_total_bytes=STAGE_REC["total_bytes"],
                   stage_mb_s=STAGE_REC["mb_s"], stage_cold=STAGE_REC["cold"], stage_streams=a.stage_streams, drives=len(STAGE_DRIVES),
                   manifest_sha=MANIFEST_SHA, job=os.environ.get("SLURM_JOB_ID"))
    rec = {"startup": STARTUP}
    print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()


# ------------------------------------------------------------------------------------------------- one update
def update(step, bt, timing=False):
    """forward / backward (accumulated) / clip on one global batch (Adam is the caller's). Returns loss, grad norm, total s and,
    with timing=True, separately synchronised forward and backward seconds (summed over microbatches)."""
    for g_ in opt.param_groups:
        g_["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
    torch.cuda.synchronize(); t0 = time.time(); tf = tb = 0.0
    opt.zero_grad(set_to_none=True)
    nacc = a.batch // a.micro; loss_v = 0.0
    for i in range(nacc):
        if timing:
            torch.cuda.synchronize(); ta = time.time()
        l_ = flow_loss(bt, slice(i * a.micro, (i + 1) * a.micro)) / nacc
        if timing:
            torch.cuda.synchronize(); tb_ = time.time(); tf += tb_ - ta
        l_.backward()
        if timing:
            torch.cuda.synchronize(); tb += time.time() - tb_
        loss_v += l_.item()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    if not torch.isfinite(gn):
        raise SystemExit(f"non-finite grad norm at update {step + 1}")
    torch.cuda.synchronize(); t1 = time.time()
    return (loss_v, gn, t1 - t0, tf, tb) if timing else (loss_v, gn, t1 - t0)


def start_producer(s0, s1):
    """the production prefetch: one thread assembles + pins batches s0..s1-1 into a 3-deep queue"""
    Q_ = queue.Queue(maxsize=3)

    def run():
        try:
            for s in range(s0, s1):
                t0 = time.time(); bt = make_batch(s)
                for k in ("key", "guide", "src", "eps", "tok", "ray"):
                    if bt[k] is not None:
                        bt[k] = bt[k].pin_memory()
                bt["data_s"] = time.time() - t0
                Q_.put((s, bt))
        except BaseException as e:  # surface data errors instead of hanging on Q.get()
            Q_.put((-1, e))
    threading.Thread(target=run, daemon=True).start()
    return Q_


def qget(Q_, step):
    s_, bt = Q_.get()
    if s_ == -1:
        raise SystemExit(f"data producer failed: {bt!r}")
    assert s_ == step, (s_, step)
    return bt


# ------------------------------------------------------------------------------------------- first-update guard (item 5)
def frozen_snapshot():
    """production's first-update snapshot: frozen base tensors (CPU copies), the new modules / LoRA-B, the VAE sha"""
    ref = model.frozen()
    return {"ref": ref, "pre": [q.detach().cpu().clone() for _, q in ref],
            "new": {n: q.detach().clone() for n, q in TR if "lora_" not in n or "lora_B" in n}, "vae": WC.vae_sha(vae)}


def guard_verify(step, snap):
    """production's first-update guard (item 5): base transformer + VAE bit-identical; on a fresh run (step 0) the extra conv,
    all 30 gates (tokens on) and LoRA-B of 8 targets x 30 blocks moved"""
    assert all(torch.equal(q.detach().cpu(), q0) for (_, q), q0 in zip(snap["ref"], snap["pre"])), "frozen base transformer changed!"
    assert WC.vae_sha(vae) == snap["vae"] == VAE_SHA, "VAE changed!"
    if step == 0:
        cur = dict(TR); moved = {n: not torch.equal(cur[n], q0) for n, q0 in snap["new"].items()}
        must = [n for n in moved if ".patch_embedding.extra." in n or (n.endswith(".attn2.gate") and a.use_tokens)
                or ("lora_B" in n and not any(x in n for x in (".attn2.inner.to_q.", ".attn2.inner.to_k.")))]
        bad = [n for n in must if not moved[n]]
        n_b = sum(1 for n in must if "lora_B" in n); n_g = sum(1 for n in must if n.endswith(".attn2.gate"))
        assert n_b == 8 * 30 and (n_g == 30 or not a.use_tokens), f"guard coverage: {n_b} LoRA-B, {n_g} gates"
        assert not bad, f"{len(bad)} tensors did not move in update 1 (all inputs active), e.g. {bad[:3]}"
        print(f"guard after update 1: {len(snap['ref'])} base tensors + VAE bit-identical; extra conv, {n_g} gates, {n_b} LoRA-B moved", flush=True)
    else:
        print(f"guard after resumed update {step + 1}: {len(snap['ref'])} base tensors + VAE bit-identical", flush=True)


def opt_step():
    """the optimizer's ACTUAL step count (Astra t1 item 1): max Adam 'step' over the trainable parameters (0 = untouched)"""
    st = [v.get("step") for v in opt.state.values() if "step" in v]
    return int(max(float(x) for x in st)) if st else 0


# ------------------------------------------------------------------------------------------------- timing mode
def set_grad_ckpt(on):
    (model.tf.enable_gradient_checkpointing if on else model.tf.disable_gradient_checkpointing)()


def tensor_bytes_sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()[:16] if t is not None else None


if a.timing and not a.overfit:
    import copy
    T_READY = time.time()  # = where production enters its loop (staging, model, VAE, dev probe built)
    if PROBE is None:
        raise SystemExit("timing needs the dev probe (--val_every > 0): it measures init inertness and the early sensitivity")
    EXPLICIT = {"micro": a.micro, "grad_ckpt": a.grad_ckpt}
    GCS = [int(x) for x in a.timing_grad_ckpt.split(",")]
    MBS = [int(x) for x in a.timing_micro.split(",")]
    rep = {"gpu": torch.cuda.get_device_name(), "batch": a.batch, "tok_ticks": a.tok_ticks, "use_tokens": bool(a.use_tokens), "sweep": {},
           "sweep_declared": {"grad_ckpt": GCS, "micro": MBS}, "train_drives": len(TRAIN), "dev_drives": len(DEVL),
           "train_ids_sha": WP.ids_sha(TRAIN), "dev_ids_sha": WP.ids_sha(DEVL), "stage_drives": len(STAGE_DRIVES),
           "explicit_setting": EXPLICIT, "gate_key_explicit": GATE_KEY, "manifest_sha": MANIFEST_SHA, "data_sha": DATA_SHA, "code_sha": CODE_SHA,
           "evidence_sha": EVIDENCE_SHA, "stage": STAGE_REC, "job": os.environ.get("SLURM_JOB_ID"),
           "init_parity": {"pass": True, "diag": DIAG, "note": "check_parity raised otherwise; forward-only (no_grad): independent of micro / grad_ckpt"}}
    INIT_W = model.trainable_state(); INIT_OPT = copy.deepcopy(opt.state_dict())

    def sens_probe():
        model.eval()
        b2 = to_dev({"key": PROBE["key"][:2], "eps": PROBE["eps"][:2], "guide": PROBE["guide"][:2], "src": PROBE["src"][:2],
                     "tok": PROBE["tok"][:2] if PROBE["tok"] is not None else None, "ray": PROBE["ray"][:2] if PROBE["ray"] is not None else None})
        xt = 0.4 * b2["key"] + 0.6 * b2["eps"]
        r_ = WC.sensitivity(model, xt, torch.cat([b2["guide"], b2["src"]], 1), torch.full((2,), 600.0, device=dev), b2["tok"], b2["ray"], AMP)
        model.train(); return r_

    # (0) Astra t1 item 1: the UNTOUCHED init (optimizer step 0, init weights): guide / token paths inert
    if opt_step() != 0 or WC.tensor_sha(TR) != INIT_SHA:
        raise SystemExit("timing: the model was updated before the init inertness probe")
    S0 = sens_probe()
    rep["init_inertness"] = {"pass": WP.init_inert(S0), "opt_step": opt_step(), "weights_are_init": WC.tensor_sha(TR) == INIT_SHA, "values": S0}
    print("TIMING init inertness (optimizer step 0)", rep["init_inertness"], flush=True)
    s_ = 0
    # (1) F5 sweep: the declared grad_ckpt x micro grid at global batch 8 (synchronous batches; fwd / bwd / Adam timed apart)
    for gc in GCS:
        set_grad_ckpt(gc)
        oom_key = None
        for mb in MBS:
            key = f"gc{gc}_m{mb}"
            if oom_key or a.batch % mb:
                rep["sweep"][key] = {"micro": mb, "grad_ckpt": gc, "skipped_after_oom": oom_key} if oom_key else {"micro": mb, "grad_ckpt": gc, "invalid": "batch % micro"}
                continue
            a.micro = mb; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            rows = []
            try:
                for k in range(6):  # 2 warm + 4 timed
                    tw = time.time(); bt = to_dev(make_batch(s_)); torch.cuda.synchronize(); tdata = time.time() - tw
                    loss_v, gn, tfb, tf, tb = update(s_, bt, timing=True)
                    ta = time.time(); opt.step(); torch.cuda.synchronize(); tad = time.time() - ta
                    s_ += 1
                    if k >= 2:
                        rows.append({"sync_batch_s": tdata, "fwd_s": tf, "bwd_s": tb, "fwd_bwd_clip_s": tfb, "adam_s": tad, "gpu_s": tfb + tad})
                rep["sweep"][key] = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}
                rep["sweep"][key].update(micro=mb, grad_ckpt=gc, peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                         peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
            except torch.cuda.OutOfMemoryError as e:
                rep["sweep"][key] = {"micro": mb, "grad_ckpt": gc, "oom": str(e)[:200]}
            if "oom" in rep["sweep"][key]:  # free outside the except block (its traceback pins the failed step's tensors)
                import gc as _gc
                bt = None; opt.zero_grad(set_to_none=True); _gc.collect(); torch.cuda.empty_cache()
                oom_key = key; print("TIMING", key, "OOM", flush=True); continue
            print("TIMING", key, rep["sweep"][key], flush=True)
    fits = {k: v for k, v in rep["sweep"].items() if "gpu_s" in v}
    if not fits:
        raise SystemExit(f"no setting fits: {rep['sweep']}")
    # (2) Astra t1 item 2: matched update = per-tensor parameter deltas after ONE Adam step from identical weights, optimizer
    # state, batch and RNG under every fitting setting (loss / grad norm alone cannot see reversed or permuted gradients)
    SNAP_W = model.trainable_state(); SNAP_OPT = copy.deepcopy(opt.state_dict()); BFIX = to_dev(make_batch(10 ** 6 + 7))
    ref_key = min(fits, key=lambda k: (-fits[k]["grad_ckpt"], fits[k]["micro"]))  # checkpointing on, smallest micro = the baseline
    DREF, mrows = {}, {}
    for key in [ref_key] + [k for k in fits if k != ref_key]:
        v = fits[key]
        set_grad_ckpt(v["grad_ckpt"]); a.micro = v["micro"]
        model.load_trainable(SNAP_W); opt.load_state_dict(copy.deepcopy(SNAP_OPT)); opt.zero_grad(set_to_none=True)
        torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        l_, g_, _ = update(a.warmup - 1, BFIX)  # full learning rate: deltas well above rounding
        opt.step(); torch.cuda.synchronize()
        rows_t = {}
        with torch.no_grad():
            for n, q in TR:
                d = q.detach().float() - SNAP_W[n].to(q.device)
                if key == ref_key:
                    DREF[n] = d; nr = d.norm().item(); rows_t[n] = {"ref": nr, "diff": 0.0, "norm": nr}
                else:
                    rows_t[n] = {"ref": DREF[n].norm().item(), "diff": (d - DREF[n]).norm().item(), "norm": d.norm().item()}
        mrows[key] = {"loss": l_, "grad_norm": float(g_), "delta": WP.delta_check(rows_t)}
    MATCH = WP.matched_updates({k: {"loss": r["loss"], "grad_norm": r["grad_norm"]} for k, r in mrows.items()}, ref_key, a.match_rtol, a.match_gn_rtol)
    for k, r in MATCH["rows"].items():
        r["delta"] = mrows[k]["delta"]; r["ok"] = bool(r["ok"] and r["delta"]["ok"])
    MATCH.update(method="per_tensor_adam_delta", all_ok=all(r["ok"] for r in MATCH["rows"].values()), delta_rtol=WP.DELTA_RTOL)
    rep["matched_update"] = MATCH
    del BFIX, DREF, SNAP_W, SNAP_OPT; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()  # released before timing the winner
    print("TIMING matched-update check", {k: (r["ok"], r["delta"]["worst_rel"]) for k, r in MATCH["rows"].items()}, flush=True)
    SEL = WP.select_setting(fits, MATCH)
    rep["selected_setting"] = SEL
    run_set = SEL or EXPLICIT  # a failed check leaves the explicit setting for the loop (retime_pass then fails)
    set_grad_ckpt(run_set["grad_ckpt"]); a.micro = run_set["micro"]
    RECIPE_SEL = {**RECIPE, "micro": run_set["micro"], "grad_ckpt": run_set["grad_ckpt"]}
    rep["gate_key"] = WP.gate_key(CODE_SHA, RECIPE_SEL, BASE_SHA, VAE_SHA, GUIDE_VERSION, MANIFEST_SHA)  # what overfit / train must run
    # (3) back to the untouched init + a fresh optimizer; production's first start at the selected setting: frozen snapshot,
    # first update + Adam, guard verification (Astra t1 item 7), then exactly SENS_STEP optimizer steps and the probe
    model.load_trainable(INIT_W); opt.load_state_dict(copy.deepcopy(INIT_OPT)); opt.zero_grad(set_to_none=True)
    if opt_step() != 0 or WC.tensor_sha(TR) != INIT_SHA:
        raise SystemExit("timing: init weights / optimizer state not restored")
    t = time.time(); snap = frozen_snapshot(); t_snap = time.time() - t
    t = time.time(); bt = to_dev(make_batch(0)); update(0, bt); opt.step(); torch.cuda.synchronize(); t_first = time.time() - t
    note_first_update()
    t = time.time(); guard_verify(0, snap); t_guard = time.time() - t; del snap
    for st_ in range(1, WP.SENS_STEP):
        bt = to_dev(make_batch(st_)); update(st_, bt); opt.step()
    torch.cuda.synchronize()
    t = time.time(); prec = probe(WP.SENS_STEP); rep["probe_s"] = round(time.time() - t, 2)
    rep["early_sensitivity"] = {"pass": bool(prec["sensitivity_pass"]), "why": prec["sensitivity_why"], "values": prec["sensitivity_sigma0.6"],
                                "opt_step": opt_step(), "required_step": WP.SENS_STEP, "micro": run_set["micro"], "grad_ckpt": run_set["grad_ckpt"],
                                "rule": "mrv.wan_cond.check_sensitivity after exactly SENS_STEP optimizer steps from the untouched init"}
    print("TIMING early sensitivity", rep["early_sensitivity"], flush=True)
    s_ = WP.SENS_STEP
    # (4) --use_tokens 0 GPU time at the selected setting: tokens dropped for every clip (diagnostic only, never a recipe)
    rows = []
    for k in range(6):
        bt = to_dev(make_batch(s_)); bt["tkeep"] = torch.zeros_like(bt["tkeep"])
        loss_v, gn, tfb = update(s_, bt); ta = time.time(); opt.step(); torch.cuda.synchronize(); s_ += 1
        if k >= 2:
            rows.append(tfb + time.time() - ta)
    with_tok = fits.get(f"gc{run_set['grad_ckpt']}_m{run_set['micro']}", {}).get("gpu_s")
    rep["use_tokens0_diag"] = {"gpu_s": round(float(np.mean(rows)), 3), "gpu_s_with_tokens": with_tok, "diagnostic_only": True,
                               "note": "tokens dropped (tok_keep = 0) at the selected setting: what F4 pooling could save at most"}
    torch.cuda.empty_cache()
    # (5) production loop: >= 200 consecutive updates from the staged data through the real prefetch thread + queue
    NP = a.timing_prod_updates; P0 = s_
    Qp = start_producer(s_, s_ + NP); torch.cuda.reset_peak_memory_stats(); prow = []
    for k in range(NP):
        t0 = time.time(); bt = qget(Qp, s_); tw = time.time() - t0
        bt = to_dev(bt); loss_v, gn, tfb = update(s_, bt)
        ta = time.time(); opt.step(); torch.cuda.synchronize(); tad = time.time() - ta
        prow.append({"queue_wait_s": tw, "producer_assembly_s": bt["data_s"], "gpu_s": tfb + tad, "wall_s": time.time() - t0})
        s_ += 1
    WALL_FC, WDET = WS.forecast_wall(prow, a.timing_block)
    BLK = WS.block_stats(prow, a.timing_block)
    rep["production_loop"] = {"wall_s": round(WALL_FC, 4), "basis": "slowest of the full-run mean and the last two block means", **WDET,
                              "mean_wall_s": round(float(np.mean([r["wall_s"] for r in prow])), 4),
                              "mean_gpu_s": round(float(np.mean([r["gpu_s"] for r in prow])), 4),
                              "mean_queue_wait_s": round(float(np.mean([r["queue_wait_s"] for r in prow])), 4),
                              "mean_producer_assembly_s": round(float(np.mean([r["producer_assembly_s"] for r in prow])), 4),
                              "updates": NP, "first_step": P0, "block": a.timing_block, "blocks": BLK, "micro": run_set["micro"],
                              "grad_ckpt": run_set["grad_ckpt"], "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                              "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2), "source": "stage dir " + STAGE}
    print("TIMING production loop", {k: v for k, v in rep["production_loop"].items() if k != "blocks"}, flush=True)
    for b_ in BLK:
        print("TIMING block", b_, flush=True)
    # (6) the stage dir must give the SAME batches as the Lustre originals (same steps, every tensor byte-equal)
    C_LU, T_LU, SH_LU = load_split(TRAIN, a.cache, a.tokens)
    eq = []
    for st_ in range(P0, P0 + a.lustre_check_batches):
        b1, b2 = make_batch(st_), make_batch(st_, C_LU, T_LU)
        ks = [k for k in ("key", "guide", "src", "tok", "ray", "sigma", "eps", "tkeep") if b1[k] is not None or b2[k] is not None]
        same = b1["sha"] == b2["sha"] and all(b1[k] is not None and b2[k] is not None and b1[k].dtype == b2[k].dtype and torch.equal(b1[k], b2[k]) for k in ks)
        eq.append({"step": st_, "equal": bool(same), "sha": {k: tensor_bytes_sha(b1[k]) for k in ks}})
    rep["staged_vs_lustre"] = {"batches": len(eq), "all_equal": all(e["equal"] for e in eq), "data_sha_equal": SH_LU == SHAS, "rows": eq}
    print("TIMING staged vs Lustre", {k: v for k, v in rep["staged_vs_lustre"].items() if k != "rows"}, flush=True)
    del C_LU, T_LU
    t = time.time(); save(s_, f"{a.out}/ckpt.pt"); rep["save_s"] = round(time.time() - t, 2)
    rep["ckpt_sha16"] = WS.sha256_file(f"{a.out}/ckpt.pt")[:16]  # the rehearsal must render THIS checkpoint (Astra t1 item 6)
    t = time.time()  # production resume: load the rolling checkpoint, trainable weights and Adam state (Astra t1 item 7)
    _ck = torch.load(f"{a.out}/ckpt.pt", map_location="cpu", weights_only=False); model.load_trainable(_ck["trainable"]); opt.load_state_dict(_ck["opt"])
    del _ck; t_resume = time.time() - t
    # (7) rollout pieces: one 30-step window (5 latents) at 1 and 2 rows, VAE encode / decode, SD encode
    model.eval(); bt = to_dev(make_batch(10 ** 6))
    for rows_ in (1, 2):
        z = torch.randn(rows_, 16, NL, 32, 64, device=dev); ex = torch.cat([bt["guide"][:rows_], bt["src"][:rows_]], 1)
        tk = bt["tok"][:rows_] if bt["tok"] is not None else None; ry = bt["ray"][:rows_] if bt["ray"] is not None else None
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
            for i in range(30):
                model(z, ex, torch.full((rows_,), 1000.0 * (1 - i / 30), device=dev), tk, ry, torch.ones(rows_, dtype=torch.bool, device=dev))
        torch.cuda.synchronize(); rep[f"window30_rows{rows_}_s"] = round(time.time() - t, 2)
    vae.to(dev)
    v17 = torch.rand(1, 3, 17, 256, 512, device=dev) * 2 - 1
    torch.cuda.synchronize(); t = time.time(); zz = WC.wan_encode(vae, v17); torch.cuda.synchronize(); rep["wan_encode_17f_s"] = round(time.time() - t, 3)
    t = time.time(); WC.wan_decode(vae, zz); torch.cuda.synchronize(); rep["wan_decode_5lat_s"] = round(time.time() - t, 3)
    from diffusers import AutoencoderKL
    sdv = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True).to(dev).eval()
    t = time.time()
    with torch.no_grad():
        sdv.encode(v17[0].permute(1, 0, 2, 3)).latent_dist.mode()
    torch.cuda.synchronize(); rep["sd_encode_17f_s"] = round(time.time() - t, 3)
    # (8) Astra t1 item 8: peak incl. staged files since the ALLOCATION start (slurm's job-level sampler + memory.peak + the
    # in-trainer sampler) over the Slurm REQUEST
    CG = WS.cgroup_mem(); SAMP = MEM_SAMPLER.max
    AS = {}
    try:
        AS = json.load(open(os.environ.get("MRV_MEM_SAMPLE", "")))
    except Exception:  # noqa: BLE001
        AS = {}
    t_alloc = float(os.environ.get("MRV_ALLOC_T0") or 0) or None
    from_alloc = bool(AS) and t_alloc is not None and AS.get("t0", 1e18) <= t_alloc + 60 and AS.get("n", 0) > 0
    MR = WP.memory_rule([AS.get("peak"), AS.get("max_current"), CG["peak"], SAMP], MEM_REQUEST)
    rep["memory"] = {"rule": MR, "request_bytes": MEM_REQUEST, "alloc_sampler": AS, "sampled_from_allocation": from_alloc, "cgroup": CG,
                     "trainer_sampled_peak": SAMP, "staged_bytes": STAGE_BYTES, "need_bytes": WS.need_bytes(STAGE_BYTES),
                     "peak_le_85pct": MR["pass"], "peak_over_request": MR["ratio"]}
    print("TIMING memory", {k: v for k, v in rep["memory"].items() if k != "cgroup"}, flush=True)
    # (9) Astra t1 item 7: the selected recipe's production start components (timing-only steps excluded)
    pre = (T_PROC0 - t_alloc - float(os.environ.get("MRV_TIMING_ONLY_S") or 0.0)) if t_alloc else None
    COMP = {"pre_trainer_s": round(pre, 1) if pre is not None else None, "process_to_ready_s": round(T_READY - T_PROC0, 1),
            "frozen_snapshot_s": round(t_snap, 2), "first_update_s": round(t_first, 2), "guard_verify_s": round(t_guard, 2),
            "first_probe_s": rep["probe_s"], "resume_load_s": round(t_resume, 2)}
    rep["startup"] = {**STARTUP, "components": COMP,
                      "startup_s": round(sum(COMP.values()), 1) if all(v is not None for v in COMP.values()) else None,
                      "timing_only_excluded_s": float(os.environ.get("MRV_TIMING_ONLY_S") or 0.0)}
    rep["startup_s"] = rep["startup"]["startup_s"]
    rem = a.budget_hours - LEDGER.hours() - a.reserve_hours
    ups = rem * 3600 / WALL_FC
    rep["forecast"] = {"basis": "provisional (scripts/wan_forecast.py adds the rehearsal rates and every deduction)", "remaining_hours_after_reserve": round(rem, 2),
                       "updates_in_remaining": int(ups), "floor_12k_reachable": ups >= WP.FLOOR_UPDATES, "ledger_by_kind": LEDGER.by_kind()}
    json.dump(rep, open(f"{a.out}/timing.json", "w"), indent=1, default=str)
    print("WAN_TIMING", json.dumps({k: v for k, v in rep.items() if k not in ("sweep", "matched_update")}, default=str)[:4000], flush=True)
    LEDGER.stop(); raise SystemExit(0)


# ------------------------------------------------------------------------------------------------ overfit eval
if a.overfit:
    K_, G_, M_, T_, R_, OF_SPANS = assemble(OF)
    OFB = {"key": K_, "guide": G_, "src": M_, "tok": T_, "ray": R_, "eps": torch.randn(a.overfit, 16, NL, 32, 64, generator=torch.Generator().manual_seed(999)),
           "tkeep": torch.ones(a.overfit, dtype=torch.bool)}

    @torch.no_grad()
    def overfit_eval(step):
        model.eval(); ls = []
        for sv in (0.3, 0.6, 0.9):
            bt = to_dev({**OFB, "sigma": torch.full((a.overfit,), sv)})
            ls.append(torch.cat([flow_loss(bt, slice(i, i + a.micro), reduce=False).cpu() for i in range(0, a.overfit, a.micro)]).mean().item())
        model.train()
        rec = {"update": step, "overfit_eval": [round(x, 5) for x in ls], "mean": round(float(np.mean(ls)), 5)}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if not OF_HIST or OF_HIST[-1][0] != step:
            OF_HIST.append((step, float(np.mean(ls))))  # saved in every checkpoint (pre-registered "still falling" reading)
        return float(np.mean(ls))

# ------------------------------------------------------------------------------------------------ prefetch + loop
end = a.steps if not a.max_steps_this_job else min(a.steps, start + a.max_steps_this_job)
Q = start_producer(start, end)
SAVE = set(int(x) for x in a.save_at.split(","))
T_WIN, N_WIN = time.time(), 0
acc = {"data": 0.0, "gpu": 0.0, "wait": 0.0, "n": 0}
stopped, last_wall = None, None
if a.overfit and start == 0:
    OF0 = overfit_eval(0)  # persisted in every checkpoint: a resumed run keeps the update-0 baseline
import signal as _signal
SIG = {"got": None}


def _on_signal(signum, frame):  # Fable 07:30: time limit / preemption notice -> rolling save at the next update boundary
    SIG["got"] = _signal.Signals(signum).name  # flag only: no I/O in the handler (a reentrant print could lose the save)


for _s in (_signal.SIGTERM, _signal.SIGUSR1):
    _signal.signal(_s, _on_signal)
for step in range(start, end):
    if SIG["got"]:  # atomic rolling checkpoint of every completed update, then a clean exit (resume with --resume)
        if step > start:
            save(step, ck)
        stopped = step
        rec = {"signal_stop": step, "signal": SIG["got"], "ledger_hours": round(LEDGER.hours(), 3)}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        break
    if PROD:  # remaining-hours check BEFORE the update (item 11)
        used = LEDGER.hours(); est = (last_wall or 0.0) / 3600
        if used + est + a.reserve_hours >= a.budget_hours:
            if step > start:
                save(step, ck)
            stopped = step
            rec = {"budget_stop": step, "ledger_hours": round(used, 3), "reserve": a.reserve_hours, "budget_hours": a.budget_hours, "by_kind": LEDGER.by_kind()}
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            break
    tstep = time.time()
    tw = time.time(); bt = qget(Q, step)
    acc["wait"] += time.time() - tw; acc["data"] += bt["data_s"]
    CHAIN = hashlib.sha1((CHAIN + "|" + bt["sha"]).encode()).hexdigest()
    bt = to_dev(bt)
    if step == start:
        SNAP0 = frozen_snapshot()
    loss_v, gn, tfb = update(step, bt)
    note_first_update()
    if step == 1 and a.use_tokens:  # gates opened at update 1: the token path must now carry gradient everywhere (item 5)
        cur = dict(TR)
        pj = [n for n in cur if n.startswith("proj.") and cur[n].grad is not None and cur[n].grad.abs().sum() > 0]
        qk = [i for i in range(len(model.tf.blocks)) if all(
            cur[f"tf.blocks.{i}.attn2.inner.{t}.lora_B.default.weight"].grad is not None and
            cur[f"tf.blocks.{i}.attn2.inner.{t}.lora_B.default.weight"].grad.abs().sum() > 0 for t in ("to_q", "to_k"))]
        n_pj = sum(1 for n in cur if n.startswith("proj."))
        assert len(pj) == n_pj and len(qk) == len(model.tf.blocks), f"token path gradients after the gate opened: projector {len(pj)}/{n_pj}, attn2 q/k LoRA {len(qk)}/30 blocks"
        print(f"gate-open check (update 2): projector {len(pj)}/{n_pj} tensors and attn2 q/k LoRA-B in {len(qk)}/30 blocks have gradients", flush=True)
    opt.step()
    if step == start:  # first-update guard (item 5)
        guard_verify(step, SNAP0)
        del SNAP0
    acc["gpu"] += tfb; acc["n"] += 1
    if (step + 1) % 100 == 0 or step == start or step == end - 1 or TEST:
        gates = [m.gate for m in model.xattn]
        rec = {"update": step + 1, "loss": round(loss_v, 5), "gn": round(gn.item(), 4), "lr": opt.param_groups[0]["lr"],
               "sigma_mean": round(bt["sigma"].mean().item(), 3), "n_start": bt["n_start"], "gdrop": bt["gdrop"], "tdrop": bt["tdrop"],
               "step_s": round(acc["gpu"] / acc["n"], 3), "data_s": round(acc["data"] / acc["n"], 3), "wait_s": round(acc["wait"] / acc["n"], 3),
               "e2e_s": round((time.time() - T_WIN) / max(1, step + 1 - start - N_WIN), 3),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 1),
               "gate_norm": round(sum(g.norm().item() ** 2 for g in gates) ** 0.5, 4), "extra_norm": round(model.tf.patch_embedding.extra.weight.norm().item(), 4),
               "batch_sha": bt["sha"], "pick_chain": CHAIN[:12], "data_sha": DATA_SHA, "manifest_sha": MANIFEST_SHA[:16],
               "ledger_hours": round(LEDGER.hours(), 3), "code_sha": CODE_SHA}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        acc = {"data": 0.0, "gpu": 0.0, "wait": 0.0, "n": 0}; T_WIN, N_WIN = time.time(), step + 1 - start
    if step + 1 in SAVE and not a.overfit:  # numbered milestone FIRST, then the rolling file
        save(step + 1, f"{a.out}/ckpt_{step + 1}.pt")
    if (step + 1) % a.save_every == 0 or step + 1 == end:
        save(step + 1, ck)
    if PROBE is not None and not a.overfit and ((step + 1) % a.val_every == 0 or step == start):
        probe(step + 1)
    if a.overfit and (step + 1) % 100 == 0:
        overfit_eval(step + 1)
    last_wall = time.time() - tstep

# ------------------------------------------------------------------------------------------------ overfit verdict
DONE_STEP = stopped if stopped is not None else end  # a signal / budget stop ends before `end`
if a.overfit and DONE_STEP < OVERFIT_STEPS:
    print(f"OVERFIT_PENDING at update {DONE_STEP} (gates are read at {OVERFIT_STEPS}); resume with --resume", flush=True)
    LEDGER.stop(); raise SystemExit(7)  # non-zero: an afterok chain must not treat a stopped gate as success (Fable 08:30)
if a.overfit:
    OFN = overfit_eval(end)
    from mrv import guide as GD
    vae.to(dev)
    model.eval(); bt = to_dev({**OFB})
    z = torch.randn(a.overfit, 16, NL, 32, 64, generator=torch.Generator().manual_seed(31337)).to(dev)
    sg = WC.sigma_shift(torch.linspace(1.0, 0.0, 31), a.shift).to(dev)
    ex = torch.cat([bt["guide"], bt["src"]], 1)
    with torch.no_grad():  # sampled in the fitting microbatch size (S1b item 6)
        for i0 in range(0, a.overfit, a.micro):
            sl = slice(i0, i0 + a.micro); zz = z[sl]
            for i in range(30):
                with torch.autocast("cuda", dtype=AMP):
                    v = model(zz, ex[sl], (sg[i] * 1000).expand(zz.shape[0]), bt["tok"][sl] if bt["tok"] is not None else None,
                              bt["ray"][sl] if bt["ray"] is not None else None, bt["tkeep"][sl]).float()
                zz = zz + (sg[i + 1] - sg[i]) * v
            z[sl] = zz
    zs = {"hybrid": z}
    for mode in ("tokens_only", "none"):  # controls (Fable 07:30): same 8 clips, same noise, guide removed / both removed
        zm = torch.randn(a.overfit, 16, NL, 32, 64, generator=torch.Generator().manual_seed(31337)).to(dev)
        exm = torch.zeros_like(ex); keep = bt["tkeep"] if mode == "tokens_only" else torch.zeros_like(bt["tkeep"])
        with torch.no_grad():
            for i0 in range(0, a.overfit, a.micro):
                sl = slice(i0, i0 + a.micro); zz = zm[sl]
                for i in range(30):
                    with torch.autocast("cuda", dtype=AMP):
                        v = model(zz, exm[sl], (sg[i] * 1000).expand(zz.shape[0]), bt["tok"][sl] if bt["tok"] is not None else None,
                                  bt["ray"][sl] if bt["ray"] is not None else None, keep[sl]).float()
                    zz = zz + (sg[i + 1] - sg[i]) * v
                zm[sl] = zz
        zs[mode] = zm
    NS = GD.import_signed_scorer(a.out)  # signed scorer namespace: segment / iou_mat / hungarian (unchanged file)
    tp = {"sample": [0, 0, 0], "roundtrip": [0, 0, 0], "tokens_only": [0, 0, 0], "none": [0, 0, 0]}; strips = []
    for b, ((kind, d), (sc, spans)) in enumerate(zip(OF, OF_SPANS)):
        c = C_TR[sc]
        if kind == "mid":  # decode with the stream's true preceding latents as causal context, keep the clip's 20 frames
            st = c["streams"][d[1]]; j0 = d[2]; ctx = st["key"][:, :j0].float().to(dev)[None]
            dec = lambda zc: WC.wan_decode(vae, torch.cat([ctx, zc], 2))[0][:, 4 * j0 - 3:4 * j0 + 4 * NL - 3]
            frames = [i for sp in spans for i in sp]
        else:
            dec = lambda zc: WC.wan_decode(vae, zc)[0]
            frames = [spans[0][0]] + [i for sp in spans[1:] for i in sp]
        xs, xr = dec(z[b:b + 1]), dec(bt["key"][b:b + 1])
        xt_, xn_ = dec(zs["tokens_only"][b:b + 1]), dec(zs["none"][b:b + 1])
        for q, i in enumerate(frames):
            kimg = GD.key_image(a.keys, sc, int(c["ticks"][i]))
            kc = NS["segment"](kimg)[0]
            for name, xx in (("sample", xs), ("roundtrip", xr), ("tokens_only", xt_), ("none", xn_)):
                from PIL import Image
                im = Image.fromarray(((xx[:, q] + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())
                rc = NS["segment"](im)[0]
                mm = NS["hungarian"](NS["iou_mat"](rc, kc), 0.3)
                tp[name][0] += len(mm); tp[name][1] += len(rc) - len(mm); tp[name][2] += len(kc) - len(mm)
                if q == len(frames) // 2 and name == "sample":
                    strips.append(np.concatenate([np.asarray(im), np.asarray(kimg)], 1))
    f1 = {k: 2 * v[0] / max(1, 2 * v[0] + v[1] + v[2]) for k, v in tp.items()}
    loss_ok = OFN <= a.overfit_ratio * OF0
    rt_ok = f1["roundtrip"] > 0 and tp["roundtrip"][0] + tp["roundtrip"][2] > 0  # key cars exist and the codec keeps some
    f1_ok = rt_ok and f1["sample"] >= a.overfit_f1_frac * f1["roundtrip"]
    PRIOR_NR = WP.prior_not_reached(a.gates, GATE_KEY)
    status = WP.overfit_status(loss_ok, f1_ok, rt_ok, f1["sample"], f1["none"], OF_HIST, PRIOR_NR)
    res = {"eval_loss_update0": OF0, "eval_loss_final": OFN, "ratio": OFN / OF0, "declared_ratio": a.overfit_ratio, "loss_pass": loss_ok,
           "car_f1_sample": f1["sample"], "car_f1_roundtrip": f1["roundtrip"], "declared_f1_frac": a.overfit_f1_frac, "f1_pass": f1_ok,
           "car_f1_controls": {"hybrid": f1["sample"], "tokens_only": f1["tokens_only"], "none": f1["none"]},
           "roundtrip_positive": rt_ok, "counts": tp, "eval_history": OF_HIST, "clips": [(k, list(d)) for k, d in OF], "updates": end,
           "declared_updates": OVERFIT_STEPS, "status": status, "PASS": bool(DONE_STEP == OVERFIT_STEPS and status == "PASS"),
           "gate_key": GATE_KEY, "prior_not_reached": PRIOR_NR, "none_margin": WP.NONE_MARGIN,
           "preregistered": "FAIL if hybrid car F1 <= no-inputs car F1 + 0.02 or the round trip has no cars; loss ratio missed while "
                            "placement passes or the eval loss is still falling = NOT_REACHED (rerun longer / lr 3e-4), allowed ONCE per "
                            "gate key (a second one = FAIL); tokens-only is informational"}
    from PIL import Image
    Image.fromarray(np.concatenate(strips, 0)).save(f"{a.out}/overfit_samples.png")
    json.dump(res, open(f"{a.out}/overfit.json", "w"), indent=1)
    WP.gate_append(a.gates, {"kind": "overfit", "gate_key": GATE_KEY, "status": "PASS" if res["PASS"] else status, "out": a.out,
                             "f1": res["car_f1_controls"], "ratio": res["ratio"], "job": os.environ.get("SLURM_JOB_ID")})
    print("OVERFIT_GATE", json.dumps(res), flush=True)
    LEDGER.stop(); raise SystemExit(0 if res["PASS"] else (6 if status.startswith("NOT_REACHED") else 4))
LEDGER.stop()
print(("WAN_TEACHER_SIGNAL_STOP" if SIG["got"] else "WAN_TEACHER_BUDGET_STOP") if stopped is not None else "WAN_TEACHER_DONE",
      stopped if stopped is not None else end)
