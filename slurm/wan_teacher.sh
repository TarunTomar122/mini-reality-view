#!/bin/bash
#SBATCH --signal=B:USR1@300
export MRV_ALLOC_T0=$(date +%s.%N)  # start-up is measured from here (allocation) to the trainer's first update (CONVERGED item 7)
# PLAN-video S1: Wan2.1-T2V-1.3B bidirectional teacher on the H200-3g slice (docs/PLAN-video-impl.md). EVERY mode debits the
# S1 ledger runs/wan_s1_ledger.jsonl (24 h S1 budget incl. timing, bug gate, resume test, panels; Astra item 11).
#   MODE=timing     : the re-time (council/s1-timing-r1-CONVERGED.md item 7): token geometry check; trainer timing from the staged
#                     data (grad_ckpt {1,0} x micro {1,2,4,8} sweep + matched-update check -> selected setting, --use_tokens 0
#                     diagnostic, >= 200 production updates in 32-update blocks, staged == Lustre batches, cgroup peak memory);
#                     panel-loop rehearsal (load-once render byte-equal to a fresh load; load-once rollout rate); forecast with
#                     ONE panel + retime_pass  -> runs/wan_timing_<job>/timing.json (scoring runs on the 2080 Ti, not here)
#   MODE=overfit    : bug gate: 8 distinct clips, 2,000 updates, fixed-noise eval + decoded placement check, hybrid + tokens-only
#                     + none samples (exit 0 PASS, 6 NOT_REACHED = rerun longer / lr 3e-4 (once per gate key), 4 FAIL,
#                     7 PENDING = stopped before 2,000: resubmit with --export=ALL,MODE=overfit,OUT=<that run dir>);
#                     verdict appended to runs/wan_gates/gates.jsonl
#   MODE=resume_test: 16 updates uninterrupted vs 8 + resume 8; scripts/wan_resume_check.py must pass
#   MODE=train      : the S1 run, runs/wan_s1, resumable; REFUSES to start without PASS overfit + PASS resume_test records for
#                     its gate key (code / model recipe / base / VAE / guide version) and the bound A@20k replay-parity PASS
#                     (2080 Ti scorer parity: slurm/wan_score.sh parity + parity_check); stops when ledger + ONE MEASURED render-only panel reaches 24 h
#   MODE=extend     : the one signed 12 h extension (budget 36 h); refuses (exit 8) before using the allocation unless
#                     runs/wan_s1/S1_24H_DECISION.json holds an EXTEND verdict
# Every mode needs (CONVERGED items 2, 3, 5):
#   MANIFEST=<read-only manifest of scripts/verify_manifest.py> (default data/wan_manifest_v1.json; its sha enters the gate key)
#   MICRO=<n> GRAD_CKPT=<0|1> (explicit in every mode, recorded in the recipe; overfit / resume_test / train / extend must use the
#     re-time's selected_setting, and refuse unless timing.json retime_pass passed every rule)
# Staging: the trainer copies the token + latent caches it uses and the Wan weights to /dev/shm/mrv_<job> (8 streams, hashed on
# copy against the manifest); the EXIT trap below removes it. Files in /dev/shm count against the job's memory cgroup:
#   --mem = staged bytes + 48 GB = $(.venv/bin/python -m mrv.wan_stage need --manifest data/wan_manifest_v1.json)   [train / timing]
#           ($(... need --manifest ... --no_dev) for overfit / resume_test, which open no dev drives); with ~86 GB of tokens
#           (150 drives) + ~2.4 GB latents + the Wan folder this is about 145G. The script refuses (exit 9) when it can read a
#           cgroup limit below that.
# Submit (resubmit the same line to continue after a time limit; --resume picks up runs/wan_s1/ckpt.pt):
#   M=$(.venv/bin/python -m mrv.wan_stage need --manifest data/wan_manifest_v1.json)
#   sbatch -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=$M --time=2:00:00 -J wan-timing -o logs2/%x-%j.out \
#          --export=ALL,MODE=timing,MICRO=1,GRAD_CKPT=1 slurm/wan_teacher.sh                      # re-time, 144 drives, cold
#   sbatch -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=$M --time=12:00:00 -J wan-s1 -o logs2/%x-%j.out \
#          --export=ALL,MODE=train,MICRO=<selected>,GRAD_CKPT=<selected> slurm/wan_teacher.sh
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2
mkdir -p $TMPDIR logs2
PY=.venv/bin/python; MODE=${MODE:-timing}; J=${SLURM_JOB_ID:-manual}; LEDGER=runs/wan_s1_ledger.jsonl
PANEL_WIN=${PANEL_WIN:-5}  # rollout window size of the S1 panels: the forecast prices it and train / extend reserve for it (S1e 4b)
MANIFEST=${MANIFEST:-data/wan_manifest_v1.json}
[ -n "$MICRO" ] && [ -n "$GRAD_CKPT" ] || { echo "MICRO=<n> and GRAD_CKPT=<0|1> must be exported explicitly (F5; e.g. --export=ALL,MODE=$MODE,MICRO=1,GRAD_CKPT=1)"; exit 2; }
[ -s "$MANIFEST" ] || { echo "MANIFEST $MANIFEST missing: run scripts/verify_manifest.py (CPU job) first"; exit 2; }
STAGE=/dev/shm/mrv_$J; GEOM=/dev/shm/mrv_${J}_geom
# Astra t1 item 8: job-level memory sampler from the allocation start (staged tmpfs pages, copy workers, shell, trainer)
export MRV_MEM_SAMPLE=$TMPDIR/mem_$J.json
$PY -m mrv.wan_stage sample --out $MRV_MEM_SAMPLE & MEMS=$!
trap 'kill $MEMS 2>/dev/null' EXIT  # replaced by cleanup below once the heartbeat runs
if [ "$MODE" = extend ]; then  # Astra 08:30: the 12 h extension only on the qualifying 24 h verdict, checked BEFORE any allocation use
  $PY -c "import json,sys;v=json.load(open('runs/wan_s1/S1_24H_DECISION.json'))['verdict'];print('24h marker:',v);sys.exit(0 if v.startswith('EXTEND') else 1)" \
    || { echo "MODE=extend refused: runs/wan_s1/S1_24H_DECISION.json missing or not an EXTEND verdict"; exit 8; }
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
sha256sum mrv/wan_cond.py mrv/wan_plan.py mrv/wan_stage.py scripts/train_wan_teacher.py scripts/wan_rollout.py scripts/wan_panel.py slurm/wan_teacher.sh $MANIFEST
echo "start $(date -u) on $(hostname) mode=$MODE micro=$MICRO grad_ckpt=$GRAD_CKPT manifest=$MANIFEST stage=$STAGE"
# F1: staged files count against this job's memory cgroup: refuse before any work if the limit cannot hold them + 48 GB
ND=; case $MODE in overfit|resume_test) ND=--no_dev ;; esac
$PY -m mrv.wan_stage memcheck --manifest $MANIFEST $ND || { echo "memory request too small for the stage-in (see MEMCHECK above)"; exit 9; }
# S1b item 10: ONE heartbeat charges the WHOLE allocation (every step below, the timing scorer and the shell included);
# python processes inside only read the ledger (WAN_LEDGER_OUTER). SIGTERM (end / preemption notice) writes the end beat;
# a hard kill leaves no end beat and the ledger then charges the last interval as well.
export WAN_LEDGER_OUTER=1
$PY -m mrv.wan_plan beat $LEDGER alloc_$MODE & BEAT=$!
stop_beat() { kill -TERM $BEAT 2>/dev/null; wait $BEAT 2>/dev/null; }
cleanup() { stop_beat; kill $MEMS 2>/dev/null; [ -n "$STAGE" ] && rm -rf "$STAGE" "$GEOM"; }  # job-local stage dirs never outlive the allocation
trap cleanup EXIT
# Fable 07:30: Slurm sends USR1 to this shell 300 s before the time limit (#SBATCH --signal above; pass it on the sbatch line
# too if you override the script header); it and TERM are forwarded to the running trainer, which saves an atomic rolling
# checkpoint at its next update boundary and exits cleanly (< 1 update lost); resubmit with the same line (--resume).
TPID=
on_sig() {
  if [ -n "$TPID" ] && kill -0 $TPID 2>/dev/null; then echo "forwarding $1 to trainer $TPID"; kill -$1 $TPID
  elif [ "$1" != USR1 ]; then cleanup; exit 143; fi
}
trap 'on_sig USR1' USR1
trap 'on_sig TERM' TERM
trap 'on_sig INT' INT
run_fg() {  # run in the background and wait, so the shell can forward signals while the trainer runs
  "$@" & TPID=$!; local rc
  while :; do wait $TPID; rc=$?; kill -0 $TPID 2>/dev/null || break; done
  TPID=; return $rc
}
$PY scripts/wan_tests.py > /dev/null || { echo "WAN_TESTS failed"; exit 3; }
T="$PY scripts/train_wan_teacher.py --ledger $LEDGER --manifest $MANIFEST --micro $MICRO --grad_ckpt $GRAD_CKPT --stage_dir $STAGE --keep_stage 1"
case $MODE in
  timing)
    O=runs/wan_timing_$J; mkdir -p $O
    G0=$(date +%s.%N)  # timing-only steps (geometry) are excluded from the priced production start-up (Astra t1 item 7)
    DRV=$($PY -c "import json;print(sorted(json.load(open('data/temporal_lists.json'))['dev'])[0])"); echo $DRV > $O/one.txt
    # S1b item 1 + Astra t1 item 9: the pinned token-geometry check opens ONLY hashed copies of the guide / token files (manifest)
    $PY -m mrv.wan_stage copy --manifest $MANIFEST --dst $GEOM data/guides_v1/$DRV.pt data/wan_tok_v1/$DRV.pt || { echo "TOKEN_GEOM inputs differ from the manifest"; exit 5; }
    TK=$($PY -c "import torch;g=torch.load('$GEOM/guides_v1/$DRV.pt',map_location='cpu',weights_only=False,mmap=True);t=[int(x) for x,h in zip(g['ticks'].tolist(),g['hard'].tolist()) if not h];print(t[len(t)//2])")
    $PY scripts/wan_token_check.py --scene $DRV --tick $TK --guides $GEOM/guides_v1 --tokens $GEOM/wan_tok_v1 --out $O/token_geom_${DRV}_t$TK \
      || { echo "TOKEN_GEOM check failed: no timing"; exit 5; }
    rm -rf "$GEOM"
    export MRV_TIMING_ONLY_S=$(echo "$(date +%s.%N) - $G0" | bc)
    run_fg $T --out $O --timing "$@" || exit $?
    rm -rf "$STAGE"  # the rehearsal reads its own inputs; free the staged bytes before the rollouts
    # F3 rehearsal: one receiver run from the load-once panel loop byte-equal to a fresh-load render; load-once rollout rate
    $PY scripts/wan_panel.py --rehearsal --ckpt $O/ckpt.pt --out $O/rehearsal --ledger $LEDGER --rollout_args "--window_latents $PANEL_WIN"
    REH=$?; echo "rehearsal rc=$REH"
    $PY - <<P || exit $?
import json; j = json.load(open("$O/timing.json"))
j["token_geometry"] = {"pass": True, "manifest_verified": True, "dir": "$O/token_geom_${DRV}_t$TK",
                      "note": "wan_token_check.py exits 5 on failure (no timing); it read hash-on-copy staged guide / token files"}
json.dump(j, open("$O/timing.json", "w"), indent=1)
P
    # CONVERGED item 7: forecast with ONE panel (render-only, load-once rate) + every re-time rule -> retime_pass
    $PY scripts/wan_forecast.py --timing_dir $O --ledger $LEDGER --window_latents $PANEL_WIN --swap_at 50
    rc=$?
    [ $REH -ne 0 ] && { echo "REHEARSAL FAILED (rc $REH): the timing job fails"; rc=$REH; } ;;
  overfit)
    # OUT=<dir of the stopped run> resubmits a preempted / signal-stopped gate in the SAME directory with --resume (Astra 08:30)
    O=${OUT:-runs/wan_ctest_overfit_$J}; RS=; [ -n "$OUT" ] && RS=--resume
    run_fg $T --out $O $RS --overfit 8 --batch 8 --steps 2000 --p_drop_both 0 --p_drop_guide_only 0 --p_drop_tokens_only 0 --p_holes 0 \
       --val_every 0 --save_every 500 "$@"; rc=$? ;;
  resume_test)
    A=runs/wan_ctest_resA_$J; B=runs/wan_ctest_resB_$J; C="--resume_test --kind resume_test --steps 16 --save_every 8 --val_every 0 --save_at 16"  # 16 vs 8 + 8
    # Astra S1c item 7: leg B must really have saved update 8 (a signal stop at 6 would otherwise test 6 + 10)
    ck8() { $PY -c "import torch,sys;s=torch.load('$B/ckpt.pt',map_location='cpu',weights_only=False,mmap=True)['step'];print('leg B saved step',s);sys.exit(0 if s==8 else 1)"; }
    run_fg $T --out $A $C "$@" && run_fg $T --out $B $C --max_steps_this_job 8 "$@" \
      && { ck8 || { echo "RESUME_TEST invalid: leg B did not stop at update 8"; false; }; } \
      && run_fg $T --out $B $C --resume "$@" \
      && $PY scripts/wan_resume_check.py $A/ckpt.pt $B/ckpt.pt --gates runs/wan_gates/gates.jsonl; rc=$? ;;
  train)  run_fg $T --out runs/wan_s1 --steps 40000 --budget_hours 24 --reserve_panels 1 --panel_window_latents $PANEL_WIN --panel_swap_at 50 --resume "$@" --production; rc=$? ;;
  extend) run_fg $T --out runs/wan_s1 --steps 40000 --budget_hours 36 --reserve_panels 1 --panel_window_latents $PANEL_WIN --panel_swap_at 50 --resume "$@" --production; rc=$? ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac
echo "end $(date -u) rc=$rc"
exit $rc
