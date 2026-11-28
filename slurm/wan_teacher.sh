#!/bin/bash
#SBATCH --signal=B:USR1@300
# PLAN-video S1: Wan2.1-T2V-1.3B bidirectional teacher on the H200-3g slice (docs/PLAN-video-impl.md). EVERY mode debits the
# S1 ledger runs/wan_s1_ledger.jsonl (24 h S1 budget incl. timing, bug gate, resume test, panels; Astra item 11).
#   MODE=timing     : trainer timing sweep (global batch 8, microbatch 1/2/4/8, peak alloc / reserved, probe, save, window, VAE),
#                     then one 81-tick dev rollout + the RGB scorer on it, each timed  -> runs/wan_timing_<job>/
#   MODE=overfit    : bug gate: 8 distinct clips, 2,000 updates, fixed-noise eval + decoded placement check, hybrid + tokens-only
#                     + none samples (exit 0 PASS, 6 NOT_REACHED = rerun longer / lr 3e-4 (once per gate key), 4 FAIL,
#                     7 PENDING = stopped before 2,000: resubmit with --export=ALL,MODE=overfit,OUT=<that run dir>);
#                     verdict appended to runs/wan_gates/gates.jsonl
#   MODE=resume_test: 16 updates uninterrupted vs 8 + resume 8; scripts/wan_resume_check.py must pass
#   MODE=train      : the S1 run, runs/wan_s1, resumable; REFUSES to start without PASS overfit + PASS resume_test records for
#                     its gate key (code / model recipe / base / VAE / guide version) and the bound A@20k replay-parity PASS
#                     (slurm/wan_panel.sh parity); stops when ledger + the MEASURED panel reserve reaches 24 h
#   MODE=extend     : the one signed 12 h extension (budget 36 h); refuses (exit 8) before using the allocation unless
#                     runs/wan_s1/S1_24H_DECISION.json holds an EXTEND verdict
# Submit (resubmit the same line to continue after a time limit; --resume picks up runs/wan_s1/ckpt.pt):
#   sbatch -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=12:00:00 -J wan-s1 -o logs2/%x-%j.out \
#          --export=ALL,MODE=train slurm/wan_teacher.sh [extra trainer flags, e.g. --micro 4]
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2
mkdir -p $TMPDIR logs2
PY=.venv/bin/python; MODE=${MODE:-timing}; J=${SLURM_JOB_ID:-manual}; LEDGER=runs/wan_s1_ledger.jsonl
PANEL_WIN=${PANEL_WIN:-5}  # rollout window size of the S1 panels: the forecast prices it and train / extend reserve for it (S1e 4b)
if [ "$MODE" = extend ]; then  # Astra 08:30: the 12 h extension only on the qualifying 24 h verdict, checked BEFORE any allocation use
  $PY -c "import json,sys;v=json.load(open('runs/wan_s1/S1_24H_DECISION.json'))['verdict'];print('24h marker:',v);sys.exit(0 if v.startswith('EXTEND') else 1)" \
    || { echo "MODE=extend refused: runs/wan_s1/S1_24H_DECISION.json missing or not an EXTEND verdict"; exit 8; }
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
sha256sum mrv/wan_cond.py mrv/wan_plan.py scripts/train_wan_teacher.py scripts/wan_rollout.py scripts/score_rgb.py slurm/wan_teacher.sh
echo "start $(date -u) on $(hostname) mode=$MODE"
# S1b item 10: ONE heartbeat charges the WHOLE allocation (every step below, the timing scorer and the shell included);
# python processes inside only read the ledger (WAN_LEDGER_OUTER). SIGTERM (end / preemption notice) writes the end beat;
# a hard kill leaves no end beat and the ledger then charges the last interval as well.
export WAN_LEDGER_OUTER=1
$PY -m mrv.wan_plan beat $LEDGER alloc_$MODE & BEAT=$!
stop_beat() { kill -TERM $BEAT 2>/dev/null; wait $BEAT 2>/dev/null; }
trap stop_beat EXIT
# Fable 07:30: Slurm sends USR1 to this shell 300 s before the time limit (#SBATCH --signal above; pass it on the sbatch line
# too if you override the script header); it and TERM are forwarded to the running trainer, which saves an atomic rolling
# checkpoint at its next update boundary and exits cleanly (< 1 update lost); resubmit with the same line (--resume).
TPID=
on_sig() {
  if [ -n "$TPID" ] && kill -0 $TPID 2>/dev/null; then echo "forwarding $1 to trainer $TPID"; kill -$1 $TPID
  elif [ "$1" != USR1 ]; then stop_beat; exit 143; fi
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
T="$PY scripts/train_wan_teacher.py --ledger $LEDGER"
case $MODE in
  timing)
    O=runs/wan_timing_$J; mkdir -p $O
    DRV=$($PY -c "import json;print(sorted(json.load(open('data/temporal_lists.json'))['dev'])[0])"); echo $DRV > $O/one.txt
    # S1b item 1: pinned token-geometry check (wrong-camera control) must pass before any timing
    TK=$($PY -c "import torch;g=torch.load('data/guides_v1/$DRV.pt',map_location='cpu',weights_only=False,mmap=True);t=[int(x) for x,h in zip(g['ticks'].tolist(),g['hard'].tolist()) if not h];print(t[len(t)//2])")
    $PY scripts/wan_token_check.py --scene $DRV --tick $TK --out $O/token_geom_${DRV}_t$TK || { echo "TOKEN_GEOM check failed: no timing"; exit 5; }
    $T --out $O --timing --verify_cache 1 "$@" || exit $?
    t0=$(date +%s.%N)
    $PY scripts/wan_rollout.py --ckpt $O/ckpt.pt --split dev --drive_list $O/one.txt --seeds 0 --max_ticks 81 --allow_inert \
        --window_latents $PANEL_WIN --swap_at 50 --ledger $LEDGER --out $O/roll || exit $?
    t1=$(date +%s.%N)
    $PY scripts/score_rgb.py --rgb $O/roll --out $O/m_roll.json || exit $?
    t2=$(date +%s.%N)
    $PY - <<P || exit $?
import json; j = json.load(open("$O/timing.json"))
j["rollout_81ticks_s"] = round($t1 - $t0, 1); j["score_rgb_81ticks_s"] = round($t2 - $t1, 1)
import torch; r = torch.load("$O/roll/${DRV}_s0.pt", map_location="cpu", weights_only=False); j["rollout_breakdown"] = r["timing"]
j["rollout_window_latents"] = int(r["window_latents"])  # Astra S1f 4b
json.dump(j, open("$O/timing.json", "w"), indent=1); print("WAN_TIMING_FULL", json.dumps(j))
P
    # S1b item 5: the forecast is recomputed only now, after the scorer, with every deduction
    $PY scripts/wan_forecast.py --timing_dir $O --rollout_s $(echo "$t1 - $t0" | bc) --score_s $(echo "$t2 - $t1" | bc) --ledger $LEDGER \
        --window_latents $PANEL_WIN --swap_at 50
    rc=$? ;;
  overfit)
    # OUT=<dir of the stopped run> resubmits a preempted / signal-stopped gate in the SAME directory with --resume (Astra 08:30)
    O=${OUT:-runs/wan_ctest_overfit_$J}; RS=; [ -n "$OUT" ] && RS=--resume
    run_fg $T --out $O $RS --overfit 8 --batch 8 --steps 2000 --p_drop_both 0 --p_drop_guide_only 0 --p_drop_tokens_only 0 --p_holes 0 \
       --val_every 0 --save_every 500 "$@"; rc=$? ;;
  resume_test)
    A=runs/wan_ctest_resA_$J; B=runs/wan_ctest_resB_$J; C="--kind resume_test --steps 16 --save_every 8 --val_every 0 --save_at 16"  # 16 vs 8 + 8
    # Astra S1c item 7: leg B must really have saved update 8 (a signal stop at 6 would otherwise test 6 + 10)
    ck8() { $PY -c "import torch,sys;s=torch.load('$B/ckpt.pt',map_location='cpu',weights_only=False,mmap=True)['step'];print('leg B saved step',s);sys.exit(0 if s==8 else 1)"; }
    run_fg $T --out $A $C "$@" && run_fg $T --out $B $C --max_steps_this_job 8 "$@" \
      && { ck8 || { echo "RESUME_TEST invalid: leg B did not stop at update 8"; false; }; } \
      && run_fg $T --out $B $C --resume "$@" \
      && $PY scripts/wan_resume_check.py $A/ckpt.pt $B/ckpt.pt --gates runs/wan_gates/gates.jsonl; rc=$? ;;
  train)  run_fg $T --out runs/wan_s1 --steps 40000 --budget_hours 24 --reserve_panels 1 --panel_window_latents $PANEL_WIN --panel_swap_at 50 --resume "$@"; rc=$? ;;
  extend) run_fg $T --out runs/wan_s1 --steps 40000 --budget_hours 36 --reserve_panels 1 --panel_window_latents $PANEL_WIN --panel_swap_at 50 --resume "$@"; rc=$? ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac
echo "end $(date -u) rc=$rc"
exit $rc
