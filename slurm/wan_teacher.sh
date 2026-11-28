#!/bin/bash
# PLAN-video S1: Wan2.1-T2V-1.3B bidirectional teacher on the H200-3g slice (docs/PLAN-video-impl.md). EVERY mode debits the
# S1 ledger runs/wan_s1_ledger.jsonl (24 h S1 budget incl. timing, bug gate, resume test, panels; Astra item 11).
#   MODE=timing     : trainer timing sweep (global batch 8, microbatch 1/2/4/8, peak alloc / reserved, probe, save, window, VAE),
#                     then one 81-tick dev rollout + the RGB scorer on it, each timed  -> runs/wan_timing_<job>/
#   MODE=overfit    : bug gate: 8 distinct clips, 2,000 updates, fixed-noise eval + decoded placement check (exit 4 = FAIL)
#   MODE=resume_test: 16 updates uninterrupted vs 8 + resume 8; scripts/wan_resume_check.py must pass
#   MODE=train      : the S1 run, runs/wan_s1, resumable; stops when ledger + 2 h panel reserve reaches 24 h
#   MODE=extend     : the one signed 12 h extension (budget 36 h), only after the 24 h panel said EXTEND
# Submit (resubmit the same line to continue after a time limit; --resume picks up runs/wan_s1/ckpt.pt):
#   sbatch -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=12:00:00 -J wan-s1 -o logs2/%x-%j.out \
#          --export=ALL,MODE=train slurm/wan_teacher.sh [extra trainer flags, e.g. --micro 4]
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2
mkdir -p $TMPDIR logs2
PY=.venv/bin/python; MODE=${MODE:-timing}; J=${SLURM_JOB_ID:-manual}; LEDGER=runs/wan_s1_ledger.jsonl
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
sha256sum mrv/wan_cond.py mrv/wan_plan.py scripts/train_wan_teacher.py scripts/wan_rollout.py scripts/score_rgb.py slurm/wan_teacher.sh
echo "start $(date -u) on $(hostname) mode=$MODE"
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
        --ledger $LEDGER --out $O/roll || exit $?
    t1=$(date +%s.%N)
    $PY scripts/score_rgb.py --rgb $O/roll --out $O/m_roll.json || exit $?
    t2=$(date +%s.%N)
    $PY - <<P
import json; j = json.load(open("$O/timing.json"))
j["rollout_81ticks_s"] = round($t1 - $t0, 1); j["score_rgb_81ticks_s"] = round($t2 - $t1, 1)
import torch; r = torch.load("$O/roll/${DRV}_s0.pt", map_location="cpu", weights_only=False); j["rollout_breakdown"] = r["timing"]
json.dump(j, open("$O/timing.json", "w"), indent=1); print("WAN_TIMING_FULL", json.dumps(j))
P
    rc=$? ;;
  overfit)
    $T --out runs/wan_ctest_overfit_$J --overfit 8 --batch 8 --steps 2000 --p_guide_drop 0 --p_tok_drop 0 --p_holes 0 \
       --val_every 0 --save_every 1000 "$@"; rc=$? ;;
  resume_test)
    A=runs/wan_ctest_resA_$J; B=runs/wan_ctest_resB_$J; C="--kind resume_test --steps 16 --save_every 8 --val_every 0 --save_at 16"
    $T --out $A $C "$@" && $T --out $B $C --max_steps_this_job 8 "$@" && $T --out $B $C --resume "$@" \
      && $PY scripts/wan_resume_check.py $A/ckpt.pt $B/ckpt.pt; rc=$? ;;
  train)  $T --out runs/wan_s1 --steps 40000 --budget_hours 24 --reserve_hours 2 --resume "$@"; rc=$? ;;
  extend) $T --out runs/wan_s1 --steps 40000 --budget_hours 36 --reserve_hours 1 --resume "$@"; rc=$? ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac
echo "end $(date -u) rc=$rc"
exit $rc
