#!/bin/bash
# PLAN-video S1: Wan2.1-T2V-1.3B bidirectional teacher on the H200-3g slice (docs/PLAN-video-impl.md).
#   MODE=timing  : 30 updates into runs/wan_timing_<job> (throwaway; logs step_s / data_s / peak GB every update)
#   MODE=overfit : bug gate, 8 fixed clips, 2,000 updates into runs/wan_ctest_overfit_<job> (loss must collapse)
#   MODE=train   : the S1 run, runs/wan_s1, resumable; stops at the 24 h budget (cumulative over all jobs)
#   MODE=extend  : the one signed 12 h extension (budget 36 h), same run, only after the council reads the 24 h panel
# Submit (one slice, 8 CPUs; resubmit the same line to continue after a time limit: --resume picks up runs/wan_s1/ckpt.pt):
#   sbatch -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=12:00:00 -J wan-s1 -o logs2/%x-%j.out \
#          --export=ALL,MODE=train slurm/wan_teacher.sh
# Extra trainer flags can be appended after the script name (e.g. --micro 4 if batch 8 does not fit).
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2
mkdir -p $TMPDIR logs2
PY=.venv/bin/python; MODE=${MODE:-timing}; J=${SLURM_JOB_ID:-manual}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
sha256sum mrv/wan_cond.py scripts/train_wan_teacher.py slurm/wan_teacher.sh
echo "start $(date -u) on $(hostname) mode=$MODE"
case $MODE in
  timing)  $PY scripts/train_wan_teacher.py --out runs/wan_timing_$J --timing --steps 30 --val_every 10 --verify_cache 0 "$@" ;;
  overfit) $PY scripts/train_wan_teacher.py --out runs/wan_ctest_overfit_$J --timing --overfit 8 --steps 2000 --p_guide_drop 0 --p_tok_drop 0 \
               --p_holes 0 --val_every 500 "$@" ;;
  train)   $PY scripts/train_wan_teacher.py --out runs/wan_s1 --steps 40000 --budget_hours 24 --resume "$@" ;;
  extend)  $PY scripts/train_wan_teacher.py --out runs/wan_s1 --steps 40000 --budget_hours 36 --resume "$@" ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac
rc=$?
echo "end $(date -u) rc=$rc"
exit $rc
