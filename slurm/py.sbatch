#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# usage: sbatch -J name [-p ICF-Free --gres=gpu:1] slurm/py.sbatch scripts/x.py args...
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$HOME/mini-reality-view HF_HOME=$HOME/mini-reality-view/hf HF_HUB_OFFLINE=1
mkdir -p $HOME/mini-reality-view/tmp; export TMPDIR=$HOME/mini-reality-view/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "start $(date) on $(hostname)"
.venv/bin/python "$@"
rc=$?  # capture before anything else runs (the old line reported the exit code of $(date))
echo "end $(date) rc=$rc"
exit $rc
