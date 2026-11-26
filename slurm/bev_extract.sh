#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --output=logs/%x-%j.out
# Experiment W: BEVFormer features (scripts/bev_extract.py). usage: sbatch slurm/bev_extract.sh <args for bev_extract.py>
cd $HOME/mini-reality-view
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/mini-reality-view/envs/bev
export TMPDIR=$HOME/mini-reality-view/tmp PYTHONUNBUFFERED=1
cd bev/BEVFormer
python ../../scripts/bev_extract.py "$@"
