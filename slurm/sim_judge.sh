#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --output=logs/%x-%j.out
# Sim pilot scoring (council/PLAN-sim.md step 4): render one arm snapshot on the judge set, score with both judges.
# usage: sbatch -J judge-<name> slurm/sim_judge.sh <name> <ckpt>
set -e
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$HOME/mini-reality-view HF_HOME=$HOME/mini-reality-view/hf HF_HUB_OFFLINE=1 TMPDIR=$HOME/mini-reality-view/tmp
N=$1; CK=$2
[ -f runs/judge/$N/index.json ] || .venv/bin/python scripts/sd_render.py --ckpt $CK --out runs/judge/$N
for j in m2f segformer; do
  [ -f runs/judge/${N}_$j.json ] || .venv/bin/python scripts/judge_score.py --renders runs/judge/$N --judge $j --shift 0 --out runs/judge/${N}_$j.json
done
echo SIM_JUDGE_DONE $N
