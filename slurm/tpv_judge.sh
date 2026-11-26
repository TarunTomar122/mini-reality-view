#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --output=logs/%x-%j.out
# Experiment T scoring: render 30 held-out val drives x 4 keyed frames x 2 seeds (style 2 = game look), then score
# against the exact game keys with both judges. usage: sbatch -J tj-<name> slurm/tpv_judge.sh <name> <ckpt> [--vae]
set -e
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$HOME/mini-reality-view HF_HOME=$HOME/mini-reality-view/hf HF_HUB_OFFLINE=1 TMPDIR=$HOME/mini-reality-view/tmp
N=$1; CK=$2
[ -f runs/judge/$N/index.json ] || .venv/bin/python scripts/sd_render.py --ckpt $CK --out runs/judge/$N --style 2 \
    --keys_like data/sim/keys_tpv --first 0 --n_scenes 30
for j in m2f segformer; do
  [ -f runs/judge/${N}_$j.json ] || .venv/bin/python scripts/judge_pair.py --renders runs/judge/$N --keys data/sim/keys_tpv --judge $j --out runs/judge/${N}_$j.json
done
if [ "$3" = "--vae" ]; then  # ceiling: the true keys through the VAE, scored the same way
  for j in m2f segformer; do
    .venv/bin/python scripts/judge_pair.py --renders runs/judge/$N --keys data/sim/keys_tpv --judge $j --vae --out runs/judge/tpv_ceiling_$j.json
  done
fi
echo TPV_JUDGE_DONE $N
