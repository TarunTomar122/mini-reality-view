#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --output=logs/%x-%j.out
# Experiment T early look: 3 fixed held-out drives (busy junction scene-0915, moving traffic scene-0563, quiet street
# scene-0632) x 3 keyed frames x seed 0 (same noise for every checkpoint), for E1/E2/E3 at one snapshot, plus E1 with
# the cameras of another of the three drives (wrong street). usage: sbatch slurm/tpv_preview.sh <step>
set -e
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$HOME/mini-reality-view HF_HOME=$HOME/mini-reality-view/hf HF_HUB_OFFLINE=1 TMPDIR=$HOME/mini-reality-view/tmp
S=$1; SC=scene-0915,scene-0563,scene-0632
R="--style 2 --keys_like data/sim/keys_tpv --scenes $SC --frames 3 --seeds 0"
for e in E1 E2 E3; do
  .venv/bin/python scripts/sd_render.py --ckpt runs/tpv$e/ckpt_$S.pt --out runs/preview/tpv${e}_$S $R
done
.venv/bin/python scripts/sd_render.py --ckpt runs/tpvE1/ckpt_$S.pt --out runs/preview/tpvE1_${S}_swap $R --control swap_scene
echo PREVIEW_DONE $S
