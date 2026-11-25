#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# drivestudio preprocessing of nuScenes mini (10 Hz with 4 interpolated frames between keyframes).
cd $HOME/mini-reality-view/drivestudio
export TMPDIR=$HOME/mini-reality-view/tmp PYTHONPATH=$(pwd)
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/mini-reality-view/envs/ds
mkdir -p data/nuscenes
[ -e data/nuscenes/raw ] || ln -s $HOME/mini-reality-view/data/nuscenes data/nuscenes/raw
python datasets/preprocess.py --data_root data/nuscenes/raw --target_dir data/nuscenes/processed \
  --dataset nuscenes --split v1.0-mini --start_idx 0 --num_scenes 10 --interpolate_N 4 --workers 8 \
  --process_keys images lidar calib dynamic_masks objects
ls data/nuscenes/processed_10Hz/mini | head; ls data/nuscenes/processed_10Hz/mini/000
echo PRE_DONE
