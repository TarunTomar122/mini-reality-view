#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# drivestudio preprocessing of nuScenes trainval scenes [START, START+NUM) (index in nusc.scene order), 10 Hz.
START=$1; NUM=$2; W=${3:-8}
cd $HOME/mini-reality-view/drivestudio
export TMPDIR=$HOME/mini-reality-view/tmp PYTHONPATH=$(pwd)
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/mini-reality-view/envs/ds
mkdir -p data/nuscenes
[ -e data/nuscenes/raw_full ] || ln -s $HOME/mini-reality-view/data/nuscenes_full data/nuscenes/raw_full
python datasets/preprocess.py --data_root data/nuscenes/raw_full --target_dir data/nuscenes/processed \
  --dataset nuscenes --split v1.0-trainval --start_idx $START --num_scenes $NUM --interpolate_N 4 --workers $W \
  --process_keys images lidar calib dynamic_masks objects
echo PRE_DONE
