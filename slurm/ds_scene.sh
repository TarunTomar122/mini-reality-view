#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# Stage 6, one scene: train Street Gaussians (drivestudio) on nuScenes-mini scene $1, then render our chase camera.
# usage: sbatch -J ds-scene-3 --gres=gpu:nvidia_geforce_rtx_2080_ti:1 slurm/ds_scene.sh 3 [iters] [project] [extra overrides...]
I=$1; ITERS=${2:-30000}; PROJ=${3:-mini}
cd $HOME/mini-reality-view/drivestudio
export TMPDIR=$HOME/mini-reality-view/tmp PYTHONPATH=$(pwd) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/mini-reality-view/envs/ds
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
OUT=$HOME/mini-reality-view/runs/splat
DSROOT=${DSROOT:-data/nuscenes/processed_10Hz/mini}; NUSC_ROOT=${NUSC_ROOT:-data/nuscenes/raw}; NUSC_VER=${NUSC_VER:-v1.0-mini}
if [ ! -f $OUT/$PROJ/scene$I/checkpoint_final.pth ]; then
python tools/train.py --config_file configs/paper_legacy/streetgs.yaml --output_root $OUT --project $PROJ --run_name scene$I \
  dataset=nuscenes/6cams data.data_root=$DSROOT data.scene_idx=$I \
  data.start_timestep=0 data.end_timestep=-1 data.pixel_source.load_smpl=False trainer.optim.num_iters=$ITERS \
  logging.vis_freq=100000 logging.saveckpt_freq=100000 "${@:4}"
fi
python $HOME/mini-reality-view/scripts/ds_render_chase.py --ckpt $OUT/$PROJ/scene$I/checkpoint_final.pth --scene_idx $I --nusc_root $NUSC_ROOT --version $NUSC_VER \
  --out $HOME/mini-reality-view/data/targets_splat_$PROJ
echo "end $(date)"
