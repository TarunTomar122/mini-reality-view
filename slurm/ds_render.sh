#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
# Re-render trained splat scenes from a chase camera, no retraining. usage: slurm/ds_render.sh PROJ CHASE "0 1 2 ..."
PROJ=$1; CHASE=$2; LIST=$3
cd $HOME/mini-reality-view/drivestudio
export TMPDIR=$HOME/mini-reality-view/tmp PYTHONPATH=$(pwd) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/mini-reality-view/envs/ds
NUSC_ROOT=${NUSC_ROOT:-data/nuscenes/raw}; NUSC_VER=${NUSC_VER:-v1.0-mini}
for I in $LIST; do
  CK=$HOME/mini-reality-view/runs/splat/$PROJ/scene$I/checkpoint_final.pth
  [ -f $CK ] || { echo "no ckpt $I"; continue; }
  python $HOME/mini-reality-view/scripts/ds_render_chase.py --ckpt $CK --scene_idx $I --nusc_root $NUSC_ROOT --version $NUSC_VER \
    --chase $CHASE --out $HOME/mini-reality-view/data/targets_splat_${PROJ}_$CHASE
done
echo RENDER_ALL_DONE
