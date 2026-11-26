#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
# Experiment W step 2a: BEVFormer's temporal info files (nuscenes_infos_temporal_{train,val}.pkl) for trainval.
# BEVFormer expects bev/BEVFormer/data/{nuscenes,can_bus}; we symlink our nuScenes copy. Its create_data also tries
# v1.0-test afterwards (we do not have it) -> that second step fails after the trainval files are written.
cd $HOME/mini-reality-view
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/mini-reality-view/envs/bev
export TMPDIR=$HOME/mini-reality-view/tmp
mkdir -p bev/BEVFormer/data
ln -sfn $HOME/mini-reality-view/data/nuscenes_full bev/BEVFormer/data/nuscenes
ln -sfn $HOME/mini-reality-view/data/nuscenes_full/can_bus bev/BEVFormer/data/can_bus
cd bev/BEVFormer
export PYTHONPATH=$PWD:$PYTHONPATH
# detectron2's editable install puts its own `tools` package first on sys.path -> force BEVFormer's root to the front
python -c "import sys, runpy; sys.path.insert(0, '.'); sys.path.insert(0, 'tools'); sys.argv = ['tools/create_data.py'] + sys.argv[1:]; runpy.run_path('tools/create_data.py', run_name='__main__')" \
    nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data || true
ls -la data/nuscenes/nuscenes_infos_temporal_*.pkl && echo BEV_PREP_DONE
