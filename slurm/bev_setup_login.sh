#!/bin/bash
# Experiment W step 1a (LOGIN NODE: downloads only, no compiling): env for BEVFormer (council/PLAN-bev.md).
# torch 1.10.2+cu113 (BEVFormer is known to run on it; torch 1.9 rejects a non-matching nvcc), mmcv-full 1.4.0
# prebuilt wheel, CUDA 11.3 toolkit + gcc 10 for building mmdet3d 0.17.1 ops later on a GPU node.
set -e
cd $HOME/mini-reality-view
mkdir -p tmp && export TMPDIR=$HOME/mini-reality-view/tmp PIP_CACHE_DIR=$HOME/mini-reality-view/tmp/pip
source $HOME/miniconda3/etc/profile.d/conda.sh
ENV=$HOME/mini-reality-view/envs/bev
[ -d $ENV ] || conda create -y -q -p $ENV --override-channels -c conda-forge python=3.8
conda activate $ENV
conda install -y -q --override-channels -c nvidia/label/cuda-11.3.1 -c conda-forge cuda-toolkit gcc=10 gxx=10
pip install -q torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -q mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install --timeout 60 --retries 5 --progress-bar off mmdet==2.14.0 mmsegmentation==0.14.1 "numpy<1.20" "numba==0.53.0" "nuscenes-devkit==1.1.9" \
    "timm<0.6" "einops" "fvcore" "iopath" "setuptools<60" "yapf==0.40.1" "lyft_dataset_sdk" "networkx<2.6" \
    "trimesh==2.35.39" "scikit-image<0.20" "plyfile" "tensorboard" "pandas<2"
mkdir -p bev && cd bev
[ -d BEVFormer ] || git clone -q https://github.com/fundamentalvision/BEVFormer.git
[ -d mmdetection3d ] || git clone -q -b v0.17.1 https://github.com/open-mmlab/mmdetection3d.git
[ -d detectron2 ] || git clone -q https://github.com/facebookresearch/detectron2.git
mkdir -p ckpts
[ -s ckpts/bevformer_r101_dcn_24ep.pth ] || curl -sSL --retry 5 -o ckpts/bevformer_r101_dcn_24ep.pth https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth
[ -s ckpts/r101_dcn_fcos3d_pretrain.pth ] || curl -sSL --retry 5 -o ckpts/r101_dcn_fcos3d_pretrain.pth https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth || true
cd $HOME/mini-reality-view
[ -d data/nuscenes_full/can_bus ] || { curl -sSL --retry 5 -o tmp/can_bus.zip https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip && unzip -q tmp/can_bus.zip -d data/nuscenes_full/ && rm tmp/can_bus.zip; }
python -c "import torch, mmcv; print('BEV_LOGIN_OK', torch.__version__, mmcv.__version__)"
ls -la bev/ckpts data/nuscenes_full/can_bus | head -5
