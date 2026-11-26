#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --output=logs/%x-%j.out
# Experiment W step 1b (GPU node): build mmdet3d 0.17.1 + detectron2 ops in envs/bev, then an import + CUDA check.
set -e
# conda-forge gcc 10 pulled a sysroot with Linux 6.12 headers; nvcc 11.3 cannot parse their __s128 typedef,
# so envs/bev pins sysroot_linux-64=2.17
cd $HOME/mini-reality-view
export TMPDIR=$HOME/mini-reality-view/tmp PIP_CACHE_DIR=$HOME/mini-reality-view/tmp/pip
source $HOME/miniconda3/etc/profile.d/conda.sh
ENV=$HOME/mini-reality-view/envs/bev
conda activate $ENV
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH CC=$ENV/bin/gcc CXX=$ENV/bin/g++ TORCH_CUDA_ARCH_LIST="7.5;8.6" MAX_JOBS=4 FORCE_CUDA=1
nvcc --version | tail -1
python -c 'import mmdet3d' 2>/dev/null || (cd bev/mmdetection3d && rm -rf build && pip install -q --no-build-isolation -e .)
cd bev
# detectron2 master needs torch >= 1.11 (ATen/cuda/Atomic.cuh); v0.6 supports torch 1.10
cd detectron2 && git checkout -q v0.6 && rm -rf build && pip install -q --no-build-isolation -e . && cd ..
cd $HOME/mini-reality-view
python -c "import torch, mmcv, mmdet, mmdet3d, detectron2; from mmcv.ops import MultiScaleDeformableAttention; print('BEV_BUILD_OK', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), mmdet3d.__version__)"
