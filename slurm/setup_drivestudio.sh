#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# Stage 6 env: drivestudio (OmniRe / Street Gaussians) in its own conda env with a CUDA 11.8 compiler.
set -e
cd $HOME/mini-reality-view
mkdir -p tmp && export TMPDIR=$HOME/mini-reality-view/tmp PIP_CACHE_DIR=$HOME/mini-reality-view/tmp/pip
source $HOME/miniconda3/etc/profile.d/conda.sh
ENV=$HOME/mini-reality-view/envs/ds
[ -d $ENV ] || conda create -y -q -p $ENV --override-channels -c conda-forge python=3.9
conda activate $ENV
conda install -y -q --override-channels -c nvidia/label/cuda-11.8.0 -c conda-forge cuda-toolkit
conda install -y -q --override-channels -c conda-forge gcc=11 gxx=11  # CUDA 11.8 needs gcc <= 11
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH CC=$ENV/bin/gcc CXX=$ENV/bin/g++ TORCH_CUDA_ARCH_LIST="7.5;8.6;9.0" MAX_JOBS=8
pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -q xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118
[ -d drivestudio ] || git clone -q --recursive https://github.com/ziyc/drivestudio.git
cd drivestudio
grep -v -E "^(--extra-index-url|torch==|torchvision==|xformers==)" requirements.txt > $TMPDIR/ds_req.txt
pip install -q -r $TMPDIR/ds_req.txt nuscenes-devkit "transformers<4.46"
pip install -q --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install -q --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install -q --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
(cd third_party/smplx && pip install -q -e .)
python -c "import torch, gsplat, pytorch3d, nvdiffrast; print('DS_ENV_OK', torch.__version__, gsplat.__version__)"
