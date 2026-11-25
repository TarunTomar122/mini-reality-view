#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
# Finish the drivestudio env: nvdiffrast (ptxas crashed with 8 parallel jobs) + smplx, then import check.
cd $HOME/mini-reality-view
export TMPDIR=$HOME/mini-reality-view/tmp PIP_CACHE_DIR=$HOME/mini-reality-view/tmp/pip
source $HOME/miniconda3/etc/profile.d/conda.sh
ENV=$HOME/mini-reality-view/envs/ds
conda activate $ENV
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH CC=$ENV/bin/gcc CXX=$ENV/bin/g++ MAX_JOBS=1
for ARCH in "7.5;8.6;9.0" "7.5;8.6"; do
  export TORCH_CUDA_ARCH_LIST="$ARCH"
  echo "try nvdiffrast arch $ARCH"
  pip install -q --no-build-isolation git+https://github.com/NVlabs/nvdiffrast && break
done
(cd drivestudio/third_party/smplx && pip install -q -e .)
cd drivestudio && PYTHONPATH=. python -c "import torch, gsplat, pytorch3d, nvdiffrast.torch; import models.modules; print('DS_ENV_OK', torch.__version__, gsplat.__version__, '$TORCH_CUDA_ARCH_LIST')"
