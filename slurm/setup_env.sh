#!/bin/bash
# Creates ~/mini-reality-view/.venv and pre-downloads the frozen models (login node has internet).
set -e
cd $HOME/mini-reality-view
export PATH=$HOME/.local/bin:$PATH
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; fi
[ -x .venv/bin/python ] || uv venv -q --python 3.11 .venv
uv pip install -q --python .venv/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -q --python .venv/bin/python nuscenes-devkit pyquaternion numpy pillow opencv-python-headless matplotlib tqdm \
  "transformers>=4.45" "diffusers>=0.31" accelerate safetensors einops scipy lpips
export HF_HOME=$HOME/mini-reality-view/hf
.venv/bin/python - <<'EOF'
import torch, nuscenes
from transformers import AutoModel
from diffusers import AutoencoderKL
AutoModel.from_pretrained("facebook/dinov2-small")
AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
print("ENV_OK", torch.__version__, torch.version.cuda)
EOF
