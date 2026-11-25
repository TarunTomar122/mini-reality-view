#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
# Experiment S: MetaDrive in its own venv + headless render test.
cd $HOME/mini-reality-view
export PATH=$HOME/.local/bin:$PATH TMPDIR=$HOME/mini-reality-view/tmp
[ -x envs/sim/bin/python ] || uv venv -q --python 3.10 envs/sim
uv pip install -q --python envs/sim/bin/python metadrive-simulator pillow numpy
nvidia-smi --query-gpu=name --format=csv,noheader
envs/sim/bin/python -c "import metadrive, panda3d; print('metadrive', metadrive.__version__)"
envs/sim/bin/python scripts/sim_metadrive_test.py runs/sim_test
