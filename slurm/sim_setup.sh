#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
# Experiment S: headless MetaDrive render test (env built on the login node: envs/sim, Python 3.11).
cd $HOME/mini-reality-view
nvidia-smi --query-gpu=name --format=csv,noheader
envs/sim/bin/python -c "import metadrive, panda3d; print('metadrive import ok')"
envs/sim/bin/python scripts/sim_metadrive_test.py runs/sim_test
