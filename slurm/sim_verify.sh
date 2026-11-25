#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
cd $HOME/mini-reality-view
envs/sim/bin/python scripts/sim_verify.py data/scn_mini data/nuscenes v1.0-mini runs/sim_verify 5
