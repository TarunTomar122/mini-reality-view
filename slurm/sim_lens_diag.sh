#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x-%j.out
cd $HOME/mini-reality-view
envs/sim/bin/python scripts/sim_lens_diag.py 2>&1 | grep -vE "pkg_resources|warn"
