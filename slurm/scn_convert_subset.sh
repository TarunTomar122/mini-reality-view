#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
cd $HOME/mini-reality-view
envs/sim/bin/python scripts/scn_convert_subset.py data/scn_pilot 100 2 2>&1 | grep -vE "pkg_resources|from pkg"
