#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
# usage: sbatch slurm/sim_render.sh i/n
cd $HOME/mini-reality-view
envs/sim/bin/python scripts/sim_render.py data/scn_full data/nuscenes_full v1.0-trainval data/sim --scenes 100 --every 2 --shard $1 2>&1 | grep -vE "pkg_resources|from pkg"
