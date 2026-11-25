#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# Convert nuScenes v1.0-trainval to ScenarioNet scenarios (for sim replays).
cd $HOME/mini-reality-view
envs/sim/bin/python -m scenarionet.convert_nuscenes -d data/scn_full --split v1.0-trainval --dataroot data/nuscenes_full --num_workers 8 --overwrite
ls data/scn_full | head; echo SCN_CONVERT_DONE
