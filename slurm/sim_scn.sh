#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
# Experiment S: convert nuScenes-mini to ScenarioNet scenarios and render a replay test.
cd $HOME/mini-reality-view
PY=envs/sim/bin/python
[ -d data/scn_mini ] || $PY -m scenarionet.convert_nuscenes -d data/scn_mini --split v1.0-mini --dataroot data/nuscenes --num_workers 4 --overwrite
$PY scripts/sim_scenarionet_test.py data/scn_mini runs/sim_scn_test 3
