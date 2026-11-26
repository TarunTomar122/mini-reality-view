#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
# usage: sbatch slurm/sim_render.sh i/n [skip-list] [scenario dir, default data/scn_pilot] [n scenes, default 100]
# MetaDrive can hang on a scenario: sim_render.py exits after 5 min on one scene (C-level watchdog); the scene keeps its
# STARTED marker and is skipped by the next attempt.
cd $HOME/mini-reality-view
for attempt in $(seq 1 20); do
  timeout 11h envs/sim/bin/python scripts/sim_render.py ${3:-data/scn_pilot} data/nuscenes_full v1.0-trainval data/sim \
      --scenes ${4:-100} --every 2 --shard $1 --skip "${2:-}" 2>&1 | grep --line-buffered -vE "pkg_resources|from pkg"
  rc=${PIPESTATUS[0]}; echo "attempt $attempt rc=$rc"
  [ $rc -eq 0 ] && break
done
echo SIM_RENDER_END
exit $rc  # non-zero if the last attempt failed, so dependent encode jobs do not start on partial data
