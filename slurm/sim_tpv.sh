#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --exclude=damnii07
#SBATCH --output=logs/%x-%j.out
# Third-person game answer keys only (the real photos are the inputs). usage: sbatch slurm/sim_tpv.sh <shard i/n> <scenario dir> <n scenes> <train|val>
# MetaDrive can hang: sim_render.py exits after 5 min on one scene (C watchdog); the scene is skipped on the next attempt.
# damnii07 hangs on every scene -> excluded.
cd $HOME/mini-reality-view
for attempt in $(seq 1 25); do
  envs/sim/bin/python scripts/sim_render.py $2 data/nuscenes_full v1.0-trainval data/sim --scenes $3 --every 2 --shard $1 \
      --view tpv --keys_name keys_tpv --keys_only --split $4 2>&1 | grep --line-buffered -vE "pkg_resources|from pkg|libEGL|^$"
  rc=${PIPESTATUS[0]}; echo "attempt $attempt rc=$rc"
  [ $rc -eq 0 ] && break
done
echo SIM_TPV_END
exit 0
