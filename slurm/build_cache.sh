#!/bin/bash
# A5 vmix cache: submit N drive shards of scripts/build_cache.py on 2080 Tis (damnii10 excluded: broken GPU), then ONE
# scripts/cache_check.py job (also on a 2080 Ti, the cache's GPU type / fp16) that runs after all shards succeed.
# Submission helper only (run it by hand on the login node). Shards beyond ACTIVE are submitted held (-H; job cap: release with
# scontrol release <id> as slots free up).
#   bash slurm/build_cache.sh [N=10] [ACTIVE=N] [BATCH=8]
cd $HOME/mini-reality-view
N=${1:-10}; ACTIVE=${2:-$N}; BATCH=${3:-8}
OUT=runs/cache/A20k_off_rho1_train; A=runs/longA/ckpt_20000.pt
G="--gres=gpu:nvidia_geforce_rtx_2080_ti:1 --exclude=damnii10 --cpus-per-task=4 --mem=48G"
mkdir -p $OUT logs2
IDS=""
for i in $(seq 0 $((N - 1))); do
  H=""; [ $i -ge $ACTIVE ] && H="-H"
  J=$(sbatch --parsable $H $G --time=8:00:00 -J cache$i -o logs2/%x-%j.out \
      slurm/py.sbatch scripts/build_cache.py --ckpt $A --split train --seed 0 --batch $BATCH --shard $i/$N --out $OUT) || { echo "sbatch failed at shard $i"; exit 1; }
  echo "shard $i/$N: job $J ${H:+(held)}"; IDS="$IDS:$J"
done
C=$(sbatch --parsable -d afterok$IDS $G --time=2:00:00 -J cachecheck -o logs2/%x-%j.out \
    slurm/py.sbatch scripts/cache_check.py --cache $OUT --ckpt $A --out runs/cache/check.json) || { echo "sbatch cache_check failed"; exit 1; }
echo "cache_check: job $C (after shards$IDS) -> runs/cache/check.json"
