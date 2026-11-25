#!/bin/bash
# One processing round on the full dataset: targets for every downloaded scene (8 CPU shards),
# then tokens+latents (4 GPU shards). Idempotent: finished scenes are skipped. Rerun as parts arrive.
cd $HOME/mini-reality-view
S="sbatch --parsable"
A="--dataroot data/nuscenes_full --version v1.0-trainval"
deps=""
for i in 0 1 2 3 4 5 6 7; do
  j=$($S -J full-targets-$i --time=8:00:00 --mem=32G slurm/py.sbatch scripts/make_targets.py $A --out data/targets_full --shard $i/8)
  deps="$deps:$j"
done
for i in 0 1 2 3; do
  $S -J full-encode-$i --gres=gpu:1 --time=8:00:00 --dependency=afterany$deps slurm/py.sbatch scripts/encode.py $A --targets data/targets_full --out data/enc_full --pool 2 --shard $i/4
done
