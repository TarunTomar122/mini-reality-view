#!/bin/bash
# Per-scene stage-6 chain for trainval scenes: preprocess (1 worker) -> sky masks -> splat (15k iters) -> chase render.
# Skips preprocessing for scenes that are already complete. usage: bash slurm/ds_tv_chain.sh 0 69
cd $HOME/mini-reality-view
R=drivestudio/data/nuscenes/processed_10Hz/trainval
for i in $(seq $1 $2); do
  d=$R/$(printf %03d $i)
  ni=$(ls $d/images 2>/dev/null | wc -l); nl=$(ls $d/lidar 2>/dev/null | wc -l)
  dep=""
  if [ -f $d/instances/instances_info.json ] && [ $nl -gt 0 ] && [ $ni -eq $((nl*6)) ]; then :; else
    p=$(sbatch --parsable -J ds-pre-$i --cpus-per-task=2 --mem=24G --time=4:00:00 slurm/ds_preprocess_tv.sh $i 1 1); dep="--dependency=afterok:$p"; fi
  s=$(sbatch --parsable -J s6-sky-$i --gres=gpu:1 --time=1:00:00 $dep slurm/py.sbatch scripts/sky_masks.py --data_root $R --scenes $(printf %03d $i))
  sbatch --parsable -J ds-tv-$i --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --time=5:00:00 --dependency=afterok:$s \
    --export=ALL,DSROOT=data/nuscenes/processed_10Hz/trainval,NUSC_ROOT=data/nuscenes/raw_full,NUSC_VER=v1.0-trainval slurm/ds_scene.sh $i 15000 tv > /dev/null
  echo -n "$i${dep:+p} "
done; echo
