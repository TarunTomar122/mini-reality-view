#!/bin/bash
# Car map scoring panel (council/PLAN-carmap.md decision rules) on 2080 Ti jobs via slurm/tpv_judge.sh.
# usage (login node, from ~/mini-reality-view): bash slurm/cm_score.sh <cmX1|cmX2> <step> <what...> [--hold]
#   dec      decision drives (30 val x 4 keyframes x seeds 0/1), detector boxes (camera-only cache)
#   mapswap  same, but the car map of the drive half the list away (map-only swap)
#   swap     full swap: cameras, rays and map all from the other drive
#   oracle   nuScenes annotations instead of detector boxes (oracle test)
#   train    30 fixed train drives (tmp/bev_leak_train30.txt; 29 have game keys), degraded training boxes
cd $HOME/mini-reality-view
ARM=$1; STEP=$2; shift 2
HOLD=""; for w in "$@"; do [ "$w" = "--hold" ] && HOLD="--hold"; done
CK=runs/$ARM/ckpt_$STEP.pt
G=""; [ "$ARM" = "cmX2" ] && G="--carmap_grid data/bev_tpv"
for w in "$@"; do
  B="--carmap_boxes tmp/bev_boxes_decision_zero.json"; X=""
  case $w in
    dec) ;;
    mapswap) X="--map_swap" ;;
    swap) X="--control swap_scene" ;;
    oracle) X="--carmap_oracle" ;;
    train) B="--carmap_boxes tmp/bev_boxes_trainD.json"; X="--split train --scene_list tmp/bev_leak_train30.txt" ;;
    --hold) continue ;;
    *) echo "unknown $w"; exit 1 ;;
  esac
  N=${ARM}_${STEP}_$w
  echo $N $(sbatch --parsable $HOLD -J tj-$N slurm/tpv_judge.sh $N $CK novae $B $G $X)
done
