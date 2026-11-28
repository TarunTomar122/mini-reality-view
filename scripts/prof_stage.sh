#!/bin/bash
# Cold copy throughput Lustre -> job-local /dev/shm with N parallel streams (council S1 timing r1: staging decision). CPU only.
#   sbatch -w saxa --gres=gpu:h200_1g.18gb:0 --cpus-per-task=8 --mem=24G -t 0:20:00 -J prof-stage -o logs2/%x-%j.out scripts/prof_stage.sh
cd $HOME/mini-reality-view
D=/dev/shm/mrv_prof_$SLURM_JOB_ID; mkdir -p $D; trap 'rm -rf $D' EXIT
F=$(ls data/wan_tok_v1/*.pt | sort -r | head -24)   # files not touched by the reader profile
for N in 8 4; do
  set -- $F; [ $N = 8 ] && L="$(echo $F | tr ' ' '\n' | head -16)" || L="$(echo $F | tr ' ' '\n' | tail -8)"
  B=$(du -cb $L | tail -1 | cut -f1); t0=$(date +%s.%N)
  echo "$L" | xargs -P $N -I{} sh -c 'cp {} '$D'/ && sha256sum '$D'/$(basename {}) > /dev/null'
  t1=$(date +%s.%N); rm -f $D/*
  echo "PROF_STAGE streams $N files $(echo "$L" | wc -l) MB $((B/1000000)) s $(echo "$t1 - $t0" | bc) MBps $(echo "$B/1000000/($t1 - $t0)" | bc)"
done
