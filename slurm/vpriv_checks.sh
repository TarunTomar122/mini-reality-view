#!/bin/bash
# CPU-only watcher for the vpriv checks (A1: 2k / 4k informational, 8k / 12k / 16k gated; A3 reports): when
# runs/vpriv/ckpt_<k>000.pt appears, submit the SAME dev panel as vlive: 6 stream rollout shards on 2080 Tis, signed metrics
# with --cold runs/troll/Aoff_rho0.95_dev, then slurm/vpriv_score.sh (signed compares vs A@20k and own cold with panel checks,
# the A1 decision / stop at 8k+, LODO); plus the two-noise probe (shared / private). No GPU waits for a checkpoint.
# Queue errors are retried, never counted as zero; every submission must return a job id (Astra launch review 2).
# Respects the owner's cap (16 active Mini Reality View jobs, held jobs not counted).
#   sbatch --cpus-per-task=1 --mem=2G --time=9:00:00 -J vpriv-watch -o logs2/%x-%j.out slurm/vpriv_checks.sh
cd $HOME/mini-reality-view
G="--gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=4 --mem=48G"
active() {  # prints the active count; returns 1 if squeue failed
  local q; q=$(squeue -u $USER -h -o "%R %Z") || return 1
  echo "$q" | grep mini-reality-view | grep -vc JobHeldUser; return 0; }
room() { local n; while :; do n=$(active) && [ $(( n + $1 )) -le 16 ] && return; sleep 60; done; }
sub() {  # sbatch with retries; prints the job id or exits the watcher
  local j i; for i in 1 2 3 4 5; do j=$(sbatch --parsable "$@") && [[ "$j" =~ ^[0-9]+$ ]] && { echo $j; return; }; sleep 60; done
  echo "WATCH_FAIL sbatch $*" >&2; exit 1; }
running() { local q; q=$(squeue -u $USER -h -n vpriv -o %i) || return 0; [ -n "$q" ]; }  # queue error: assume running
for k in 2 4 8 12 16; do
  ck=runs/vpriv/ckpt_${k}000.pt
  until [ -f $ck ]; do
    # before the launch (suite + gate pending) just wait; once training has started, stop if the run is gone without the ckpt
    if [ -e runs/vpriv/args.json ] && ! running && [ ! -f $ck ]; then echo "vpriv not running and $ck missing: stop"; exit 1; fi
    sleep 60; done
  sleep 60  # let the save finish
  out=runs/troll/P${k}k_dev; mkdir -p $out; lfs setstripe -i 0 -c 1 $out
  room 9; ids=""
  for sh in 0 1 2 3 4 5; do
    ids="$ids:$(sub $G --time=3:00:00 -J pchk${k}k-$sh -o logs2/%x-%j.out slurm/py.sbatch scripts/causal_rollout.py \
        --ckpt $ck --mode stream --split dev --shard $sh/6 --out $out)" || exit 1; done
  m=$(sub -d afterok$ids $G --time=1:00:00 -J tmet-P${k}k -o logs2/%x-%j.out slurm/py.sbatch scripts/temporal_metrics.py \
      --roll $out --cold runs/troll/Aoff_rho0.95_dev --out runs/troll/m_P${k}k_dev.json) || exit 1
  c=$(sub -d afterok:$m --cpus-per-task=1 --mem=4G --time=0:30:00 -J pcmp${k}k -o logs2/%x-%j.out slurm/vpriv_score.sh $k) || exit 1
  p=$(sub $G --time=1:00:00 -J pprobe${k}k -o logs2/%x-%j.out --wrap "cd $HOME/mini-reality-view; \
      bash slurm/py.sbatch scripts/noise_probe.py --ckpt $ck --noise ckpt --out runs/probe/vpriv_noise_${k}000_ckpt.json && \
      bash slurm/py.sbatch scripts/noise_probe.py --ckpt $ck --noise private --out runs/probe/vpriv_noise_${k}000_private.json") || exit 1
  echo "$(date -u +%H:%M) ${k}k: rollouts${ids} metrics $m score $c probes $p"
done
echo "VPRIV_CHECKS all queued"
