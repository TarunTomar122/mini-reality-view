#!/bin/bash
# CPU-only watcher for the vpriv checks (A1: 2k / 4k informational, 8k / 12k / 16k gated; A3 reports). For every saved
# runs/vpriv/ckpt_<k>000.pt it submits the SAME dev panel as vlive: 6 stream rollout shards on 2080 Tis, signed metrics with
# --cold runs/troll/Aoff_rho0.95_dev, then slurm/vpriv_score.sh (signed compares with panel checks, A1 decision at 8k+, the
# 12k stage-2 resume, LODO), plus the two-noise probe. No GPU waits for a checkpoint.
# Admission (Astra 01:30 review (d)): jobs are admitted ONE AT A TIME as slots open under BOTH the owner cap (16 active MRV
# jobs, held not counted) and the account limit (20), in priority order 8k, 12k, 16k, then 2k, 4k; when no slot is free the
# pass ends, so lower-priority work never takes a slot ahead of gated work; 2k / 4k jobs also keep 9 slots in reserve. Every submitted id is recorded in
# runs/vpriv_gate/watch/ (a restarted watcher resumes without resubmitting). One sbatch per job under a unique name, adopted
# only if exactly one job carries it, else the watcher stops loudly (Astra launch reviews 2-5).
#   sbatch --cpus-per-task=1 --mem=2G --time=9:00:00 -J vpriv-watch -o logs2/%x-%j.out slurm/vpriv_checks.sh
cd $HOME/mini-reality-view
G="--gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=4 --mem=48G"
S=runs/vpriv_gate/watch; mkdir -p $S || exit 1
free() {  # prints the number of free slots (min of both limits); returns 1 if squeue failed
  local q m a; q=$(squeue -u $USER -h -o "%R %Z") || return 1
  m=$(echo "$q" | grep mini-reality-view | grep -vc JobHeldUser); a=$(echo "$q" | grep -c .)
  local f=$(( 16 - m )); [ $(( 20 - a )) -lt $f ] && f=$(( 20 - a )); echo $f; }
sub() {  # sub <job base name> <sbatch args...>: ONE sbatch under a unique name; on a lost / failed receipt adopt the job only if
         # exactly one job (queued or finished, squeue + sacct) carries that unique name, else exit loudly (never resubmit)
  local n="$1.$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n')" j q; shift
  j=$(sbatch --parsable -J $n "$@") && [[ "$j" =~ ^[0-9]+$ ]] && { echo $j; return; }
  sleep 30
  local q1 q2
  q1=$(squeue -u $USER -h -n $n -o %i) || { echo "WATCH_FAIL ambiguous sbatch for $n (squeue failed): reconcile by hand" >&2; exit 1; }
  q2=$(sacct -u $USER -X -n -P --name=$n -o JobID) || { echo "WATCH_FAIL ambiguous sbatch for $n (sacct failed): reconcile by hand" >&2; exit 1; }
  q=$(printf "%s\n%s\n" "$q1" "$q2" | grep -E '^[0-9]+$' | sort -u)
  [ "$(echo "$q" | grep -c .)" = 1 ] && { echo $q; return; }
  echo "WATCH_FAIL ambiguous sbatch for $n (matches: ${q:-none}): reconcile by hand" >&2; exit 1; }
running() { local q; q=$(squeue -u $USER -h -n vpriv -o %i) || return 0; [ -n "$q" ]; }  # queue error: assume running
take() {  # take one slot or fail the pass (returns 1 when none is free / squeue failed); informational checks (2k / 4k)
          # leave 9 slots free so a gated check can start in full the moment its checkpoint is saved
  local f need=1; case $k in 2|4) need=10;; esac; f=$(free) || return 1; [ "$f" -ge $need ]; }
rec() { echo "$2" > $S/$1 && [ "$(cat $S/$1)" = "$2" ] || { echo "WATCH_FAIL cannot record $1" >&2; exit 1; }; }

pass() {  # one admission pass in priority order; returns as soon as no slot is free
  alldone=1
  for k in 8 12 16 2 4; do
    [ -e $S/$k.done ] && continue
    alldone=0; ck=runs/vpriv/ckpt_${k}000.pt
    if [ ! -f $ck ]; then
      if [ $k = 16 ]; then
        [ -e runs/vpriv_gate/A1_STOPPED ] && { echo "A1 stop at 12k: no 16k check"; rec 16.done skipped; continue; }
        if [ -e runs/vpriv_gate/CONTINUED ] && ! running && [ ! -f $ck ]; then echo "continuation gone and $ck missing: stop"; exit 1; fi
      elif [ -e runs/vpriv/args.json ] && ! running && [ ! -f $ck ]; then echo "vpriv not running and $ck missing: stop"; exit 1; fi
      continue
    fi
    out=runs/troll/P${k}k_dev; mkdir -p $out; lfs setstripe -i 0 -c 1 $out 2>/dev/null
    for sh in 0 1 2 3 4 5; do
      [ -s $S/$k.sh$sh ] && continue
      take || return 0  # no slot: end this pass (priority order holds)
      j=$(sub pchk${k}k-$sh $G --time=3:00:00 -o logs2/%x-%j.out slurm/py.sbatch scripts/causal_rollout.py \
          --ckpt $ck --mode stream --split dev --shard $sh/6 --out $out) || exit 1
      rec $k.sh$sh $j; echo "$(date -u +%H:%M) ${k}k shard $sh: $j"
    done
    if [ ! -s $S/$k.m ]; then
      take || return 0
      ids=$(for sh in 0 1 2 3 4 5; do printf ":%s" $(cat $S/$k.sh$sh); done)
      j=$(sub tmet-P${k}k -d afterok$ids $G --time=1:00:00 -o logs2/%x-%j.out slurm/py.sbatch scripts/temporal_metrics.py \
          --roll $out --cold runs/troll/Aoff_rho0.95_dev --out runs/troll/m_P${k}k_dev.json) || exit 1
      rec $k.m $j; echo "$(date -u +%H:%M) ${k}k metrics: $j"
    fi
    if [ ! -s $S/$k.c ]; then
      dep=afterok:$(cat $S/$k.m)
      case $k in 12) p=8;; 16) p=12;; *) p="";; esac  # each gated decision also waits for the previous gated score job
      if [ -n "$p" ]; then [ -s $S/$p.c ] || continue; dep=$dep:$(cat $S/$p.c); fi
      take || return 0
      j=$(sub pcmp${k}k -d $dep --cpus-per-task=1 --mem=4G --time=0:30:00 -o logs2/%x-%j.out slurm/vpriv_score.sh $k) || exit 1
      rec $k.c $j; echo "$(date -u +%H:%M) ${k}k score: $j"
    fi
    if [ ! -s $S/$k.p ]; then
      take || return 0
      j=$(sub pprobe${k}k $G --time=1:00:00 -o logs2/%x-%j.out --wrap "cd $HOME/mini-reality-view; \
          bash slurm/py.sbatch scripts/noise_probe.py --ckpt $ck --noise ckpt --out runs/probe/vpriv_noise_${k}000_ckpt.json && \
          bash slurm/py.sbatch scripts/noise_probe.py --ckpt $ck --noise private --out runs/probe/vpriv_noise_${k}000_private.json") || exit 1
      rec $k.p $j; echo "$(date -u +%H:%M) ${k}k probes: $j"
    fi
    rec $k.done queued
  done
}
while :; do
  pass
  [ $alldone = 1 ] && { echo "VPRIV_CHECKS all queued"; exit 0; }
  sleep 60
done
