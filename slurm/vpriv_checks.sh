#!/bin/bash
# CPU-only watcher for the vpriv checks (A1: 2k / 4k informational, 8k / 12k / 16k gated; A3 reports): when
# runs/vpriv/ckpt_<k>000.pt appears, submit the SAME dev panel as vlive: 6 stream rollout shards on 2080 Tis, signed metrics
# with --cold runs/troll/Aoff_rho0.95_dev, then a CPU job with the signed compares (vs A@20k, vs own cold) + leave-one-drive-out;
# plus the two-noise probe (shared / private). No GPU waits for a checkpoint. Respects the owner's cap (16 active Mini Reality
# View jobs, held jobs not counted) before each submission.
#   sbatch --cpus-per-task=1 --mem=2G --time=9:00:00 -J vpriv-watch -o logs2/%x-%j.out slurm/vpriv_checks.sh
cd $HOME/mini-reality-view
G="--gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=4 --mem=48G"
active() { squeue -u $USER -h -o "%R %Z" | grep mini-reality-view | grep -vc JobHeldUser; }
room() { until [ $(( $(active) + $1 )) -le 16 ]; do sleep 60; done; }
for k in 2 4 8 12 16; do
  ck=runs/vpriv/ckpt_${k}000.pt
  until [ -f $ck ]; do
    # before the launch (suite + gate pending) just wait; once training has started, stop if the run is gone without the ckpt
    if [ -e runs/vpriv/args.json ] && [ -z "$(squeue -u $USER -h -n vpriv -o %i)" ] && [ ! -f $ck ]; then
      echo "vpriv not running and $ck missing: stop"; exit 1; fi
    sleep 60; done
  sleep 60  # let the save finish
  out=runs/troll/P${k}k_dev; mkdir -p $out; lfs setstripe -i 0 -c 1 $out
  room 9; ids=""
  for sh in 0 1 2 3 4 5; do
    ids="$ids:$(sbatch --parsable $G --time=3:00:00 -J pchk${k}k-$sh -o logs2/%x-%j.out slurm/py.sbatch scripts/causal_rollout.py \
        --ckpt $ck --mode stream --split dev --shard $sh/6 --out $out)"; done
  m=$(sbatch --parsable -d afterok$ids $G --time=1:00:00 -J tmet-P${k}k -o logs2/%x-%j.out slurm/py.sbatch scripts/temporal_metrics.py \
      --roll $out --cold runs/troll/Aoff_rho0.95_dev --out runs/troll/m_P${k}k_dev.json)
  c=$(sbatch --parsable -d afterok:$m --cpus-per-task=1 --mem=4G --time=0:30:00 -J pcmp${k}k -o logs2/%x-%j.out --wrap "cd $HOME/mini-reality-view; \
      for x in 'A20k_indep_dev A20k' 'Aoff_rho0.95_dev Aoff0.95'; do set -- \$x; .venv/bin/python scripts/temporal_compare.py --compare \
      runs/troll/m_P${k}k_dev.json runs/troll/m_\$1.json --count_list runs/count/count_\$2_dev_k3.json --floor 0 --cold_ref B \
      > runs/s_lock/cmp_P${k}k__\$1.json; done; .venv/bin/python scripts/lodo.py runs/troll/m_P${k}k_dev.json runs/troll/m_A20k_indep_dev.json \
      --out runs/s_lock/lodo_P${k}k_vs_A20k.json")
  p=$(sbatch --parsable $G --time=1:00:00 -J pprobe${k}k -o logs2/%x-%j.out --wrap "cd $HOME/mini-reality-view; \
      bash slurm/py.sbatch scripts/noise_probe.py --ckpt $ck --noise ckpt --out runs/probe/vpriv_noise_${k}000_ckpt.json && \
      bash slurm/py.sbatch scripts/noise_probe.py --ckpt $ck --noise private --out runs/probe/vpriv_noise_${k}000_private.json")
  echo "$(date -u +%H:%M) ${k}k: rollouts${ids} metrics $m compare $c probes $p"
done
echo "VPRIV_CHECKS all queued"
