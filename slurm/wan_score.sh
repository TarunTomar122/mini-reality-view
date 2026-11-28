#!/bin/bash
# F3 (council/s1-timing-r1-CONVERGED.md item 4): signed RGB scoring of S1 panels on 2080 Ti jobs (scripts/wan_score.py), under its
# own ledger runs/wan_score_ledger.jsonl (this allocation's outer heartbeat, kind score_<name> / score_parity), reported separately
# from the 24 H200-3g slice-hour S1 budget. Cluster cap: 16 Mini Reality View jobs at once (submit the rest --hold).
# 1. scorer parity (once, before any PASS; re-run after any scorer / adapter / SD VAE / wrapper change):
#      G="--gres=gpu:nvidia_geforce_rtx_2080_ti:1 --exclude=damnii10 --cpus-per-task=4 --mem=32G"
#      sbatch $G --time=8:00:00 -J wsc-par -o logs2/%x-%j.out slurm/wan_score.sh parity            # unsharded A@20k replay (+ rate)
#      for i in 0 1 2; do sbatch $G --time=4:00:00 -J wsc-par$i -o logs2/%x-%j.out slurm/wan_score.sh parity $i/3; done
#      sbatch -d afterok:<the four ids> --cpus-per-task=1 --mem=8G --time=0:30:00 -J wsc-pchk -o logs2/%x-%j.out \
#             slurm/wan_score.sh parity_check 3                                                     # CPU: sharded == unsharded
# 2. one rendered panel (after slurm/wan_panel.sh ... finished on the slice), sharded by drive (7 scorings x 2 shards = 14 jobs):
#      for W in hybrid guide tokens none swap_full swap_postdonor swap_postrecv; do for i in 0 1; do
#        sbatch $G --time=3:00:00 -J wsc-$W$i -o logs2/%x-%j.out slurm/wan_score.sh panel s1_12k $W $i/2; done; done
#    then slurm/wan_panel.sh decide <ckpt> s1_12k <stage> (CPU).
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
mkdir -p $TMPDIR logs2
PY=.venv/bin/python; SL=runs/wan_score_ledger.jsonl
case $1 in
  parity)       KIND=score_parity; ARGS="--parity --shard ${2:-0/1}" ;;
  parity_check) KIND=;             ARGS="--parity_check --shards ${2:-3}" ;;
  panel)        [ $# -ge 3 ] || { echo "usage: wan_score.sh panel <name> <what> [i/n]"; exit 2; }
                KIND=score_$2;     ARGS="--panel $2 --what $3 --shard ${4:-0/1}" ;;
  *) echo "usage: wan_score.sh parity [i/n] | parity_check <n> | panel <name> <what> [i/n]"; exit 2 ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
sha256sum scripts/wan_score.py scripts/score_rgb.py scripts/temporal_metrics.py mrv/wan_plan.py
echo "start $(date -u) on $(hostname) $*"
if [ -n "$KIND" ]; then  # the whole 2080 Ti allocation is charged once to the SCORE ledger (never the S1 slice ledger)
  export WAN_LEDGER_OUTER=1
  $PY -m mrv.wan_plan beat $SL $KIND & BEAT=$!
  stop_beat() { kill -TERM $BEAT 2>/dev/null; wait $BEAT 2>/dev/null; }
  trap stop_beat EXIT
  trap 'stop_beat; exit 143' TERM INT
fi
$PY scripts/wan_tests.py > /dev/null || { echo "WAN_TESTS failed"; exit 3; }
$PY scripts/wan_score.py $ARGS
rc=$?
echo "end $(date -u) rc=$rc"
exit $rc
