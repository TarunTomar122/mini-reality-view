#!/bin/bash
# PLAN-video S1 dev panel for one checkpoint (scripts/wan_panel.py): A@20k RGB-path parity, hybrid / guide / tokens / none at fixed
# noise through the signed temporal_compare (candidate first, A@20k second, A's k=3 list, --floor 0 --cold_ref B), camera swap,
# S1 decision. Debits the S1 ledger. Needs the slice (or another bf16 GPU) while training is NOT running on it.
#   sbatch -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=4:00:00 -J wan-panel -o logs2/%x-%j.out \
#          slurm/wan_panel.sh runs/wan_s1/ckpt_12000.pt s1_12k 24h
#   slurm/wan_panel.sh parity     : only (re)establish the bound A@20k RGB-path replay parity (MODE=train requires its PASS)
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
mkdir -p $TMPDIR logs2
if [ "$1" = parity ]; then PARITY=1; shift; else PARITY=0
  [ $# -ge 3 ] || { echo "usage: wan_panel.sh <ckpt> <name> <check|24h|final> [extra wan_panel.py flags] | wan_panel.sh parity"; exit 2; }
  CK=$1; NAME=$2; STAGE=$3; shift 3
fi
sha256sum scripts/wan_panel.py scripts/wan_rollout.py scripts/score_rgb.py scripts/temporal_metrics.py scripts/temporal_compare.py mrv/wan_plan.py
LEDGER=runs/wan_s1_ledger.jsonl
export WAN_LEDGER_OUTER=1  # S1b item 10: the whole allocation is charged once by this heartbeat; the panel only reads
.venv/bin/python -m mrv.wan_plan beat $LEDGER alloc_panel & BEAT=$!
stop_beat() { kill -TERM $BEAT 2>/dev/null; wait $BEAT 2>/dev/null; }
trap stop_beat EXIT
trap 'stop_beat; exit 143' TERM INT
.venv/bin/python scripts/wan_tests.py > /dev/null || { echo "WAN_TESTS failed"; exit 3; }
BUDGET=24; [ -s runs/wan_s1/S1_24H_DECISION.json ] && grep -q '"EXTEND' runs/wan_s1/S1_24H_DECISION.json && BUDGET=36
if [ $PARITY = 1 ]; then
  .venv/bin/python scripts/wan_panel.py --parity_only --ledger $LEDGER --budget_hours $BUDGET "$@"
else
  .venv/bin/python scripts/wan_panel.py --ckpt $CK --name $NAME --stage $STAGE --ledger $LEDGER --budget_hours $BUDGET "$@"
fi
