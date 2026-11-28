#!/bin/bash
# PLAN-video S1 dev panel for one checkpoint (scripts/wan_panel.py): A@20k RGB-path parity, hybrid / guide / tokens / none at fixed
# noise through the signed temporal_compare (candidate first, A@20k second, A's k=3 list, --floor 0 --cold_ref B), camera swap,
# S1 decision. Debits the S1 ledger. Needs the slice (or another bf16 GPU) while training is NOT running on it.
#   sbatch -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=4:00:00 -J wan-panel -o logs2/%x-%j.out \
#          slurm/wan_panel.sh runs/wan_s1/ckpt_12000.pt s1_12k 24h
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
mkdir -p $TMPDIR logs2
[ $# -ge 3 ] || { echo "usage: wan_panel.sh <ckpt> <name> <check|24h|final> [extra wan_panel.py flags]"; exit 2; }
CK=$1; NAME=$2; STAGE=$3; shift 3
sha256sum scripts/wan_panel.py scripts/wan_rollout.py scripts/score_rgb.py scripts/temporal_metrics.py scripts/temporal_compare.py mrv/wan_plan.py
.venv/bin/python scripts/wan_tests.py > /dev/null || { echo "WAN_TESTS failed"; exit 3; }
.venv/bin/python scripts/wan_panel.py --ckpt $CK --name $NAME --stage $STAGE --ledger runs/wan_s1_ledger.jsonl "$@"
