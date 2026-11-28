#!/bin/bash
# PLAN-video S1 dev panel for one checkpoint (scripts/wan_panel.py), split by F3 (council/s1-timing-r1-CONVERGED.md item 4):
#   1. RENDER on the slice (this script, default): swap preflight, render-only budget check, production / marker checks, ONE
#      wan_rollout.py --plan process (Wan loaded once) renders hybrid / guide / tokens / none + swap + shared references, writes
#      runs/wroll/<name>_rgb_manifest.json (+ .sha256); debits the S1 ledger; the slice is released afterwards.
#      sbatch -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=4:00:00 -J wan-panel -o logs2/%x-%j.out \
#             slurm/wan_panel.sh runs/wan_s1/ckpt_12000.pt s1_12k 24h
#   2. SCORE on 2080 Ti jobs (slurm/wan_score.sh panel <name> <what> <i/n>; own ledger + cap, reported separately).
#   3. DECIDE on CPU (no slice time; needs the bound 2080 Ti scorer-parity PASS and complete score shards of THIS render):
#      sbatch --cpus-per-task=2 --mem=16G --time=1:00:00 -J wan-decide -o logs2/%x-%j.out \
#             slurm/wan_panel.sh decide runs/wan_s1/ckpt_12000.pt s1_12k 24h
#   The A@20k replay parity moved off the slice: slurm/wan_score.sh parity (+ parity_check).
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
mkdir -p $TMPDIR logs2
if [ "$1" = parity ]; then echo "moved: the A@20k replay parity runs on the 2080 Ti now (slurm/wan_score.sh parity, then parity_check)"; exit 2; fi
PHASE=render; [ "$1" = decide ] && { PHASE=decide; shift; }
[ $# -ge 3 ] || { echo "usage: wan_panel.sh [decide] <ckpt> <name> <check|24h|final> [extra wan_panel.py flags]"; exit 2; }
CK=$1; NAME=$2; STAGE=$3; shift 3
sha256sum scripts/wan_panel.py scripts/wan_rollout.py scripts/wan_score.py scripts/score_rgb.py scripts/temporal_metrics.py scripts/temporal_compare.py mrv/wan_plan.py
LEDGER=runs/wan_s1_ledger.jsonl
.venv/bin/python scripts/wan_tests.py > /dev/null || { echo "WAN_TESTS failed"; exit 3; }
BUDGET=24; [ -s runs/wan_s1/S1_24H_DECISION.json ] && grep -q '"EXTEND' runs/wan_s1/S1_24H_DECISION.json && BUDGET=36
if [ $PHASE = render ]; then
  export WAN_LEDGER_OUTER=1  # S1b item 10: the whole slice allocation is charged once by this heartbeat; the panel only reads
  .venv/bin/python -m mrv.wan_plan beat $LEDGER alloc_panel & BEAT=$!
  stop_beat() { kill -TERM $BEAT 2>/dev/null; wait $BEAT 2>/dev/null; }
  trap stop_beat EXIT
  trap 'stop_beat; exit 143' TERM INT
fi
.venv/bin/python scripts/wan_panel.py --phase $PHASE --ckpt $CK --name $NAME --stage $STAGE --ledger $LEDGER --budget_hours $BUDGET "$@"
