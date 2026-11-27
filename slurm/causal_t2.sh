#!/bin/bash
# T2 of council/AGREED-causal-r1.md on a 2080 Ti (the signed A@20k dev render was made there in fp16); exits nonzero on failure.
#   sbatch --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=4 --mem=48G --time=1:00:00 -J ct2 -o logs2/%x-%j.out slurm/causal_t2.sh
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
PY=.venv/bin/python; A=runs/longA/ckpt_20000.pt; T=runs/ctest_${SLURM_JOB_ID:-manual}; mkdir -p $T; FAILS=0
chk() { if [ "$1" != 0 ]; then echo "FAIL $2 (exit $1)"; FAILS=$((FAILS+1)); fi; }
nvidia-smi --query-gpu=name --format=csv,noheader
DRV=$($PY -c "import json;print(json.load(open('data/temporal_lists.json'))['dev'][0])"); echo $DRV > $T/one.txt
echo "== T2: --off at rho 1.0 vs signed A@20k ($DRV, seed 0; signed render: temporal_rollout independent, fp16, rows [cond, uncond], 1 stream)"
$PY scripts/causal_rollout.py --ckpt $A --mode off --rho 1.0 --split dev --drive_list $T/one.txt --seeds 0 --out $T/off_rho1 > $T/t2.log; chk $? T2-render
$PY - <<P; chk $? T2
import torch, sys
a = torch.load("$T/off_rho1/${DRV}_s0.pt"); b = torch.load("runs/troll/A20k_indep_dev/${DRV}_s0.pt")
ok = len(a["lat"]) == len(b["lat"]) and torch.equal(a["ticks"], b["ticks"]) and torch.equal(a["src_ticks"], b["src_ticks"]) \
     and all(torch.equal(a[k], b[k]) for k in ("ok", "hard", "restart")) and a["amp"] == b["amp"]
d = (a["lat"].float() - b["lat"].float()).abs().max().item() if len(a["lat"]) == len(b["lat"]) else 1e9
print(("PASS" if ok and d < 1e-2 else "FAIL"), "T2 off@rho1 == signed A@20k: ticks/src/flags/amp equal %s (amp %s vs %s), %d ticks, max|diff| %.3g (<1e-2)" % (ok, a["amp"], b["amp"], len(a["lat"]), d))
sys.exit(0 if ok and d < 1e-2 else 1)
P
echo "CAUSAL_T2 FAILS=$FAILS dir=$T"
exit $FAILS
