#!/bin/bash
# Gate + launch of the causal video training (council/AGREED-causal-r1.md + A1; Astra SIGN-IF r4 / r5):
#   1. the test suite log must end "CAUSAL_TESTS_SH FAILS=0" and the T2 log "CAUSAL_T2 FAILS=0";
#   2. scripts/rho_decision.py must return a valid decision on reviewed-code renders (writes council/rho_decision.json);
#   3. only then submit training on ONE saxa h200_3g.71gb slice (agreed with the live-VSR session).
#   sbatch -d afterany:<ctests>:<tmet0.5>:<tmet0.8> --cpus-per-task=1 --mem=8G --time=0:30:00 -J launch-vlive -o logs2/%x-%j.out slurm/launch_vlive.sh <ctests log> <t2 log>
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1
CT=$1; T2=$2
grep -q "CAUSAL_TESTS_SH FAILS=0" $CT || { echo "GATE FAIL: test suite not clean ($CT)"; tail -5 $CT; exit 1; }
grep -q "CAUSAL_T2 FAILS=0" $T2 || { echo "GATE FAIL: T2 not clean ($T2)"; exit 1; }
# the suite must have tested the code that will train / render now
D=$(grep -o "dir=[^ ]*" $CT | tail -1 | cut -d= -f2)
( cd $HOME/mini-reality-view && sha256sum -c $D/code_sha256.txt ) || { echo "GATE FAIL: code changed since the test suite ran"; exit 1; }
.venv/bin/python scripts/rho_decision.py --out council/rho_decision.json; rc=$?
[ $rc = 0 ] || { echo "GATE FAIL: rho decision not valid (rc $rc) -> back to council"; exit 1; }
RHO=$(.venv/bin/python -c "import json;print(json.load(open('council/rho_decision.json'))['chosen'])")
mkdir -p runs/vlive; lfs setstripe -i 0 -c 1 runs/vlive
[ -e runs/vlive/ckpt.pt ] && { echo "runs/vlive already has a checkpoint; not relaunching"; exit 1; }
J=$(sbatch --parsable -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=6:00:00 -J vlive -o logs2/%x-%j.out \
    slurm/py.sbatch scripts/train_causal.py --init runs/longA/ckpt_20000.pt --out runs/vlive --K 4 --rho $RHO \
    --rho_decision council/rho_decision.json --steps 16000)
[ -n "$J" ] || { echo "GATE PASS but sbatch FAILED (account limit?) -> resubmit by hand"; exit 1; }
echo "GATE PASS: tests clean, rho $RHO; training job vlive $J"
