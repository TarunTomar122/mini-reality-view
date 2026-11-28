#!/bin/bash
# Gate + launch of A3 "vpriv" (council/AGREED-causal-r1.md A3 CONVERGED + init note; Astra vpriv SIGN-IF + final review):
#   1. the LAST line of the named suite / T2 / vprivA / vprivB logs is their clean result line (FAILS=0 dir=...);
#   2. the code that trains now == the code the suite and the vpriv fixtures tested (sha256 -c of the three manifests);
#   3. the A3 trigger is recomputed NOW with the signed comparator (k=3 A-derived count lists, floor 0, cold_ref B) on
#      signed-scorer metrics, panel identity / 6 drives x seeds 0/1 / finite estimates checked: no joint pass vs A@20k at
#      8k / 12k / 16k, both safety bars (car F1 diff >= -0.03, late gap diff <= +0.03) met at 16k; mixture = 2 private clips
#      unless 16k vs own cold has ID AND W diff < 0 -> 1 alternating private clip; digests bound in council/vpriv_launch.json;
#   4. the rho decision is still valid and == 0.95 (rollout rho, unchanged from vlive);
#   5. only then ONE training job on ONE saxa h200_3g.71gb slice (agreed with the live-VSR session); flock: one launcher.
#   sbatch -d afterok:<ctests> --cpus-per-task=1 --mem=8G --time=0:30:00 -J launch-vpriv -o logs2/%x-%j.out \
#          slurm/launch_vpriv.sh <ctests log> <t2 log> <vprivA log> <vprivB log>
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1
CT=$1; T2=$2; VA=$3; VB=$4; PY=.venv/bin/python
exec 9>runs/.launch_vpriv.lock; flock -n 9 || { echo "another launch-vpriv holds the lock: nothing to do"; exit 0; }
if [ -n "$(squeue -u $USER -h -n vpriv -o %i)" ] || [ -e runs/vpriv/args.json ] || [ -e runs/vpriv/ckpt.pt ]; then
  echo "vpriv already submitted / running: nothing to do"; exit 0; fi
last() { tail -n 1 "$1" | grep -qE "^$2 FAILS=0 dir=[^ ]+$"; }
last $CT CAUSAL_TESTS_SH || { echo "GATE FAIL: test suite not clean ($CT)"; tail -5 $CT; exit 1; }
last $T2 CAUSAL_T2 || { echo "GATE FAIL: T2 not clean ($T2)"; exit 1; }
last $VA "VPRIV_TESTS part a" || { echo "GATE FAIL: vpriv fixtures part a not clean ($VA)"; exit 1; }
last $VB "VPRIV_TESTS part b" || { echo "GATE FAIL: vpriv fixtures part b not clean ($VB)"; exit 1; }
for L in $CT $VA $VB; do
  D=$(tail -n 1 $L | grep -o "dir=[^ ]*" | cut -d= -f2)
  sha256sum -c --quiet $D/code_sha256.txt || { echo "GATE FAIL: code changed since $L ran"; exit 1; }
done
mkdir -p runs/s_lock/gate_vpriv
PRIV=$($PY scripts/a3_trigger.py)
case "$PRIV" in 1|2) ;; *) echo "GATE FAIL: A3 trigger not met ($PRIV) -> back to council"; exit 1;; esac
$PY scripts/rho_decision.py --out council/rho_decision.json; rc=$?
[ $rc = 0 ] || { echo "GATE FAIL: rho decision not valid (rc $rc) -> back to council"; exit 1; }
RHO=$($PY -c "import json;print(json.load(open('council/rho_decision.json'))['chosen'])")
[ "$RHO" = "0.95" ] || { echo "GATE FAIL: rollout rho $RHO != vlive's 0.95"; exit 1; }
mkdir -p runs/vpriv; lfs setstripe -i 0 -c 1 runs/vpriv
[ -e runs/vpriv/ckpt.pt ] && { echo "runs/vpriv already has a checkpoint; not relaunching"; exit 1; }
J=$(sbatch --parsable -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=6:00:00 -J vpriv -o logs2/%x-%j.out \
    slurm/py.sbatch scripts/train_causal.py --init runs/longA/ckpt_20000.pt --out runs/vpriv --K 4 --rho $RHO \
    --rho_decision council/rho_decision.json --private_clips $PRIV --steps 16000)
[ -n "$J" ] || { echo "GATE PASS but sbatch FAILED (account limit?) -> resubmit by hand"; exit 1; }
$PY -c "import json;f='council/vpriv_launch.json';d=json.load(open(f));d.update(job='$J',rho=$RHO,logs=['$CT','$T2','$VA','$VB']);json.dump(d,open(f,'w'),indent=1)"
echo "GATE PASS: tests clean, A3 trigger met, private_clips $PRIV, rollout rho $RHO; training job vpriv $J"
