#!/bin/bash
# Gate + launch of the A5 signed attribution CONTROL "vctl" = pure key + jitter (--arm vclean --jitter 1,41,81; council/
# vmix-pending-final-astra.md + vmix-pending-fable.md: run it now while the vmix cache is rebuilt; vmix keeps priority).
# Derived from slurm/launch_vmix.sh. Original header follows.
# Gate + launch of A5 "vmix" (council/AGREED-causal-r1.md A5; Astra vmix reviews 1-4). Copy of slurm/launch_vpriv.sh (unchanged)
# adapted to vmix. Arguments: the JOB IDs of the clean suite (slurm/clean_tests.sh, -J cleantests) and clean T2 (slurm/clean_t2.sh,
# -J cleant2):
#   1. under one lock, with an intent file before sbatch and a receipt after (never resubmit blindly);
#   2. scripts/vclean_gate.py --arm vmix builds a NEW record dir runs/vmix_gate/<stamp>/vmix.json binding: the signed vpriv
#      no-joint-pass outcome (A1 decisions re-verified), the suite log (last line FAILS=0, job-ID-bound dir) with its code manifest
#      == the files on disk (incl. council/vmix_pins.json), the timing record, the clean T2 -> rho_gate, and the PASSED cache check
#      runs/cache/check.json accepted by the shared verifier (pinned parity reference re-measured, pinned dev digests) -- not pending;
#      the record is re-verified (--verify) before use; rho_gate is read from it;
#   3. stage 1 trains to 12k and pauses (A1 stop must be able to act); slurm/vmix_score.sh resumes to 16k only if allowed;
#   4. ONE training job on ONE saxa h200_3g.71gb slice (the trainer re-verifies the gate record and the cache record itself);
#      the job id receipt is written into the record dir and the marker runs/vmix_gate/SUBMITTED, and checked.
#   sbatch -H -d afterok:<suite>:<t2> --cpus-per-task=1 --mem=8G --time=0:45:00 -J launch-vmix -o logs2/%x-%j.out \
#          slurm/launch_vmix.sh <suite id> <t2 id> [passed | pending]
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1
mkdir -p $PWD/runs/tmp && export TMPDIR=$PWD/runs/tmp
SID=$1; TID=$2; MODE=control; PY=.venv/bin/python; G2=runs/vctl_gate; MARK=$G2/SUBMITTED; INTENT=$G2/PENDING
fail() { echo "GATE FAIL: $*"; exit 1; }
for x in "$SID" "$TID"; do [[ "$x" =~ ^[0-9]+$ ]] || fail "job ids must be numeric ($*)"; done
# MODE pending (council/vmix-pending-astra.md, Astra SIGNED A, 28 Nov ~05:00): A5's "--cache_pending": updates 1-2000 are pure key
# + jitter; the trainer re-checks runs/cache/check.json at update 2000 with the shared verifier (original < 0.01 parity) and holds
# without a passing record; the first record it uses is pinned

mkdir -p $G2
exec 9>$G2/.lock; flock -n 9 || { echo "another launch-vctl holds the lock: nothing to do"; exit 0; }
[ -e $MARK ] && { echo "vctl already submitted ($(cat $MARK)): nothing to do"; exit 0; }
# an intent without a receipt = an earlier launcher was interrupted around sbatch: never resubmit blindly
[ -e $INTENT ] && fail "ambiguous earlier submission ($(cat $INTENT)): reconcile by hand (sacct -n -X --name=vctl), then write $MARK or remove $INTENT"
Q=$(squeue -u $USER -h -n vctl -o %i) || fail "squeue failed"
[ -z "$Q" ] || { echo "vctl job $Q already queued: nothing to do"; exit 0; }
{ [ -e runs/vctl/args.json ] || [ -e runs/vctl/ckpt.pt ]; } && fail "runs/vctl already has a run"
STAMP=$(date -u +%Y%m%dT%H%M%SZ); REC=$G2/$STAMP
[ -e $REC ] && fail "record dir $REC exists"
mkdir -p $REC || fail "cannot create $REC"
OUT=$($PY scripts/vclean_gate.py --arm vclean --control --suite $SID --t2 $TID --out $REC/vctl.json); rc=$?; echo "$OUT"
[ $rc = 0 ] && [ -s $REC/vctl.json ] || fail "vclean_gate exit $rc"
OUT2=$($PY scripts/vclean_gate.py --verify $REC/vctl.json); rc=$?; echo "$OUT2"
[ $rc = 0 ] || fail "gate record does not re-verify"
read -r RHO ARM <<< "$($PY -c "import json;r=json.load(open('$REC/vctl.json'));print(r['rho_gate'],r['arm'])")"
[ "$ARM" = vclean ] || fail "record arm '$ARM' (need vclean control)"
[ "$RHO" = 1.0 ] || fail "rho_gate '$RHO': this launch / scoring path uses signed A@20k as the own cold, valid only at rho 1.0 (Astra launcher review)"
sha256sum $REC/vctl.json > $REC/inputs.sha256
mkdir -p runs/vctl; lfs setstripe -i 0 -c 1 runs/vctl 2>/dev/null
echo "intent $(date -u +%FT%TZ) record=$REC suite=$SID t2=$TID rho=$RHO" > $INTENT && [ -s $INTENT ] || fail "cannot write the intent file"
J=$(sbatch --parsable -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=6:00:00 -J vctl -o logs2/%x-%j.out \
    slurm/py.sbatch scripts/train_clean.py --arm vclean --jitter 1,41,81 --init runs/longA/ckpt_20000.pt --out runs/vctl --rho_gate $RHO \
    --gate_record $REC/vctl.json --steps 16000)
[[ "$J" =~ ^[0-9]+$ ]] || fail "sbatch did not return a job id ('$J'): the intent file stays; check sacct before any retry"
R="job=$J arm=vctl rho=$RHO record=$REC gate=$REC/vctl.json tests=$SID,$TID"
echo "$R" > $MARK && echo "$R" > $REC/job && grep -qx "$R" $MARK && grep -qx "$R" $REC/job || { echo "WARNING: training job $J SUBMITTED but the receipt write failed; record by hand"; exit 2; }
echo "GATE PASS (control): suite $SID, T2 $TID, rollout rho $RHO; training job vctl $J; record $REC"
