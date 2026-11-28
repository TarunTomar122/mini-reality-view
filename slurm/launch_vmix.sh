#!/bin/bash
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
SID=$1; TID=$2; MODE=${3:-passed}; PY=.venv/bin/python; G2=runs/vmix_gate; MARK=$G2/SUBMITTED; INTENT=$G2/PENDING; CREC=runs/cache/check.json
fail() { echo "GATE FAIL: $*"; exit 1; }
for x in "$SID" "$TID"; do [[ "$x" =~ ^[0-9]+$ ]] || fail "job ids must be numeric ($*)"; done
# MODE pending (council/vmix-pending-astra.md, Astra SIGNED A, 28 Nov ~05:00): A5's "--cache_pending": updates 1-2000 are pure key
# + jitter; the trainer re-checks runs/cache/check.json at update 2000 with the shared verifier (original < 0.01 parity) and holds
# without a passing record; the first record it uses is pinned
case "$MODE" in passed|pending) ;; *) fail "mode '$MODE' (passed | pending)";; esac
mkdir -p $G2
exec 9>$G2/.lock; flock -n 9 || { echo "another launch-vmix holds the lock: nothing to do"; exit 0; }
[ -e $MARK ] && { echo "vmix already submitted ($(cat $MARK)): nothing to do"; exit 0; }
# an intent without a receipt = an earlier launcher was interrupted around sbatch: never resubmit blindly
[ -e $INTENT ] && fail "ambiguous earlier submission ($(cat $INTENT)): reconcile by hand (sacct -n -X --name=vmix), then write $MARK or remove $INTENT"
Q=$(squeue -u $USER -h -n vmix -o %i) || fail "squeue failed"
[ -z "$Q" ] || { echo "vmix job $Q already queued: nothing to do"; exit 0; }
{ [ -e runs/vmix/args.json ] || [ -e runs/vmix/ckpt.pt ]; } && fail "runs/vmix already has a run"
[ "$MODE" = pending ] || [ -s $CREC ] || fail "missing passed cache check $CREC"
STAMP=$(date -u +%Y%m%dT%H%M%SZ); REC=$G2/$STAMP
[ -e $REC ] && fail "record dir $REC exists"
mkdir -p $REC || fail "cannot create $REC"
if [ "$MODE" = passed ]; then
  cp $CREC $REC/cache_check.json || fail "cannot snapshot the cache record"
  OUT=$($PY scripts/vclean_gate.py --arm vmix --suite $SID --t2 $TID --cache_record $CREC --out $REC/vmix.json); rc=$?
else
  OUT=$($PY scripts/vclean_gate.py --arm vmix --suite $SID --t2 $TID --cache_pending --out $REC/vmix.json); rc=$?
fi; echo "$OUT"
[ $rc = 0 ] && [ -s $REC/vmix.json ] || fail "vclean_gate exit $rc"
OUT2=$($PY scripts/vclean_gate.py --verify $REC/vmix.json); rc=$?; echo "$OUT2"
[ $rc = 0 ] || fail "gate record does not re-verify"
read -r RHO ST ARM <<< "$($PY -c "import json;r=json.load(open('$REC/vmix.json'));print(r['rho_gate'],r['cache']['status'].split(':')[0].strip(),r['arm'])")"
[ "$ARM" = vmix ] && [ "$ST" = "$MODE" ] || fail "record arm '$ARM' cache '$ST' (need vmix / $MODE)"
[ "$RHO" = 1.0 ] || fail "rho_gate '$RHO': this launch / scoring path uses signed A@20k as the own cold, valid only at rho 1.0 (Astra launcher review)"
if [ "$MODE" = passed ]; then cmp -s $CREC $REC/cache_check.json || fail "cache record changed during the gate"
  sha256sum $REC/vmix.json $CREC council/vmix_pins.json > $REC/inputs.sha256
else sha256sum $REC/vmix.json council/vmix_pins.json > $REC/inputs.sha256; fi
mkdir -p runs/vmix; lfs setstripe -i 0 -c 1 runs/vmix 2>/dev/null
echo "intent $(date -u +%FT%TZ) record=$REC suite=$SID t2=$TID rho=$RHO" > $INTENT && [ -s $INTENT ] || fail "cannot write the intent file"
J=$(sbatch --parsable -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=6:00:00 -J vmix -o logs2/%x-%j.out \
    slurm/py.sbatch scripts/train_clean.py --arm vmix --init runs/longA/ckpt_20000.pt --out runs/vmix --rho_gate $RHO \
    --gate_record $REC/vmix.json --cache_record $CREC --steps 12000)
[[ "$J" =~ ^[0-9]+$ ]] || fail "sbatch did not return a job id ('$J'): the intent file stays; check sacct before any retry"
R="job=$J stage=12000 cache=$MODE rho=$RHO record=$REC gate=$REC/vmix.json cache_record=$CREC tests=$SID,$TID"
echo "$R" > $MARK && echo "$R" > $REC/job && grep -qx "$R" $MARK && grep -qx "$R" $REC/job || { echo "WARNING: training job $J SUBMITTED but the receipt write failed; record by hand"; exit 2; }
echo "GATE PASS: suite $SID, T2 $TID, cache $MODE, rollout rho $RHO; training job vmix $J; record $REC"
