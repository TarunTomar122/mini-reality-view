#!/bin/bash
# Gate + launch of A3 "vpriv" (council/AGREED-causal-r1.md A3 CONVERGED + init note; Astra vpriv SIGN-IF, final review and
# launch review 28 Nov). Arguments are the JOB IDs of the suite, T2, vpriv fixtures part a and part b:
#   1. the reviewed deployment manifest council/vpriv_deploy.sha256 verifies (includes this launcher, the trigger, the watcher);
#   2. logs2/<name>-<id>.out exists for each named job, its LAST line is the clean result line, its dir is runs/ctest[_vpriv]_<id>,
#      and every line of the suite / part a / part b code manifests appears verbatim in the deployment manifest;
#   3. scripts/a3_trigger.py --launch recomputes the A3 trigger on immutable, provenance-checked snapshots and writes a new
#      record dir runs/vpriv_gate/<stamp>/ (mixture: 2 private clips unless 16k vs own cold has ID AND W diff < 0 -> 1);
#   4. the rho decision is still valid and == 0.95 (rollout rho, unchanged from vlive);
#   5. under one lock, with queue errors fatal and a submission marker: ONE training job on ONE saxa h200_3g.71gb slice;
#      the job id receipt is written into the record dir and the marker, and checked, before success is reported.
#   sbatch -H -d afterok:<suite> --cpus-per-task=1 --mem=8G --time=0:30:00 -J launch-vpriv -o logs2/%x-%j.out \
#          slurm/launch_vpriv.sh <suite id> <t2 id> <vprivA id> <vprivB id>
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1
CTID=$1; T2ID=$2; AID=$3; BID=$4; PY=.venv/bin/python; MARK=runs/vpriv_gate/SUBMITTED; INTENT=runs/vpriv_gate/PENDING
fail() { echo "GATE FAIL: $*"; exit 1; }
for x in "$CTID" "$T2ID" "$AID" "$BID"; do [[ "$x" =~ ^[0-9]+$ ]] || fail "job ids must be numeric ($*)"; done
mkdir -p runs/vpriv_gate
exec 9>runs/vpriv_gate/.lock; flock -n 9 || { echo "another launch-vpriv holds the lock: nothing to do"; exit 0; }
[ -e $MARK ] && { echo "vpriv already submitted ($(cat $MARK)): nothing to do"; exit 0; }
# an intent without a receipt = an earlier launcher was interrupted around sbatch: never resubmit blindly
[ -e $INTENT ] && fail "ambiguous earlier submission ($(cat $INTENT)): reconcile by hand (sacct -n -X --name=vpriv), then write $MARK or remove $INTENT"
Q=$(squeue -u $USER -h -n vpriv -o %i) || fail "squeue failed"
[ -z "$Q" ] || { echo "vpriv job $Q already queued: nothing to do"; exit 0; }
[ -e runs/vpriv/args.json ] || [ -e runs/vpriv/ckpt.pt ] && fail "runs/vpriv already has a run"
sha256sum -c --quiet council/vpriv_deploy.sha256 || fail "deployed files differ from the reviewed manifest"
check_log() {  # <log> <result prefix> <expected dir>
  [ -f "$1" ] || fail "missing log $1"
  tail -n 1 "$1" | grep -qxE "$2 FAILS=0 dir=$3" || fail "$1: last line is not '$2 FAILS=0 dir=$3' ($(tail -n 1 $1 | cut -c1-120))"
}
check_log logs2/ctests-h200-$CTID.out CAUSAL_TESTS_SH runs/ctest_$CTID
check_log logs2/ct2-$T2ID.out CAUSAL_T2 runs/ctest_$T2ID
check_log logs2/vprivA-$AID.out "VPRIV_TESTS part a" runs/ctest_vpriv_$AID
check_log logs2/vprivB-$BID.out "VPRIV_TESTS part b" runs/ctest_vpriv_$BID
# exact file sets (Astra launch review 2): suite 13 files, each fixture part 6 files; every line verbatim in the deployment
SUITE="mrv/causal_video.py mrv/noise.py mrv/sd.py scripts/causal_rollout.py scripts/causal_tests.py scripts/det_run.py scripts/rho_decision.py scripts/rho_rule.py scripts/test_noise.py scripts/test_rho_rule.py scripts/train_causal.py slurm/causal_tests.sh slurm/launch_vlive.sh"
FIX="mrv/causal_video.py mrv/noise.py mrv/sd.py scripts/causal_rollout.py scripts/train_causal.py slurm/vpriv_tests.sh"
for x in "runs/ctest_$CTID/code_sha256.txt:$SUITE" "runs/ctest_vpriv_$AID/code_sha256.txt:$FIX" "runs/ctest_vpriv_$BID/code_sha256.txt:$FIX"; do
  M=${x%%:*}; WANT=$(echo ${x#*:} | tr ' ' '\n' | sort)
  [ -s $M ] || fail "missing manifest $M"
  GOT=$(awk '{print $2}' $M | sort); [ "$GOT" = "$WANT" ] || fail "$M: file set differs from the reviewed set"
  [ $(awk '{print $2}' $M | sort -u | wc -l) = $(wc -l < $M) ] || fail "$M: duplicate entries"
  while read -r line; do grep -qxF "$line" council/vpriv_deploy.sha256 || fail "$M: '$line' not in the reviewed deployment"; done < $M
done
OUT=$($PY scripts/a3_trigger.py --launch); rc=$?; echo "$OUT"
[ $rc = 0 ] || fail "a3_trigger exit $rc"
set -- $(echo "$OUT" | tail -n 1); PRIV=$1; REC=$2
case "$PRIV" in 1|2) ;; *) fail "A3 trigger not met ($OUT) -> back to council";; esac
[ -f "$REC/record.json" ] || fail "no trigger record in $REC"
$PY scripts/rho_decision.py --out council/rho_decision.json || fail "rho decision not valid -> back to council"
RHO=$($PY -c "import json;print(json.load(open('council/rho_decision.json'))['chosen'])")
[ "$RHO" = "0.95" ] || fail "rollout rho $RHO != vlive's 0.95"
cp council/rho_decision.json $REC/rho_decision.json
mkdir -p runs/vpriv; lfs setstripe -i 0 -c 1 runs/vpriv
echo "intent $(date -u +%FT%TZ) record=$REC tests=$CTID,$T2ID,$AID,$BID" > $INTENT && [ -s $INTENT ] || fail "cannot write the intent file"
J=$(sbatch --parsable -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=6:00:00 -J vpriv -o logs2/%x-%j.out \
    slurm/py.sbatch scripts/train_causal.py --init runs/longA/ckpt_20000.pt --out runs/vpriv --K 4 --rho $RHO \
    --rho_decision council/rho_decision.json --private_clips $PRIV --steps 16000)
[[ "$J" =~ ^[0-9]+$ ]] || fail "sbatch did not return a job id ('$J'): the intent file stays; check sacct before any retry"
R="job=$J private_clips=$PRIV rho=$RHO record=$REC tests=$CTID,$T2ID,$AID,$BID"
echo "$R" > $MARK && echo "$R" > $REC/job && grep -qx "$R" $MARK && grep -qx "$R" $REC/job || { echo "WARNING: training job $J SUBMITTED but the receipt write failed; record by hand"; exit 2; }
echo "GATE PASS: tests clean, A3 trigger met, private_clips $PRIV, rollout rho $RHO; training job vpriv $J; record $REC"
