#!/bin/bash
# Slice handoff vctl -> vmix, v2 (council/codereview-handoff-astra.md FIX; replaces slurm/handoff_vmix.sh job 3756468 under the
# same afterok:<cache check> dependency). Never touches suite-tested files.
#   0. one handoff at a time (flock runs/handoff/.lock); every phase has an intent before and a receipt after, so a retry
#      reconciles instead of repeating (YIELD_PENDING / YIELDED, then vmix's own SUBMITTED, then WATCH_PENDING / WATCH_SUBMITTED);
#   1. the vctl job ID and gate record come ONLY from the launch receipt runs/vctl_gate/SUBMITTED (== <record>/job);
#   2. yield predicate: the next rolling save T after the step seen at start (T = next multiple of 500); a save in [7000, 8000)
#      is pushed to T = 8000; at a scheduled milestone (train_clean.py save_at: 1000, 2000, 4000, 6000, 8000, ..., 16000) the
#      NUMBERED ckpt_<T>.pt must exist, load and carry step T (the trainer writes ckpt.pt, runs its probe, THEN the numbered file),
#      so the 8k scoring checkpoint is never lost. Polling has its own budget; on timeout the full predicate is applied to the
#      current state (never 7000 / 7500, never a milestone without its numbered file) and the handoff fails without cancelling;
#   3. before cancelling: the eligible checkpoint is validated (step, gate record sha256 == the receipt's gate file, control recipe
#      arm vclean / jitter [1, 41, 81] / no cache, 16 finite temporal state dicts, populated Adam state for every temporal
#      parameter, restorable jitter / source RNG states; at a milestone the numbered file holds the same weights and RNG states);
#   4. scancel of THAT job ID only (return code checked), then sacct must report a terminal state; after the writer exited the
#      checkpoint is re-read, the predicate re-applied to the FINAL step, and step + sha256 of ckpt.pt (and the numbered file)
#      are recorded in runs/handoff/vctl_yield.json (used by slurm/launch_vctl_resume.sh);
#      if vctl already finished (or ended), its last checkpoint is validated and recorded the same way;
#   5. slurm/launch_vmix.sh <suite> <t2> passed, then the vmix receipt runs/vmix_gate/SUBMITTED is REQUIRED (its job ID must be
#      queued / running / completed); only then ONE watcher (slurm/vmix_checks.sh) under its own intent / receipt.
# Wall time: polling <= POLL_MIN (default 95) + cancel / confirm <= 15 min + validation / gate / launch <= 25 min -> --time 2:30:00.
#   sbatch -d afterok:<cache check> --cpus-per-task=1 --mem=8G --time=2:30:00 -J handoff-vmix2 -o logs2/%x-%j.out \
#          slurm/handoff_vmix_v2.sh <suite id> <t2 id>
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1
mkdir -p $PWD/runs/tmp && export TMPDIR=$PWD/runs/tmp
PY=.venv/bin/python; SID=$1; TID=$2; POLL_MIN=${POLL_MIN:-95}; H=runs/handoff; VG=runs/vctl_gate; MG=runs/vmix_gate
T_START=$(date +%s)
fail() { echo "HANDOFF FAIL: $*"; exit 1; }
for x in "$SID" "$TID"; do [[ "$x" =~ ^[0-9]+$ ]] || fail "job ids must be numeric"; done
mkdir -p $H || fail "cannot create $H"
exec 9>$H/.lock; flock -n 9 || { echo "another handoff holds the lock: exiting"; exit 0; }
MILESTONES=" 1000 2000 4000 6000 8000 10000 12000 14000 16000 "
jobstate() {  # active | done | ended (other terminal) | unknown; return 1 if both queries fail
  local s; s=$(squeue -h -j $1 -o %T 2>/dev/null); local r1=$?
  [ -n "$s" ] && { echo active; return 0; }
  s=$(sacct -n -X -P -S now-3days -j $1 -o State 2>/dev/null | head -1); local r2=$?
  [ $r1 != 0 ] && [ $r2 != 0 ] && return 1
  case "$s" in COMPLETED) echo done;; PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING|COMPLETING|RESIZING) echo active;; "") echo unknown;; *) echo ended;; esac; }
# ---------- 1. the vctl job and gate from the launch receipt ----------
[ -s $VG/SUBMITTED ] || fail "no vctl receipt $VG/SUBMITTED"
LINE=$(cat $VG/SUBMITTED); [ $(echo "$LINE" | grep -c .) = 1 ] || fail "vctl receipt is not one line"
VJ=$(echo "$LINE" | grep -o "^job=[0-9]*" | cut -d= -f2); GREC=$(echo "$LINE" | grep -o " gate=[^ ]*" | cut -d= -f2)
VREC=$(echo "$LINE" | grep -o " record=[^ ]*" | cut -d= -f2)
[[ "$VJ" =~ ^[0-9]+$ ]] && [ -s "$GREC" ] && [ -s "$VREC/job" ] || fail "vctl receipt incomplete ($LINE)"
grep -qxF "$LINE" $VREC/job || fail "vctl receipt differs from $VREC/job"
GSHA=$(sha256sum $GREC | cut -c1-64)
echo "$(date -u +%H:%M) vctl job $VJ, gate $GREC ($GSHA)"
# validate <ckpt> <step> [<numbered>]: the resume-relevant content of the control checkpoint
validate() { $PY - "$1" "$2" "$GSHA" "${3:-}" <<'EOF'
import math, sys, torch
path, step, gsha, numbered = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
def chk(c, m):
    if not c:
        print("VALIDATE_FAIL", m); sys.exit(1)
c = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
chk(c.get("step") == step, f"{path}: step {c.get('step')} != {step}")
chk(c.get("gate_record_sha256") == gsha, f"{path}: gate sha {c.get('gate_record_sha256')} != receipt gate {gsha}")
chk(c.get("arm") == "vclean" and c.get("recipe") == "vclean_keyjit_control" and c.get("jitter") == [1, 41, 81] and not c.get("mix_from")
    and c.get("cache_record_sha256") is None and c.get("timing") is False, f"{path}: not the control recipe ({c.get('arm')}, {c.get('recipe')}, {c.get('jitter')})")
T = c.get("temporal") or []
chk(len(T) == 16 and all(sd and all(torch.is_tensor(v) and bool(torch.isfinite(v.float()).all()) for v in sd.values()) for sd in T), f"{path}: temporal weights")
n_par = sum(len(sd) for sd in T)
st = (c.get("opt") or {}).get("state") or {}
chk(len(st) == n_par and all("exp_avg" in s and "exp_avg_sq" in s and float(s.get("step", 0)) > 0 for s in st.values()), f"{path}: Adam state {len(st)} / {n_par} params")
r = c.get("rng") or {}
for k in ("jitter", "source"):
    g = torch.Generator()
    try:
        g.set_state(r[k])
    except Exception as e:
        chk(False, f"{path}: {k} RNG state not restorable ({e})")
if numbered:
    n = torch.load(numbered, map_location="cpu", weights_only=False, mmap=True)
    chk(n.get("step") == step and n.get("gate_record_sha256") == gsha, f"{numbered}: step / gate")
    chk(all(torch.equal(a[k_], b[k_]) for a, b in zip(T, n["temporal"]) for k_ in a), f"{numbered}: weights differ from ckpt.pt")
    chk(all(torch.equal(r[k], n["rng"][k]) for k in ("jitter", "source")), f"{numbered}: RNG states differ from ckpt.pt")
print("VALIDATE_OK", path, step, n_par, "params")
EOF
}
cstep() { $PY -c "import torch;print(torch.load('$1',map_location='cpu',weights_only=False,mmap=True)['step'])" 2>/dev/null; }
eligible() {  # eligible <rolling step> <target>: prints the numbered file ("-" if not a milestone); 1 if not (yet) eligible
  local s=$1 t=$2
  [[ "$s" =~ ^[0-9]+$ ]] || return 1
  [ "$s" -ge "$t" ] || return 1
  [ "$s" -ge 7000 ] && [ "$s" -lt 8000 ] && return 1          # never 7000 / 7500 (Fable 05:30: let it reach 8k)
  if [[ "$MILESTONES" == *" $s "* ]]; then
    [ "$(cstep runs/vctl/ckpt_$s.pt)" = "$s" ] || return 1    # numbered milestone file written and loadable
    echo runs/vctl/ckpt_$s.pt
  else echo -; fi; }
record_yield() {  # record_yield <step> <numbered|-> <how>
  local f=$H/vctl_yield.json n=$2
  $PY - "$1" "$n" "$3" "$VJ" "$GREC" "$GSHA" <<'EOF' || fail "cannot write the yield record"
import hashlib, json, sys
step, n, how, vj, grec, gsha = sys.argv[1:7]
h = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
rec = {"vctl_job": int(vj), "step": int(step), "ckpt": "runs/vctl/ckpt.pt", "ckpt_sha256": h("runs/vctl/ckpt.pt"), "how": how,
       "numbered": None if n == "-" else n, "numbered_sha256": None if n == "-" else h(n), "gate_record": grec, "gate_sha256": gsha}
json.dump(rec, open("runs/handoff/vctl_yield.json.tmp", "w"), indent=1)
import os; os.replace("runs/handoff/vctl_yield.json.tmp", "runs/handoff/vctl_yield.json"); print("YIELD_RECORD", json.dumps(rec))
EOF
  echo "$(date -u +%FT%TZ) step=$1 how=$3 job=$VJ" > $H/YIELDED && [ -s $H/YIELDED ] || fail "cannot write YIELDED"; }
# ---------- 2-4. yield (skipped if an earlier attempt already recorded it) ----------
if [ -s $H/YIELDED ] && [ -s $H/vctl_yield.json ]; then
  echo "yield already recorded ($(cat $H/YIELDED))"
else
  ST=$(jobstate $VJ) || fail "squeue and sacct both failed for $VJ"
  if [ -e $H/YIELD_PENDING ] && [ "$ST" = active ]; then fail "earlier cancel intent ($(cat $H/YIELD_PENDING)) but job $VJ still active: reconcile by hand"; fi
  if [ "$ST" = active ]; then
    S0=$(cstep runs/vctl/ckpt.pt); [[ "$S0" =~ ^[0-9]+$ ]] || S0=0
    T=$(( (S0 / 500 + 1) * 500 )); [ $T -ge 7000 ] && [ $T -lt 8000 ] && T=8000
    echo "$(date -u +%H:%M) vctl at saved step $S0: yield at the next eligible save >= $T (milestones need their numbered file)"
    DEADLINE=$(( $(date +%s) + POLL_MIN * 60 )); N=""; S=""
    while [ $(date +%s) -lt $DEADLINE ]; do
      ST=$(jobstate $VJ) || { sleep 20; continue; }
      [ "$ST" = active ] || break                                     # finished / ended while waiting: handled below
      S=$(cstep runs/vctl/ckpt.pt); N=$(eligible "$S" $T) && break
      sleep 20
    done
    if [ "$ST" = active ]; then
      S=$(cstep runs/vctl/ckpt.pt); N=$(eligible "$S" $T) || fail "polling budget ($POLL_MIN min) spent: rolling step '$S' does not meet the yield predicate (target $T): vctl NOT cancelled"
      validate runs/vctl/ckpt.pt $S $([ "$N" = - ] || echo $N) || fail "eligible checkpoint at $S failed validation: vctl NOT cancelled"
      echo "intent $(date -u +%FT%TZ) cancel job=$VJ at step=$S" > $H/YIELD_PENDING && [ -s $H/YIELD_PENDING ] || fail "cannot write the cancel intent"
      scancel $VJ; rc=$?; [ $rc = 0 ] || fail "scancel $VJ returned $rc"
      for i in $(seq 1 60); do ST=$(jobstate $VJ) && [ "$ST" != active ] && [ "$ST" != unknown ] && break; sleep 15; done
      [ "$ST" = ended ] || [ "$ST" = done ] || fail "job $VJ not terminal 15 min after scancel (state '$ST')"
      echo "$(date -u +%H:%M) job $VJ terminal ($(sacct -n -X -P -j $VJ -o State | head -1))"
      HOW=yielded
    else
      HOW="vctl $ST before the yield"
    fi
  else
    HOW="vctl $ST at handoff start"
  fi
  # the writer has exited: re-read the FINAL checkpoint and re-apply the predicate to it
  SF=$(cstep runs/vctl/ckpt.pt); [[ "$SF" =~ ^[0-9]+$ ]] || fail "runs/vctl/ckpt.pt unreadable after the writer exited"
  if [ "$HOW" = yielded ]; then
    [ "$SF" = "$S" ] || { NF=$(eligible "$SF" $T) || fail "final step $SF differs from the accepted $S and fails the predicate: reconcile by hand"; N=$NF; }
  else
    if [ "$SF" = 16000 ]; then N=$([ "$(cstep runs/vctl/ckpt_16000.pt)" = 16000 ] && echo runs/vctl/ckpt_16000.pt || echo -)
    elif [[ "$MILESTONES" == *" $SF "* ]] && [ "$(cstep runs/vctl/ckpt_$SF.pt)" = "$SF" ]; then N=runs/vctl/ckpt_$SF.pt
    else N=-; fi
  fi
  validate runs/vctl/ckpt.pt $SF $([ "$N" = - ] || echo $N) || fail "final control checkpoint at $SF failed validation"
  record_yield $SF $N "$HOW"
fi
# ---------- 5. vmix (passed mode), receipt required ----------
if [ ! -s $MG/SUBMITTED ]; then
  bash slurm/launch_vmix.sh $SID $TID passed; rc=$?
  echo "launch_vmix exit $rc"
fi
[ -s $MG/SUBMITTED ] || fail "launch_vmix left no receipt $MG/SUBMITTED (it may exit 0 without submitting): see its output above"
MJ=$(grep -o "^job=[0-9]*" $MG/SUBMITTED | cut -d= -f2); [[ "$MJ" =~ ^[0-9]+$ ]] || fail "vmix receipt has no job id"
grep -q " cache=passed " $MG/SUBMITTED || fail "vmix receipt is not a passed-cache launch ($(cat $MG/SUBMITTED))"
MS=$(jobstate $MJ) || fail "cannot query vmix job $MJ"
case $MS in active|done) echo "vmix job $MJ: $MS";; *) fail "vmix job $MJ is '$MS'";; esac
# ---------- 6. ONE watcher under its own intent / receipt ----------
if [ -s $H/WATCH_SUBMITTED ]; then echo "watcher already submitted ($(cat $H/WATCH_SUBMITTED))"
else
  [ -e $H/WATCH_PENDING ] && fail "earlier watcher intent ($(cat $H/WATCH_PENDING)) without receipt: reconcile by hand (sacct --name=vmix-watch)"
  Q=$(squeue -u $USER -h -n vmix-watch -o %i) || fail "squeue failed"
  [ -z "$Q" ] || { echo "job=$Q adopted" > $H/WATCH_SUBMITTED; echo "vmix watcher $Q already queued: adopted"; Q=adopted; }
  if [ "$Q" != adopted ]; then
    echo "intent $(date -u +%FT%TZ) vmix=$MJ" > $H/WATCH_PENDING && [ -s $H/WATCH_PENDING ] || fail "cannot write the watcher intent"
    W=$(sbatch --parsable --cpus-per-task=1 --mem=8G --time=9:00:00 -J vmix-watch -o logs2/%x-%j.out slurm/vmix_checks.sh)
    [[ "$W" =~ ^[0-9]+$ ]] || fail "watcher sbatch returned '$W': intent kept; check sacct --name=vmix-watch"
    echo "job=$W vmix=$MJ" > $H/WATCH_SUBMITTED && grep -q "job=$W" $H/WATCH_SUBMITTED || fail "watcher $W submitted but receipt failed"
  fi
fi
echo "HANDOFF DONE in $(( ($(date +%s) - T_START) / 60 )) min: vctl $(cat $H/YIELDED); vmix job $MJ; watcher $(cat $H/WATCH_SUBMITTED)"
