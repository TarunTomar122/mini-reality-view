#!/bin/bash
# GPU fixtures for A3 vpriv (Astra council/codereview-vpriv-astra.md items 1, 2, 4). Usage: sbatch ... slurm/vpriv_tests.sh <part a|b>
#  a: matched start + modes   -- (1) init determinism: the temporal-block init hash is identical in two processes and
#     recorded; (2) private 0 reproduces vlive's update-100 batch_sha EXACTLY (data / t / noise draws) and its loss closely;
#     (3) private 0 / 1 / 2 for 100 updates: identical cumulative pick hash (picks, t, dropout), frozen guard passes in each.
#  b: resume + rollout -- (4) private 1: 3 updates straight vs 1 update + resume to 3: same update-3 batch_sha, loss within
#     1e-3 relative; resuming with --private_clips 2 must fail before any update; (5) rollout from a private-2 checkpoint
#     reads rho 0.95 and records it.
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
PY=.venv/bin/python; A=runs/longA/ckpt_20000.pt; T=runs/ctest_vpriv_${SLURM_JOB_ID:-manual}; mkdir -p $T; FAILS=0
chk() { if [ "$1" != 0 ]; then echo "FAIL $2 (exit $1)"; FAILS=$((FAILS+1)); fi; }
TR="$PY scripts/train_causal.py --init $A --K 4 --rho 0.95 --steps 100000 --timing"
if [ "$1" = a ]; then
  for m in 0 1 2; do $TR --out $T/timing_m$m --private_clips $m --max_steps_this_job 100 > $T/m$m.log 2>&1; chk $? "train private $m"; done
  $PY - <<P; chk $? part-a
import json, sys
L = lambda d: [json.loads(l) for l in open(f"$T/{d}/log.jsonl") if l.startswith('{"update"')]
V = [json.loads(l) for l in open("runs/vlive/log.jsonl") if l.startswith('{"update"')]
v100 = [r for r in V if r["update"] == 100][0]
r = {m: L(f"timing_m{m}") for m in (0, 1, 2)}
a = {m: json.load(open(f"$T/timing_m{m}/args.json")) for m in (0, 1, 2)}
vi = json.load(open("runs/vlive/args.json")).get("init_sha")
init_same = a[0]["init_sha"] == a[1]["init_sha"] == a[2]["init_sha"]
m0 = [x for x in r[0] if x["update"] == 100][0]
ok1 = init_same; ok2 = m0["batch_sha"] == v100["batch_sha"] and abs(m0["loss"] - v100["loss"]) <= 2e-4
p100 = {m: [x for x in r[m] if x["update"] == 100][0]["pick_sha_cum"] for m in (0, 1, 2)}
ok3 = len(set(p100.values())) == 1 and all("frozen check after first update" in open(f"$T/m{m}.log").read() for m in (0, 1, 2))
print(("PASS" if ok1 else "FAIL"), "init hash identical across 3 processes:", a[0]["init_sha"], "(vlive args has none:", vi, ")")
print(("PASS" if ok2 else "FAIL"), "private 0 vs vlive update 100: batch_sha", m0["batch_sha"], v100["batch_sha"], "loss", m0["loss"], v100["loss"])
print(("PASS" if ok3 else "FAIL"), "modes 0/1/2: cumulative pick hash at 100", p100, "+ frozen guard in all three")
sys.exit(0 if ok1 and ok2 and ok3 else 1)
P
else
  $TR --out $T/timing_full --private_clips 1 --max_steps_this_job 3 > $T/full.log 2>&1; chk $? full
  $TR --out $T/timing_split --private_clips 1 --max_steps_this_job 1 > $T/split1.log 2>&1; chk $? split1
  $TR --out $T/timing_split --private_clips 1 --max_steps_this_job 2 --resume > $T/split2.log 2>&1; chk $? split2
  $TR --out $T/timing_split --private_clips 2 --max_steps_this_job 1 --resume > $T/bad.log 2>&1; rb=$?
  $TR --out $T/timing_p2 --private_clips 2 --max_steps_this_job 1 > $T/p2.log 2>&1; chk $? p2
  echo scene-0043 > $T/one.txt
  $PY scripts/causal_rollout.py --ckpt $T/timing_p2/ckpt.pt --split dev --drive_list $T/one.txt --seeds 0 --max_ticks 3 --out $T/roll_p2 > $T/roll.log 2>&1; chk $? rollout
  $PY - <<P; chk $? part-b
import json, sys, torch
L = lambda d: [json.loads(l) for l in open(f"$T/{d}/log.jsonl") if l.startswith('{"update"')]
f3 = [x for x in L("timing_full") if x["update"] == 3][0]; s3 = [x for x in L("timing_split") if x["update"] == 3][0]
ok4 = f3["batch_sha"] == s3["batch_sha"] and abs(f3["loss"] - s3["loss"]) <= 1e-3 * max(abs(f3["loss"]), 1e-6) + 1e-4
bad = open("$T/bad.log").read(); ok4b = $rb != 0 and "incompatible resume" in bad and len(L("timing_split")) == len([1 for l in open("$T/timing_split/log.jsonl") if l.startswith('{"update"')])
ck = torch.load("$T/timing_p2/ckpt.pt", map_location="cpu"); ro = torch.load("$T/roll_p2/scene-0043_s0.pt")
ok5 = ck.get("rho") == 0.95 and ck.get("train_noise", {}).get("private_clips") == 2 and ro.get("rho") == 0.95
print(("PASS" if ok4 else "FAIL"), "resume (private 1, split after odd update 1): update-3 batch_sha", f3["batch_sha"], s3["batch_sha"], "loss", f3["loss"], s3["loss"])
print(("PASS" if ok4b else "FAIL"), "resume with a changed private count fails before an update (exit $rb)")
print(("PASS" if ok5 else "FAIL"), "private-2 checkpoint: rollout rho", ro.get("rho"), "train_noise", ck.get("train_noise"))
sys.exit(0 if ok4 and ok4b and ok5 else 1)
P
fi
echo "VPRIV_TESTS part $1 FAILS=$FAILS dir=$T"; exit $FAILS
