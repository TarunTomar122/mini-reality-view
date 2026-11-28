#!/bin/bash
# Throwaway resume test for the A5 control recipe "vctl" (--arm vclean --jitter 1,41,81; council/codereview-handoff-astra.md:
# "a separate control gate / verify and split-versus-uninterrupted resume check (draws, loss, weights, wrong-gate rejection)
# against the original suite-pinned code"). Never touches suite-tested files; writes only to runs/ctest_vctl_<job>.
#   0. code pin: the sha256 of every trainer file used here must appear verbatim in the passing clean suite's manifest;
#      the control gate record from runs/vctl_gate/SUBMITTED must re-verify (scripts/vclean_gate.py --verify);
#   1. --timing, deterministic kernels (scripts/det_run.py + CUBLAS_WORKSPACE_CONFIG): 3 updates straight vs 1 update + resume to 3:
#      updates 1-3 batch_sha, mem_sha, mem_t identical, loss within 1e-3 relative; final weights, Adam state and jitter / source
#      RNG states compared (weights / Adam: max |diff| reported, < 1e-5 required, bitwise reported; RNG: exact);
#   2. production-path gate binding (the real control gate record): a checkpoint stamped with ANOTHER gate digest is refused
#      before any update (log / checkpoint unchanged); the same checkpoint stamped with the real gate digest resumes 1 update.
#   sbatch --gres=gpu:nvidia_rtx_a6000:1 --cpus-per-task=8 --mem=96G --time=1:30:00 -J vctl-rtest -o logs2/%x-%j.out \
#          slurm/vctl_resume_test.sh <clean suite id>
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1
mkdir -p $PWD/runs/tmp && export TMPDIR=$PWD/runs/tmp
PY=.venv/bin/python; A=runs/longA/ckpt_20000.pt; SID=$1; T=runs/ctest_vctl_${SLURM_JOB_ID:-manual}; mkdir -p $T; FAILS=0
chk() { if [ "$1" != 0 ]; then echo "FAIL $2 (exit $1)"; FAILS=$((FAILS+1)); fi; }
[[ "$SID" =~ ^[0-9]+$ ]] || { echo "usage: vctl_resume_test.sh <clean suite id>"; exit 1; }
M=runs/ctest_clean_$SID/code_sha256.txt
sha256sum scripts/train_clean.py mrv/clean_video.py mrv/sd.py mrv/noise.py mrv/cache_verify.py council/vmix_pins.json scripts/vclean_gate.py \
  scripts/det_run.py slurm/vctl_resume_test.sh | tee $T/code_sha256.txt
echo "== code pin vs the clean suite manifest $M"
n=0; while read -r line; do case "$line" in *slurm/vctl_resume_test.sh) continue;; esac
  grep -qxF "$line" $M || { echo "not suite-pinned: $line"; n=$((n+1)); }; done < $T/code_sha256.txt
chk $n code-pin
LINE=$(cat runs/vctl_gate/SUBMITTED 2>/dev/null); GREC=$(echo "$LINE" | grep -o " gate=[^ ]*" | cut -d= -f2)
[ -s "$GREC" ] || { echo "FAIL no control gate record in runs/vctl_gate/SUBMITTED"; FAILS=$((FAILS+1)); }
GSHA=$(sha256sum $GREC | cut -c1-64)
$PY scripts/vclean_gate.py --verify $GREC; chk $? control-gate-verify
[ $FAILS = 0 ] || { echo "fail-fast"; echo "VCTL_RESUME_TEST FAILS=$FAILS dir=$T"; exit $FAILS; }
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TR="$PY scripts/det_run.py scripts/train_clean.py --arm vclean --jitter 1,41,81 --init $A --K 4 --steps 100000 --timing"
echo "== 1. split vs uninterrupted (control recipe, --timing)"
$TR --out $T/timing_full --max_steps_this_job 3 > $T/full.log 2>&1; chk $? full
$TR --out $T/timing_split --max_steps_this_job 1 > $T/split1.log 2>&1; chk $? split1
$TR --out $T/timing_split --max_steps_this_job 2 --resume > $T/split2.log 2>&1; chk $? split2
$PY - <<P; chk $? split-vs-straight
import json, sys, torch
L = lambda d: {x["update"]: x for x in (json.loads(l) for l in open(f"$T/{d}/log.jsonl") if l.startswith('{"update"'))}
F_, S_ = L("timing_full"), L("timing_split")
okd = all(u in F_ and u in S_ and all(F_[u][k] == S_[u][k] for k in ("batch_sha", "mem_sha", "mem_t", "mem_src"))
          and abs(F_[u]["loss"] - S_[u]["loss"]) <= 1e-3 * max(abs(F_[u]["loss"]), 1e-6) + 1e-4 for u in (1, 2, 3))
okj = all(t_ in (1, 41, 81) for u in (1, 2, 3) for t_ in F_.get(u, {}).get("mem_t", [0])) and all(F_.get(u, {}).get("mem_src") == ["key", "key"] for u in (1, 2, 3))
a = torch.load("$T/timing_full/ckpt.pt", map_location="cpu", weights_only=False); b = torch.load("$T/timing_split/ckpt.pt", map_location="cpu", weights_only=False)
dw = max((x[k] .float() - y[k].float()).abs().max().item() for x, y in zip(a["temporal"], b["temporal"]) for k in x)
bit = all(torch.equal(x[k], y[k]) for x, y in zip(a["temporal"], b["temporal"]) for k in x)
sa, sb = a["opt"]["state"], b["opt"]["state"]
da = max((sa[i][k].float() - sb[i][k].float()).abs().max().item() for i in sa for k in ("exp_avg", "exp_avg_sq")) if sorted(sa) == sorted(sb) else float("inf")
okr = all(torch.equal(a["rng"][k], b["rng"][k]) for k in ("jitter", "source"))
okw = a["step"] == b["step"] == 3 and dw < 1e-5 and da < 1e-5 and a.get("recipe") == b.get("recipe") == "vclean_keyjit_control"
print(("PASS" if okd and okj else "FAIL"), "updates 1-3 draws (batch_sha, mem_sha, mem_t, mem_src) identical, loss within 1e-3:",
      [(u, F_.get(u, {}).get("mem_sha"), S_.get(u, {}).get("mem_sha"), F_.get(u, {}).get("loss"), S_.get(u, {}).get("loss")) for u in (1, 2, 3)])
print(("PASS" if okw else "FAIL"), "update-3 weights max |diff| %.3g (bitwise %s), Adam max |diff| %.3g (< 1e-5), recipe %s" % (dw, bit, da, a.get("recipe")))
print(("PASS" if okr else "FAIL"), "jitter / source RNG states after update 3 identical")
sys.exit(0 if okd and okj and okw and okr else 1)
P
echo "== 2. production-path gate binding with the real control gate record"
$PY - <<P; chk $? craft
import os, torch
s = torch.load("$T/timing_full/ckpt.pt", map_location="cpu", weights_only=False)
for d, g in (("prod_wrong", "ab" * 32), ("prod_right", "$GSHA")):
    os.makedirs(f"$T/{d}", exist_ok=True); torch.save(dict(s, timing=False, gate_record_sha256=g), f"$T/{d}/ckpt.pt")
P
PR="$PY scripts/train_clean.py --arm vclean --jitter 1,41,81 --init $A --rho_gate 1.0 --gate_record $GREC --steps 4 --max_steps_this_job 1 --resume"
sw=$(sha256sum $T/prod_wrong/ckpt.pt | cut -c1-16)
$PR --out $T/prod_wrong > $T/prod_wrong.log 2>&1; rw=$?
sw2=$(sha256sum $T/prod_wrong/ckpt.pt | cut -c1-16)
$PR --out $T/prod_right > $T/prod_right.log 2>&1; rr=$?
$PY - <<P; chk $? gate-binding
import os, sys
w, r = open("$T/prod_wrong.log").read(), open("$T/prod_right.log").read()
okw = $rw != 0 and "checkpoint was made under gate record" in w and "$sw" == "$sw2" and not os.path.exists("$T/prod_wrong/log.jsonl")
okr = $rr == 0 and "resumed at 3" in r and "CLEAN_TRAIN_DONE 4" in r
print(("PASS" if okw else "FAIL"), "a checkpoint stamped with another gate digest is refused before any update (exit $rw; ckpt $sw -> $sw2)")
print(("PASS" if okr else "FAIL"), "the same checkpoint stamped with the real control gate digest resumes one update (exit $rr)")
sys.exit(0 if okw and okr else 1)
P
echo "VCTL_RESUME_TEST FAILS=$FAILS dir=$T"
exit $FAILS
