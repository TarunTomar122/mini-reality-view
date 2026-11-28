#!/bin/bash
# Resume the A5 attribution control "vctl" ONCE after vmix is finished (council/codereview-handoff-astra.md). Never touches
# suite-tested files. Conditions (all checked under the vctl gate lock, separate continuation intent / receipt):
#   * vmix is over: every vmix training job in its receipts (runs/vmix_gate/SUBMITTED, CONTINUED) is terminal, and either the
#     A1 stop was taken (runs/vmix_gate/A1_STOPPED) or runs/vmix/ckpt.pt is at step 16000;
#   * the handoff record runs/handoff/vctl_yield.json exists; runs/vctl/ckpt.pt EXISTS and its step and sha256 equal the recorded
#     ones (train_clean.py silently starts from zero when ckpt.pt is missing), and it still validates as the control checkpoint
#     made under the ORIGINAL gate record (sha256 recorded in the checkpoint == the gate file named by runs/vctl_gate/SUBMITTED);
#   * the job re-checks step + sha256 of runs/vctl/ckpt.pt at START, before the trainer runs (exit 3 otherwise);
#   * the trainer gets the ORIGINAL gate record, init, rho and output, --arm vclean --jitter 1,41,81 --resume --steps 16000 (its
#     own resume identity re-checks gate digest / approval / pin).
#   intent runs/vctl_gate/RESUME_PENDING before sbatch, receipt runs/vctl_gate/RESUMED after (a retry never resubmits blindly).
#   sbatch --cpus-per-task=1 --mem=4G --time=0:30:00 -J launch-vctl-resume -o logs2/%x-%j.out slurm/launch_vctl_resume.sh
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1
mkdir -p $PWD/runs/tmp && export TMPDIR=$PWD/runs/tmp
PY=.venv/bin/python; VG=runs/vctl_gate; MG=runs/vmix_gate; Y=runs/handoff/vctl_yield.json
fail() { echo "RESUME FAIL: $*"; exit 1; }
exec 9>$VG/.lock; flock -n 9 || { echo "another vctl launcher holds the lock: nothing to do"; exit 0; }
[ -s $VG/RESUMED ] && { echo "vctl already resumed ($(cat $VG/RESUMED)): nothing to do"; exit 0; }
[ -e $VG/RESUME_PENDING ] && fail "ambiguous earlier resume ($(cat $VG/RESUME_PENDING)): reconcile by hand (sacct -n -X --name=vctl)"
jobstate() {  # active | done | ended | unknown; 1 if both queries fail (squeue errors on purged ids, so sacct decides)
  local s; s=$(squeue -h -j $1 -o %T 2>/dev/null); local r1=$?
  [ -n "$s" ] && { echo active; return 0; }
  s=$(sacct -n -X -P -S now-7days -j $1 -o State 2>/dev/null | head -1); local r2=$?
  [ $r1 != 0 ] && [ $r2 != 0 ] && return 1
  case "$s" in COMPLETED) echo done;; PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING|COMPLETING|RESIZING) echo active;; "") echo unknown;; *) echo ended;; esac; }
# ---------- vmix is over ----------
[ -s $MG/SUBMITTED ] || fail "no vmix receipt"
for f in $MG/SUBMITTED $MG/CONTINUED; do
  [ -s $f ] || continue
  j=$(grep -o "^job=[0-9]*" $f | cut -d= -f2); [[ "$j" =~ ^[0-9]+$ ]] || fail "no job id in $f"
  st=$(jobstate $j) || fail "cannot query vmix job $j"; [ "$st" = done ] || [ "$st" = ended ] || fail "vmix job $j ($f) is '$st': vmix not finished"
done
Q=$(squeue -u $USER -h -n vmix -o %i) || fail "squeue failed"; [ -z "$Q" ] || fail "a vmix job ($Q) is queued"
[ -e $MG/CONT_PENDING ] && [ ! -s $MG/CONTINUED ] && fail "vmix continuation intent without receipt: reconcile first"
if [ -e $MG/A1_STOPPED ]; then WHY="vmix A1 stop"
else
  VS=$($PY -c "import torch;print(torch.load('runs/vmix/ckpt.pt',map_location='cpu',weights_only=False,mmap=True)['step'])") || fail "cannot read runs/vmix/ckpt.pt"
  [ "$VS" = 16000 ] || fail "vmix at step $VS without an A1 stop: not finished"; WHY="vmix finished at 16000"
fi
# ---------- the recorded control checkpoint, under the original gate ----------
[ -s $VG/SUBMITTED ] || fail "no vctl receipt"
LINE=$(cat $VG/SUBMITTED); GREC=$(echo "$LINE" | grep -o " gate=[^ ]*" | cut -d= -f2); RHO=$(echo "$LINE" | grep -o " rho=[0-9.]*" | cut -d= -f2)
[ -s "$GREC" ] && [ "$RHO" = 1.0 ] || fail "vctl receipt gate '$GREC' / rho '$RHO'"
GSHA=$(sha256sum $GREC | cut -c1-64)
[ -s $Y ] || fail "no handoff record $Y"
[ -f runs/vctl/ckpt.pt ] || fail "runs/vctl/ckpt.pt is MISSING (the trainer would silently start at zero)"
read -r YSTEP YSHA YG <<< "$($PY -c "import json;r=json.load(open('$Y'));print(r['step'],r['ckpt_sha256'],r['gate_sha256'])")"
CSHA=$(sha256sum runs/vctl/ckpt.pt | cut -c1-64)
[ "$CSHA" = "$YSHA" ] || fail "runs/vctl/ckpt.pt sha256 $CSHA != recorded $YSHA"
[ "$YG" = "$GSHA" ] || fail "handoff record gate $YG != the receipt's gate file $GSHA"
$PY - "$YSTEP" "$GSHA" <<'EOF' || fail "control checkpoint does not validate"
import sys, torch
step, gsha = int(sys.argv[1]), sys.argv[2]
c = torch.load("runs/vctl/ckpt.pt", map_location="cpu", weights_only=False, mmap=True)
ok = (c.get("step") == step and c.get("gate_record_sha256") == gsha and c.get("arm") == "vclean" and c.get("recipe") == "vclean_keyjit_control"
      and c.get("jitter") == [1, 41, 81] and not c.get("mix_from") and c.get("timing") is False and len(c.get("temporal") or []) == 16
      and len((c.get("opt") or {}).get("state") or {}) > 0 and all(k in (c.get("rng") or {}) for k in ("jitter", "source")))
print("control checkpoint", "OK" if ok else "BAD", "step", c.get("step")); sys.exit(0 if ok else 1)
EOF
[ "$YSTEP" -lt 16000 ] || { echo "vctl already at 16000: nothing to resume"; exit 0; }
# ---------- one resume job (re-checks the checkpoint at start) ----------
echo "intent $(date -u +%FT%TZ) resume vctl from step=$YSTEP sha=$CSHA ($WHY)" > $VG/RESUME_PENDING && [ -s $VG/RESUME_PENDING ] || fail "cannot write the resume intent"
J=$(sbatch --parsable -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=6:00:00 -J vctl -o logs2/%x-%j.out --wrap "cd $HOME/mini-reality-view; \
    S=\$(sha256sum runs/vctl/ckpt.pt 2>/dev/null | cut -c1-64); [ \"\$S\" = $CSHA ] || { echo 'RESUME_ABORT runs/vctl/ckpt.pt missing or changed before start'; exit 3; }; \
    bash slurm/py.sbatch scripts/train_clean.py --arm vclean --jitter 1,41,81 --init runs/longA/ckpt_20000.pt --out runs/vctl --rho_gate $RHO \
    --gate_record $GREC --steps 16000 --resume")
[[ "$J" =~ ^[0-9]+$ ]] || fail "sbatch did not return a job id ('$J'): RESUME_PENDING kept; check sacct --name=vctl"
R="job=$J resume_from=$YSTEP ckpt_sha256=$CSHA gate=$GREC rho=$RHO reason=$WHY"
echo "$R" > $VG/RESUMED && grep -qxF "$R" $VG/RESUMED || { echo "WARNING: resume job $J submitted but the receipt write failed"; exit 2; }
echo "VCTL RESUMED: job $J from step $YSTEP ($WHY)"
