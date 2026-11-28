#!/bin/bash
# Slice handoff vctl -> vmix (council: vmix-pending-final-*.md, Astra 05:30 (b)/(d)). Runs only after the batch-1 cache check
# succeeded (sbatch -d afterok:<cachecheck>). Steps:
#   1. wait for the NEXT completed rolling save of runs/vctl/ckpt.pt (atomic replace every 500 updates): its step must advance
#      past the step seen at start (if that save is in [7000, 8000), keep waiting for the 8000 save: Fable 05:30); then cancel the vctl training job and wait until it has left the queue;
#   2. verify the saved control checkpoint (step multiple of 500, optimizer + RNG state present) so vctl can resume later under its
#      original control record;
#   3. run slurm/launch_vmix.sh <suite> <t2> passed (lock, gate record with the PASSED cache, --verify, intent / receipt, one
#      H200-3g job to 12k), then submit the vmix watcher.
#   sbatch -d afterok:<cachecheck> --cpus-per-task=1 --mem=8G --time=1:30:00 -J handoff-vmix -o logs2/%x-%j.out \
#          slurm/handoff_vmix.sh <suite id> <t2 id>
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD; PY=.venv/bin/python
SID=$1; TID=$2; fail() { echo "HANDOFF FAIL: $*"; exit 1; }
for x in "$SID" "$TID"; do [[ "$x" =~ ^[0-9]+$ ]] || fail "job ids must be numeric"; done
step() { $PY -c "import torch;print(torch.load('runs/vctl/ckpt.pt',map_location='cpu',weights_only=False,mmap=True)['step'])" 2>/dev/null; }
Q=$(squeue -u $USER -h -n vctl -o %i) || fail "squeue failed"
if [ -n "$Q" ]; then
  S0=$(step) || fail "cannot read runs/vctl/ckpt.pt"; echo "$(date -u +%H:%M) vctl job $Q at saved step $S0; waiting for the next save"
  for i in $(seq 1 180); do  # <= 90 min
    sleep 30; S=$(step) || continue
    [ "$S" -gt "$S0" ] 2>/dev/null || continue
    # Fable 05:30: if the control is in [7000, 8000) let it reach its 8k save (the first gated checkpoint), else yield now
    if [ "$S" -ge 7000 ] && [ "$S" -lt 8000 ]; then continue; fi
    break
  done
  [ "$S" -gt "$S0" ] 2>/dev/null || fail "no new vctl save within 90 min (still $S)"
  echo "$(date -u +%H:%M) new vctl save at step $S: cancelling $Q"; scancel $Q
  for i in $(seq 1 40); do [ -z "$(squeue -u $USER -h -n vctl -o %i)" ] && break; sleep 15; done
  [ -z "$(squeue -u $USER -h -n vctl -o %i)" ] || fail "vctl job did not leave the queue"
else
  echo "no vctl job queued (finished or never started)"
fi
$PY - <<'EOF' || fail "control checkpoint check failed"
import torch
c = torch.load("runs/vctl/ckpt.pt", map_location="cpu", weights_only=False, mmap=True)
assert c["step"] % 500 == 0 or c["step"] == 16000, c["step"]
assert "opt" in c and c["opt"], "no optimizer state"
print("vctl checkpoint OK: step", c["step"], "keys", sorted(k for k in c if "rng" in k.lower()))
EOF
mkdir -p runs/vmix_gate; echo "vctl yielded at step $(step); see runs/vctl for resume (same --gate_record, --resume)" > runs/vmix_gate/vctl_yield.txt
bash slurm/launch_vmix.sh $SID $TID passed || fail "launch_vmix failed"
W=$(sbatch --parsable --cpus-per-task=1 --mem=8G --time=9:00:00 -J vmix-watch -o logs2/%x-%j.out slurm/vmix_checks.sh) || fail "watcher sbatch failed"
echo "HANDOFF DONE: vmix launched; watcher $W"
