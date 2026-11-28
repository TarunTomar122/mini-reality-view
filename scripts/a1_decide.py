"""A1 decision for one gated vpriv check (council/AGREED-causal-r1.md gate + A1; Astra launch review 2 item 5).
Reads the signed compare runs/s_lock/cmp_P<k>k__A20k_indep_dev.json (written by the watcher's compare job), checks panel
completeness (6 drives, 12 runs, no unpaired, cold_ref B, finite values), and writes runs/vpriv_gate/a1_<k>k.json (mode "x"):
  joint pass  = ID and W eligible, smaller reduction > 9.4%, car F1 diff >= -0.03, late gap diff <= +0.03 -> PASS (advance to the
                30-drive exam; training is not stopped by a pass);
  safety miss = F1 or late gap bar missed. Two consecutive safety misses counted from 8k (8k -> 12k -> 16k) -> A1 stop.
Training is launched to 12k and PAUSES there (Astra launch review 3 item 3); slurm/vpriv_score.sh resumes it to 16k only if
the 12k decision has stop == false. So the stop needs no cancellation: it is "do not continue".
A retry reuses an existing decision file only if it equals the recomputed one.
2k / 4k are informational and never call this. Exit 0 = decision written; 3 = malformed / incomplete panel (no decision).
  .venv/bin/python scripts/a1_decide.py 8
"""
import json, math, os, subprocess, sys



def need(c, m):
    if not c:
        print("A1_ERR", m); sys.exit(3)


k = int(sys.argv[1]); need(k in (8, 12, 16), f"k {k} is not a gated check")
PREV = {8: None, 12: 8, 16: 12}[k]


f = f"runs/s_lock/cmp_P{k}k__A20k_indep_dev.json"
s = open(f).read(); need("TCOMPARE " in s, f"{f}: no comparator output")
d = json.loads(s.split("TCOMPARE ", 1)[1])
need(d["A"] == f"runs/troll/m_P{k}k_dev.json" and d["B"] == "runs/troll/m_A20k_indep_dev.json", f"{f}: identity")
need(d["drives"] == 6 and d["runs"] == 12 and not d["unpaired_drives"] and d["cold_ref"] == "B", f"{f}: incomplete panel")
for key, fl in (("ID", ("rel",)), ("W", ("rel",)), ("car_f1", ("diff",)), ("late_lpips_gap", ("diff",))):
    for x in fl:
        v = d[key][x]; need(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v), f"{key}.{x}={v!r}")
elig = d["ID"].get("can_pass") is True and d["W"].get("can_pass") is True
red = min(-d["ID"]["rel"], -d["W"]["rel"])
safe = d["car_f1"]["diff"] >= -0.03 and d["late_lpips_gap"]["diff"] <= 0.03
joint = elig and red > 0.094 and safe
prev_safe = True
if PREV:
    pf = f"runs/vpriv_gate/a1_{PREV}k.json"
    need(os.path.exists(pf), f"previous gated decision {pf} missing")
    prev_safe = json.load(open(pf))["safe"]
stop = (not safe) and (not prev_safe)
rec = {"k": k, "eligible": elig, "smaller_reduction": red, "f1_diff": d["car_f1"]["diff"], "late_diff": d["late_lpips_gap"]["diff"],
       "safe": safe, "joint_pass": joint, "prev_safe": prev_safe, "stop": stop}
out = f"runs/vpriv_gate/a1_{k}k.json"
if os.path.exists(out):  # a retry: reuse the decision only if it is exactly what the same panel gives now
    old = json.load(open(out))
    same = json.dumps(old, sort_keys=True) == json.dumps(rec, sort_keys=True)  # type-strict (1 != true)
    need(same and isinstance(old.get("stop"), bool), f"{out} exists and differs from the recomputed decision: reconcile by hand")
    print("A1_DECISION (existing, verified)", json.dumps(old)); sys.exit(0)
with open(out, "x") as fh:
    json.dump(rec, fh, indent=1)
print("A1_DECISION", json.dumps(rec))
