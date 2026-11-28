"""A1 decision for one gated vmix check (council/AGREED-causal-r1.md gate + A1 + A4 pass conditions, kept by A5). Copy of
scripts/a1_decide.py (vpriv; unchanged) adapted to vmix:
Reads the signed compare runs/s_lock/cmp_X<k>k__A20k_indep_dev.json (candidate runs/troll/m_X<k>k_dev.json vs signed A@20k; at
rollout rho 1.0 the memory-off cold IS signed A@20k), checks panel completeness (6 drives, 12 runs, no unpaired, cold_ref B,
finite values), and writes runs/vmix_gate/a1_<k>k.json (mode "x"):
  joint pass  = ID and W eligible, smaller reduction > 9.4%, car F1 diff >= -0.03, late gap diff <= +0.03 (the signed bars,
                unchanged) AND the A4 pass conditions at the same checkpoint, read from the comparator's PAIRED drive bootstrap
                (2,000 draws, equal drive weights, seeds pooled within drive; "ci95" = 2.5 / 97.5 percentiles of arm - A):
                  ghosts : ghost_ge5 "at or below the upper end of A's paired 95% interval" = the arm is not significantly above A
                           on the paired bootstrap, i.e. ghost_ge5.ci95[0] (lower end of the arm - A interval) <= 0;
                  road   : "the paired 95% interval of (arm - A) road IoU not wholly below zero", i.e. road_iou.ci95[1] >= 0.
                (Fable final r1 (d): road "stated the same way" as ghosts; both conditions are on the same paired interval.)
                A pass advances to the 30-drive exam only after N1 review; training is not stopped by a pass;
  safety miss = F1 or late gap bar missed (unchanged). Two consecutive safety misses counted from 8k (8k -> 12k -> 16k) -> A1 stop.
Ghost / road are pass conditions only: they never enter "safe" or "stop".
Training is launched to 12k and PAUSES there; slurm/vmix_score.sh resumes it to 16k only if the 12k decision has stop == false
(or early, after a COMPLETED safe 8k decision). A retry reuses an existing decision file only if it equals the recomputed one.
2k / 4k are informational and never call this. Exit 0 = decision written; 3 = malformed / incomplete panel (no decision).
  .venv/bin/python scripts/a1_decide_vmix.py 8
"""
import json, math, os, sys


def need(c, m):
    if not c:
        print("A1_ERR", m); sys.exit(3)


num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
k = int(sys.argv[1]); need(k in (8, 12, 16), f"k {k} is not a gated check")
PREV = {8: None, 12: 8, 16: 12}[k]
G = "runs/vmix_gate"

f = f"runs/s_lock/cmp_X{k}k__A20k_indep_dev.json"
s = open(f).read(); need("TCOMPARE " in s, f"{f}: no comparator output")
d = json.loads(s.split("TCOMPARE ", 1)[1])
need(d["A"] == f"runs/troll/m_X{k}k_dev.json" and d["B"] == "runs/troll/m_A20k_indep_dev.json", f"{f}: identity")
need(d["drives"] == 6 and d["runs"] == 12 and not d["unpaired_drives"] and d["cold_ref"] == "B", f"{f}: incomplete panel")
for key, fl in (("ID", ("rel",)), ("W", ("rel",)), ("car_f1", ("diff",)), ("late_lpips_gap", ("diff",)), ("ghost_ge5", ("A", "B", "diff")), ("road_iou", ("A", "B", "diff"))):
    for x in fl:
        v = d[key][x]; need(num(v), f"{key}.{x}={v!r}")
for key in ("ghost_ge5", "road_iou"):
    ci = d[key].get("ci95"); need(isinstance(ci, list) and len(ci) == 2 and all(num(v) for v in ci) and ci[0] <= ci[1], f"{key}.ci95={ci!r}")
elig = d["ID"].get("can_pass") is True and d["W"].get("can_pass") is True
red = min(-d["ID"]["rel"], -d["W"]["rel"])
safe = d["car_f1"]["diff"] >= -0.03 and d["late_lpips_gap"]["diff"] <= 0.03
ghost_ok = d["ghost_ge5"]["ci95"][0] <= 0.0   # arm - A ghost interval not wholly above zero
road_ok = d["road_iou"]["ci95"][1] >= 0.0     # arm - A road IoU interval not wholly below zero
bars = elig and red > 0.094 and safe
joint = bars and ghost_ok and road_ok
prev_safe = True
if PREV:
    pf = f"{G}/a1_{PREV}k.json"
    need(os.path.exists(pf), f"previous gated decision {pf} missing")
    prev_safe = json.load(open(pf))["safe"]
stop = (not safe) and (not prev_safe)
rec = {"k": k, "eligible": elig, "smaller_reduction": red, "f1_diff": d["car_f1"]["diff"], "late_diff": d["late_lpips_gap"]["diff"],
       "safe": safe, "signed_bars_pass": bars, "ghost_ge5": {"arm": d["ghost_ge5"]["A"], "A": d["ghost_ge5"]["B"], "diff_ci95": d["ghost_ge5"]["ci95"], "ok": ghost_ok},
       "road_iou": {"arm": d["road_iou"]["A"], "A": d["road_iou"]["B"], "diff_ci95": d["road_iou"]["ci95"], "ok": road_ok},
       "joint_pass": joint, "prev_safe": prev_safe, "stop": stop, "n1_review_before_exam": joint}
out = f"{G}/a1_{k}k.json"
if os.path.exists(out):  # a retry: reuse the decision only if it is exactly what the same panel gives now
    old = json.load(open(out))
    same = json.dumps(old, sort_keys=True) == json.dumps(rec, sort_keys=True)  # type-strict (1 != true)
    need(same and isinstance(old.get("stop"), bool), f"{out} exists and differs from the recomputed decision: reconcile by hand")
    print("A1_DECISION (existing, verified)", json.dumps(old)); sys.exit(0)
os.makedirs(G, exist_ok=True)
with open(out, "x") as fh:
    json.dump(rec, fh, indent=1)
print("A1_DECISION", json.dumps(rec))
