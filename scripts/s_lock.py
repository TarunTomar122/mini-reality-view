"""Mechanical s-lock for arm B / C (council/temporal-s-lock-rule.md, drift checks 6-10; committed BEFORE any sweep score is
read). Input: one temporal_compare.py --compare output per s (candidate = arm at s, comparator = A 4k independent dev,
--count_list council/count_A4k_dev_k3.json --floor 0 --cold_ref B). Each clause reads exactly the field named here:

  eligibility (all must hold, point estimates on dev):
    car_f1.diff        >= -0.02      road_iou.diff   >= -0.01      recall.diff >= -0.02
    late_lpips_gap.A   <= +0.02      late_f1_gap.A   >= -0.03      (A = candidate's own drift vs its own cold rollout)
    ghost_growth.diff  <= 0          ghost_ge5.diff  <= +1.0
    ID.B > 0 and W.B > 0 (positive reference errors)
    ID.can_pass and W.can_pass (no counting dev drive with eligible but no shared transitions)
  rank: score = max(ID.A / ID.B, W.A / W.B); lowest wins; scores within 0.01 of the best -> the LARGEST s.
  no eligible s -> s = 0.3, labelled "failed dev screen" (the clause(s) that failed are printed for every s).

  python scripts/s_lock.py --arm B runs/s_lock/B_s0.json runs/s_lock/B_s0.1.json runs/s_lock/B_s0.3.json runs/s_lock/B_s0.5.json
"""
import argparse, json, re
p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
p.add_argument("files", nargs=4)
a = p.parse_args()
rows = []
for f in a.files:
    t = open(f).read(); d = json.loads(t[t.index("{"):])
    s = float(re.search(r"_s([0-9.]+)\.json$", f).group(1))
    fails = []
    chk = lambda ok, name: fails.append(name) if not ok else None
    chk(d["car_f1"]["diff"] >= -0.02, "car_f1.diff")
    chk(d["road_iou"]["diff"] >= -0.01, "road_iou.diff")
    chk(d["recall"]["diff"] >= -0.02, "recall.diff")
    chk(d["late_lpips_gap"]["A"] <= 0.02, "late_lpips_gap.A")
    chk(d["late_f1_gap"]["A"] >= -0.03, "late_f1_gap.A")
    chk(d["ghost_growth"]["diff"] <= 0, "ghost_growth.diff")
    chk(d["ghost_ge5"]["diff"] <= 1.0, "ghost_ge5.diff")
    chk(d["ID"]["B"] > 0 and d["W"]["B"] > 0, "reference errors")
    chk(bool(d["ID"]["can_pass"]) and bool(d["W"]["can_pass"]), "ID/W can_pass")
    score = max(d["ID"]["A"] / d["ID"]["B"], d["W"]["A"] / d["W"]["B"])
    rows.append({"s": s, "eligible": not fails, "fails": fails, "score": round(score, 4),
                 "ID": [d["ID"]["A"], d["ID"]["B"]], "W": [d["W"]["A"], d["W"]["B"]], "car_f1_diff": d["car_f1"]["diff"],
                 "road_diff": d["road_iou"]["diff"], "recall_diff": d["recall"]["diff"], "late_lpips": d["late_lpips_gap"]["A"],
                 "late_f1": d["late_f1_gap"]["A"], "ghost_growth_diff": d["ghost_growth"]["diff"], "ghost_ge5_diff": d["ghost_ge5"]["diff"]})
el = [r for r in rows if r["eligible"]]
if el:
    best = min(r["score"] for r in el); lock = max(r["s"] for r in el if r["score"] <= best + 0.01); label = "locked"
else:
    lock, label = 0.3, "failed dev screen"
out = {"arm": a.arm, "lock_s": lock, "label": label, "table": rows}
print(json.dumps(out, indent=1))
