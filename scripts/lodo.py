"""Leave-one-drive-out sensitivity of a signed comparison (AGREED reporting item; Fable / Astra reviews).
For each dev drive d: drop d from BOTH metrics files (temp copies), rebuild the comparator's k=3 count list, rerun the SIGNED
temporal_compare (floor 0, cold_ref B), and report ID / W relative change, car-F1 diff and late gap. Read-only on inputs.

  python scripts/lodo.py runs/troll/m_V8k_dev.json runs/troll/m_A20k_indep_dev.json --out runs/s_lock/lodo_V8k_vs_A20k.json
"""
import argparse, json, os, subprocess, tempfile
p = argparse.ArgumentParser(); p.add_argument("cand"); p.add_argument("comp"); p.add_argument("--out", required=True); a = p.parse_args()
PY = ".venv/bin/python"
C, B = json.load(open(a.cand)), json.load(open(a.comp))
drives = sorted({r.rsplit("_s", 1)[0] for r in B["drives"]})
tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp")); rows = {}


def compare(cf, bf):
    cl = os.path.join(tmp, "cl.json")
    subprocess.run([PY, "scripts/temporal_compare.py", "--make_count_list", bf, "--k", "3", "--out", cl], check=True, capture_output=True)
    out = subprocess.run([PY, "scripts/temporal_compare.py", "--compare", cf, bf, "--count_list", cl, "--floor", "0", "--cold_ref", "B"],
                         capture_output=True, text=True).stdout
    d = json.loads(out.split("TCOMPARE ", 1)[1])
    return {k: {"A": d[k]["A"], "B": d[k]["B"], "rel": d[k]["rel"], "diff": d[k]["diff"]} for k in ("ID", "W", "car_f1", "late_lpips_gap", "flips", "ghost_ge5")}


rows["all"] = compare(a.cand, a.comp)
for d in drives:
    cf, bf = os.path.join(tmp, "c.json"), os.path.join(tmp, "b.json")
    json.dump({**C, "drives": {r: v for r, v in C["drives"].items() if r.rsplit("_s", 1)[0] != d}}, open(cf, "w"))
    json.dump({**B, "drives": {r: v for r, v in B["drives"].items() if r.rsplit("_s", 1)[0] != d}}, open(bf, "w"))
    try:
        rows[f"without {d}"] = compare(cf, bf)
    except Exception as e:
        rows[f"without {d}"] = {"error": str(e)[:200]}
json.dump({"cand": a.cand, "comp": a.comp, "rows": rows}, open(a.out, "w"), indent=1)
for k, v in rows.items():
    if "error" in v:
        print(k, v["error"]); continue
    print(f"{k:22s} ID {v['ID']['rel']:+.3f}  W {v['W']['rel']:+.3f}  F1 {v['car_f1']['diff']:+.4f}  late {v['late_lpips_gap']['diff']:+.4f}  flips {v['flips']['rel']}")
