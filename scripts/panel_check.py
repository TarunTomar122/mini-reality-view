"""Panel check before any temporal comparison (drift check 9, Astra): the metric files must contain EXACTLY the expected
drives of the split (data/temporal_lists.json) x seeds 0 and 1 - a drive missing from both sides of a comparison would
otherwise go unnoticed. Also prints the Slurm submit line of each rollout folder's job if given (provenance, Fable).

  python scripts/panel_check.py --split dev 'runs/troll/m_A4k_indep_dev.json' 'runs/troll/m_B4k_s0.3_dev.json'
"""
import argparse, glob, json, sys
p = argparse.ArgumentParser()
p.add_argument("--split", required=True, choices=["dev", "decision", "final"])
p.add_argument("--late_cold", default="", help="also require, for every drive with a contiguous 170-tick run, finite late LPIPS and late car-F1 gaps on BOTH seeds scored against this cold folder (drift check 10, Astra); prints the run ids")
p.add_argument("files", nargs="+")
a = p.parse_args()
want = {f"{s}_s{q}" for s in json.load(open("data/temporal_lists.json"))[a.split] for q in ("0", "1")}
bad = 0
for pat in a.files:
    runs = set()
    for f in sorted(glob.glob(pat)):
        runs |= set(json.load(open(f))["drives"])
    miss, extra = sorted(want - runs), sorted(runs - want)
    ok = not miss and not extra
    bad += not ok
    print(("OK  " if ok else "FAIL"), pat, len(runs), "runs", ("missing " + str(miss[:6]) if miss else ""), ("extra " + str(extra[:6]) if extra else ""))
    if a.late_cold:
        D = {}
        for f in sorted(glob.glob(pat)):
            j_ = json.load(open(f)); D.update(j_["drives"])
            if j_.get("cold") != a.late_cold:
                print("FAIL", f, "scored against cold", j_.get("cold"), "expected", a.late_cold); bad += 1
        fin = lambda x: isinstance(x, (int, float)) and x == x and abs(x) != float("inf")
        cont = sorted({r.rsplit("_s", 1)[0] for r, q in D.items() if q.get("contiguous_170")})
        ids = []
        for sc in cont:
            for q in ("0", "1"):
                r = f"{sc}_s{q}"; d = D.get(r, {})
                good = d.get("contiguous_170") and fin(d.get("late_lpips_gap")) and fin(d.get("late_f1_gap"))
                ids.append(f"{r}:{'ok' if good else 'MISSING'}"); bad += not good
        print("    late window:", len(cont), "drives;", " ".join(ids))
sys.exit(1 if bad else 0)
