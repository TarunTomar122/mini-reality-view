"""Copy warning (AGREED r1 gate reports): per-tick latent change of a rollout vs A@20k's, on the 6 dev drives x seeds 0/1.
Warning if the candidate's mean |z_t - z_{t-1}| is < 0.7 x A's (frozen output = copying the past) or falling along the drive
(first vs last third). Only consecutive usable ticks (ok, no hard break, tick + 1) count. Read-only.
  python scripts/latent_change.py runs/troll/V16k_dev runs/troll/A20k_indep_dev --out runs/s_lock/latchange_V16k.json
"""
import argparse, glob, json, os
import torch
p = argparse.ArgumentParser(); p.add_argument("cand"); p.add_argument("ref"); p.add_argument("--out", required=True); a = p.parse_args()


def change(f):
    r = torch.load(f, map_location="cpu", weights_only=False); z = r["lat"].float(); t = r["ticks"].tolist()
    ok = [bool(x) for x in r["ok"]]; hard = [bool(x) for x in r["hard"]]
    d = [(z[i] - z[i - 1]).abs().mean().item() for i in range(1, len(t)) if ok[i] and ok[i - 1] and not hard[i] and t[i] == t[i - 1] + 1]
    n = len(d) // 3
    return {"mean": sum(d) / max(len(d), 1), "first_third": sum(d[:n]) / max(n, 1), "last_third": sum(d[-n:]) / max(n, 1), "n": len(d)}


rows = {}
for f in sorted(glob.glob(f"{a.cand}/scene-*_s*.pt")):
    b = os.path.basename(f); g = f"{a.ref}/{b}"
    if os.path.exists(g):
        c, r = change(f), change(g); rows[b[:-3]] = {"cand": c, "ref": r, "ratio": c["mean"] / r["mean"]}
ratio = sum(v["ratio"] for v in rows.values()) / len(rows)
fall = sum(v["cand"]["last_third"] < v["cand"]["first_third"] for v in rows.values())
fall_ref = sum(v["ref"]["last_third"] < v["ref"]["first_third"] for v in rows.values())
out = {"cand": a.cand, "ref": a.ref, "runs": len(rows), "mean_ratio": ratio, "runs_falling": fall, "ref_runs_falling": fall_ref,
       "warning_low": ratio < 0.7, "rows": rows}
json.dump(out, open(a.out, "w"), indent=1)
print(f"LATCHANGE runs {len(rows)} mean ratio vs ref {ratio:.3f} (warn < 0.7: {ratio < 0.7}); runs with falling change {fall}/{len(rows)} (ref {fall_ref})")
