"""Pre-registered rho choice (council/AGREED-causal-r1.md): compare `--off` renders of frozen A@20k at rho 0.5 and 0.8 with the
signed A@20k dev renders (temporal_compare, A-derived k=3 count list, floor 0). Choose 0.5 if both its ID and W are <= 1.20 x
A's; else 0.8 if it meets both; else NONE (back to council). Writes the decision JSON that train_causal.py requires.

  python scripts/rho_decision.py --out council/rho_decision.json
"""
import argparse, json, subprocess, sys
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); a = p.parse_args()
PY = ".venv/bin/python"; rows = {}
subprocess.run([PY, "scripts/temporal_compare.py", "--make_count_list", "runs/troll/m_A20k_indep_dev.json", "--k", "3",
                "--out", "runs/count/count_A20k_dev_k3.json"], check=True, capture_output=True)
for rho in ("0.5", "0.8"):
    out = subprocess.run([PY, "scripts/temporal_compare.py", "--compare", f"runs/troll/m_Aoff_rho{rho}_dev.json", "runs/troll/m_A20k_indep_dev.json",
                          "--count_list", "runs/count/count_A20k_dev_k3.json", "--floor", "0", "--cold_ref", "B"], check=True, capture_output=True, text=True).stdout
    d = json.loads(out.split("TCOMPARE ", 1)[1])
    rows[rho] = {"ID_ratio": d["ID"]["A"] / d["ID"]["B"], "W_ratio": d["W"]["A"] / d["W"]["B"], "ID": d["ID"], "W": d["W"],
                 "car_f1": d["car_f1"], "eligible": {"ID": d["ID"].get("can_pass"), "W": d["W"].get("can_pass")}}
    rows[rho]["meets"] = rows[rho]["ID_ratio"] <= 1.20 and rows[rho]["W_ratio"] <= 1.20
chosen = 0.5 if rows["0.5"]["meets"] else (0.8 if rows["0.8"]["meets"] else None)
json.dump({"rule": "first of (0.5, 0.8) with ID and W each <= 1.20 x signed A@20k; else back to council", "rows": rows, "chosen": chosen}, open(a.out, "w"), indent=1)
print(json.dumps({k: {"ID_ratio": round(v["ID_ratio"], 3), "W_ratio": round(v["W_ratio"], 3), "meets": v["meets"]} for k, v in rows.items()}), "chosen", chosen)
sys.exit(0 if chosen is not None else 2)
