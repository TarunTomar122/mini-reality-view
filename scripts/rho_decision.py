"""Pre-registered rho choice (council/AGREED-causal-r1.md): compare `--off` renders of frozen A@20k at rho 0.5 and 0.8 with the
signed A@20k dev renders (signed temporal_compare, A-derived k=3 count list, floor 0). Choose 0.5 if both its ID and W are
<= 1.20 x A's; else 0.8 if it meets both; else NONE (back to council). A row only counts if: ID and W are both eligible
(can_pass), the metrics cover exactly the 6 dev drives x seeds 0/1, every render file is frozen A@20k with temporal blocks off at
that rho, and the ratio is not within 0.002 of the 1.20 line (temporal_compare rounds means to 4 decimals). Writes the
decision JSON that train_causal.py requires (valid = a choice was made on admissible evidence).

  python scripts/rho_decision.py --out council/rho_decision.json
"""
import argparse, glob, hashlib, json, os, subprocess, sys
import torch
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); a = p.parse_args()
PY = ".venv/bin/python"; A = "runs/longA/ckpt_20000.pt"
DEVL = json.load(open("data/temporal_lists.json"))["dev"]
WANT = {f"{d}_s{s}" for d in DEVL for s in (0, 1)}
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
subprocess.run([PY, "scripts/temporal_compare.py", "--make_count_list", "runs/troll/m_A20k_indep_dev.json", "--k", "3",
                "--out", "runs/count/count_A20k_dev_k3.json"], check=True, capture_output=True)
rows = {}
for rho in ("0.5", "0.8"):
    roll = f"runs/troll/Aoff_rho{rho}_dev"; met = f"runs/troll/m_Aoff_rho{rho}_dev.json"
    files = {os.path.basename(f)[:-3]: f for f in glob.glob(f"{roll}/*.pt")}
    ident = all((lambda r_: r_.get("rho") == float(rho) and r_.get("method") == "causal_off" and r_.get("snapshot") == A)(torch.load(f))
                for f in files.values())
    m = json.load(open(met))
    panel_ok = set(files) == WANT and ident and m.get("roll", "").rstrip("/").endswith(roll)
    out = subprocess.run([PY, "scripts/temporal_compare.py", "--compare", met, "runs/troll/m_A20k_indep_dev.json",
                          "--count_list", "runs/count/count_A20k_dev_k3.json", "--floor", "0", "--cold_ref", "B"],
                         check=True, capture_output=True, text=True).stdout
    d = json.loads(out.split("TCOMPARE ", 1)[1])
    idr, wr = d["ID"]["A"] / d["ID"]["B"], d["W"]["A"] / d["W"]["B"]
    elig = {"ID": bool(d["ID"].get("can_pass")), "W": bool(d["W"].get("can_pass"))}
    borderline = abs(idr - 1.20) < 0.002 or abs(wr - 1.20) < 0.002
    rows[rho] = {"ID_ratio": idr, "W_ratio": wr, "ID": d["ID"], "W": d["W"], "car_f1": d["car_f1"], "eligible": elig,
                 "panel_ok": panel_ok, "render_files": len(files), "render_identity_ok": ident, "borderline": borderline,
                 "metrics_sha": sha(met), "compare_drives": d.get("drives"), "compare_runs": d.get("runs"),
                 "meets": idr <= 1.20 and wr <= 1.20}
    rows[rho]["admissible"] = panel_ok and elig["ID"] and elig["W"] and not borderline
chosen, valid = None, False
for rho in ("0.5", "0.8"):  # ordered rule
    if not rows[rho]["admissible"]:
        break  # inadmissible evidence at an earlier step: the rule cannot be applied -> council
    if rows[rho]["meets"]:
        chosen, valid = float(rho), True
        break
json.dump({"rule": "first of (0.5, 0.8) with ID and W each <= 1.20 x signed A@20k on admissible evidence; else back to council",
           "comparator": "runs/troll/m_A20k_indep_dev.json", "comparator_sha": sha("runs/troll/m_A20k_indep_dev.json"),
           "rows": rows, "chosen": chosen, "valid": valid}, open(a.out, "w"), indent=1)
print(json.dumps({k: {"ID_ratio": round(v["ID_ratio"], 4), "W_ratio": round(v["W_ratio"], 4), "meets": v["meets"],
                      "admissible": v["admissible"]} for k, v in rows.items()}), "chosen", chosen, "valid", valid)
sys.exit(0 if valid else 2)
