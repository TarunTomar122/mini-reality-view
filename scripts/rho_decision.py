"""Pre-registered rho choice (council/AGREED-causal-r1.md): compare `--off` renders of frozen A@20k at rho 0.5 and 0.8 with the
signed A@20k dev renders (signed temporal_compare, A-derived k=3 count list, floor 0). Choose 0.5 if both its ID and W are
<= 1.20 x A's; else 0.8 if it meets both; else NONE (back to council). A row only counts if: ID and W are both eligible
(can_pass), the metrics cover exactly the 6 dev drives x seeds 0/1, every render file is frozen A@20k with temporal blocks off at
that rho (full length, no swap / salt / reset), the metrics file covers exactly those 12 runs and is newer than every render.
The ratio comes from the signed tool's relative difference ("rel", 3 decimals: ratio = 1 + rel, exact to +-0.0005; its
4-decimal means are NOT used); a ratio within 0.0005 of 1.20 cannot be decided at that precision -> back to council. Writes the
decision JSON that train_causal.py requires (valid = a choice was made on admissible evidence).

  python scripts/rho_decision.py --out council/rho_decision.json
"""
import argparse, glob, hashlib, json, math, os, subprocess, sys
import torch
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); a = p.parse_args()
PY = ".venv/bin/python"; A = "runs/longA/ckpt_20000.pt"
DEVL = json.load(open("data/temporal_lists.json"))["dev"]
WANT = {f"{d}_s{s}" for d in DEVL for s in (0, 1)}
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
# renders must carry the reviewed rollout code's fingerprint (same recipe as causal_rollout.py ROLL_CODE) and the A@20k digest
ROLL_CODE = hashlib.sha256(b"".join(open(f, "rb").read() for f in ["scripts/causal_rollout.py", "mrv/causal_video.py", "mrv/sd.py"])).hexdigest()[:12]
A_SHA = hashlib.sha256(open(A, "rb").read()).hexdigest()[:16]
subprocess.run([PY, "scripts/temporal_compare.py", "--make_count_list", "runs/troll/m_A20k_indep_dev.json", "--k", "3",
                "--out", "runs/count/count_A20k_dev_k3.json"], check=True, capture_output=True)
rows = {}
for rho in ("0.5", "0.8"):
    roll = f"runs/troll/Aoff_rho{rho}_dev"; met = f"runs/troll/m_Aoff_rho{rho}_dev.json"
    if not os.path.exists(met):  # missing evidence: inadmissible row (the ordered rule then stops or never reaches it)
        rows[rho] = {"admissible": False, "meets": False, "missing_metrics": met, "metrics_file": met, "metrics_sha": None,
                     "eligible": {"ID": False, "W": False}, "panel_ok": False}
        continue
    files = {os.path.basename(f)[:-3]: f for f in glob.glob(f"{roll}/*.pt")}
    def ok_render(name, f):
        r_ = torch.load(f); sc_ = name.rsplit("_s", 1)[0]
        n_ = len(torch.load(f"data/enc_tpv10/{sc_}.pt")["ticks"])
        return (r_.get("rho") == float(rho) and r_.get("method") == "causal_off" and r_.get("snapshot") == A and len(r_["lat"]) == n_
                and r_.get("roll_code") == ROLL_CODE and r_.get("ckpt_sha") == A_SHA
                and r_.get("swap_from", -1) == -1 and r_.get("salt_from", -1) == -1 and r_.get("reset_at", -1) == -1
                and r_.get("start_tick", 0) == 0)
    ident = all(ok_render(k, f) for k, f in files.items())
    m = json.load(open(met))
    fresh = os.path.getmtime(met) > max(os.path.getmtime(f) for f in files.values()) if files else False
    panel_ok = set(files) == WANT and set(m["drives"]) == WANT and ident and fresh and m.get("roll", "").rstrip("/").endswith(roll)
    out = subprocess.run([PY, "scripts/temporal_compare.py", "--compare", met, "runs/troll/m_A20k_indep_dev.json",
                          "--count_list", "runs/count/count_A20k_dev_k3.json", "--floor", "0", "--cold_ref", "B"],
                         check=True, capture_output=True, text=True).stdout
    d = json.loads(out.split("TCOMPARE ", 1)[1])
    fin = lambda v: isinstance(v, (int, float)) and math.isfinite(v)
    rel_ok = all(fin(d[k].get("rel")) and fin(d[k].get("A")) and fin(d[k].get("B")) and d[k]["B"] > 0 for k in ("ID", "W"))
    idr = 1 + d["ID"]["rel"] if rel_ok else float("nan"); wr = 1 + d["W"]["rel"] if rel_ok else float("nan")
    elig = {"ID": bool(d["ID"].get("can_pass")), "W": bool(d["W"].get("can_pass"))}
    borderline = (not rel_ok) or abs(idr - 1.20) <= 0.0005 or abs(wr - 1.20) <= 0.0005
    rows[rho] = {"ID_ratio": idr, "W_ratio": wr, "ID": d["ID"], "W": d["W"], "car_f1": d["car_f1"], "eligible": elig,
                 "panel_ok": panel_ok, "render_files": len(files), "render_identity_ok": ident, "borderline": borderline,
                 "metrics_sha": sha(met), "compare_drives": d.get("drives"), "compare_runs": d.get("runs"),
                 "meets": bool(rel_ok and idr <= 1.20 and wr <= 1.20)}
    rows[rho]["admissible"] = bool(panel_ok and elig["ID"] and elig["W"] and not borderline and d.get("drives") == 6 and d.get("runs") == 12)
    rows[rho]["metrics_file"] = met
chosen, valid = None, False
for rho in ("0.5", "0.8"):  # ordered rule
    if not rows[rho]["admissible"]:
        break  # inadmissible evidence at an earlier step: the rule cannot be applied -> council
    if rows[rho]["meets"]:
        chosen, valid = float(rho), True
        break
for rho in ("0.5", "0.8"):
    rows[rho]["metrics_sha"] = rows[rho]["metrics_sha"] if rows[rho].get("metrics_sha") or not os.path.exists(rows[rho]["metrics_file"]) else sha(rows[rho]["metrics_file"])
json.dump({"rule": "first of (0.5, 0.8) with ID and W each <= 1.20 x signed A@20k on admissible evidence; else back to council",
           "comparator": "runs/troll/m_A20k_indep_dev.json", "comparator_sha": sha("runs/troll/m_A20k_indep_dev.json"),
           "rows": rows, "chosen": chosen, "valid": valid, "roll_code": ROLL_CODE, "a20k_sha": A_SHA}, open(a.out, "w"), indent=1)
print(json.dumps({k: {"ID_ratio": v.get("ID_ratio"), "W_ratio": v.get("W_ratio"), "meets": v["meets"],
                      "admissible": v["admissible"], "missing": v.get("missing_metrics")} for k, v in rows.items()}), "chosen", chosen, "valid", valid)
sys.exit(0 if valid else 2)
