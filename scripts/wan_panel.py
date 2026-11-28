"""PLAN-video S1 dev panel (Astra items 9, 10, 12): one checkpoint -> full six-drive x two-seed panel through the SIGNED comparator.

1. RGB-path parity (once per run folder): signed A@20k rollouts replayed through scripts/score_rgb.py --from_lat must reproduce
   the signed metrics field by field (--parity_ref); otherwise the panel stops (the RGB entry is not trusted).
2. For each conditioning mode (hybrid, guide, tokens, none; same seeds = FIXED noise and window borders): scripts/wan_rollout.py
   -> scripts/score_rgb.py -> signed scripts/temporal_compare.py exactly as specified: candidate FIRST, A@20k SECOND
   (runs/troll/m_A20k_indep_dev.json), A's k=3 count list (runs/count/count_A20k_dev_k3.json), --floor 0 --cold_ref B; the
   panel must be complete (6 drives, 12 runs, nothing unpaired).
3. Camera swap (hybrid, --swap; window borders and noise as the unswapped run): scored three ways with the same RGB: full run vs
   the donor keys with the lag test against the DONOR's signed A@20k cold reference (decoded with the real SD VAE inside
   score_rgb), post-swap ticks vs the donor keys, post-swap ticks vs the receiver keys -> mrv.wan_plan.donor_following.
4. Decision: mrv.wan_plan.decide_s1 on the hybrid compare (car F1 >= 0.4456, road IoU >= A, eligible ID and W each down > 9.4 %,
   donor-following swap); below 12k updates = unresolved; --stage 24h: pass / one 12 h extension only for car F1 in
   [0.4156, 0.4456) with every other gate passed / stop; --stage final: bar 0.4456. The four modes are reported side by side.
The panel debits its wall time to the S1 ledger (rollouts run with --ledger ''). Dev only (sealed splits never touched).
Writes <wroot>/<name>_panel.json.
  python scripts/wan_panel.py --ckpt runs/wan_s1/ckpt_12000.pt --name s1_12k --stage check
"""
import argparse, json, os, statistics, subprocess, sys, time
from mrv import wan_plan as WP

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True); p.add_argument("--name", required=True)
p.add_argument("--stage", default="check", choices=["check", "24h", "final"])
p.add_argument("--conds", default="hybrid,guide,tokens,none")
p.add_argument("--wroot", default="runs/wroll")
p.add_argument("--a_metrics", default="runs/troll/m_A20k_indep_dev.json")
p.add_argument("--a_roll", default="runs/troll/A20k_indep_dev")
p.add_argument("--count_list", default="runs/count/count_A20k_dev_k3.json")
p.add_argument("--ledger", default="runs/wan_s1_ledger.jsonl")
p.add_argument("--swap_at", type=int, default=50)
p.add_argument("--no_swap", action="store_true")
p.add_argument("--rollout_args", default="", help="extra flags for wan_rollout.py, e.g. '--steps 30 --cfg 1.0'")
a = p.parse_args()
PY = sys.executable
if "hybrid" not in a.conds.split(","):
    raise SystemExit("the decision is read on the hybrid mode: --conds must include hybrid")
LEDGER = WP.Ledger(a.ledger, "panel").start()
os.makedirs(a.wroot, exist_ok=True)
T0 = time.time(); timing = {}


def run(cmd, what):
    t = time.time(); print("RUN", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    timing[what] = round(time.time() - t, 1)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        LEDGER.stop("failed"); raise SystemExit(f"PANEL_FAIL {what} (exit {r.returncode})")
    return r.stdout


def tcompare(m):
    out = run([PY, "scripts/temporal_compare.py", "--compare", m, a.a_metrics, "--count_list", a.count_list, "--floor", "0", "--cold_ref", "B"], f"compare {m}")
    d = json.loads(out.split("TCOMPARE ", 1)[1].splitlines()[0])
    if not (d["drives"] == 6 and d["runs"] == 12 and not d["unpaired_drives"]):
        LEDGER.stop("incomplete"); raise SystemExit(f"PANEL_FAIL incomplete panel for {m}: {d['drives']} drives, {d['runs']} runs, unpaired {d['unpaired_drives']}")
    if d.get("cold_ref") != "B":
        LEDGER.stop("cold_ref"); raise SystemExit(f"PANEL_FAIL compare for {m} did not run with --cold_ref B")
    return d


# 1. RGB-path parity of the signed A@20k replay
par = f"{a.wroot}/a20k_rgbpath_parity.json"
if not (os.path.exists(par) and json.load(open(par)).get("pass")):
    out = run([PY, "scripts/score_rgb.py", "--from_lat", a.a_roll, "--parity_ref", a.a_metrics, "--out", f"{a.wroot}/m_A20k_rgbpath.json"], "a20k_parity")
    json.dump({"pass": "RGB_PATH_PARITY PASS" in out, "log": out[-500:]}, open(par, "w"), indent=1)
    if "RGB_PATH_PARITY PASS" not in out:
        LEDGER.stop("parity"); raise SystemExit("PANEL_FAIL A@20k replay through the RGB path differs from the signed metrics")
roll_extra = a.rollout_args.split() if a.rollout_args else []
res = {"ckpt": a.ckpt, "name": a.name, "stage": a.stage, "modes": {}}
import torch
step = int(torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)["step"])
res["step"] = step

# 2. conditioning modes at fixed noise
for cond in a.conds.split(","):
    rd, m = f"{a.wroot}/{a.name}_{cond}_dev", f"{a.wroot}/m_{a.name}_{cond}_dev.json"
    run([PY, "scripts/wan_rollout.py", "--ckpt", a.ckpt, "--split", "dev", "--seeds", "0,1", "--cond", cond, "--out", rd, "--ledger", ""] + roll_extra, f"rollout {cond}")
    run([PY, "scripts/score_rgb.py", "--rgb", rd, "--out", m], f"score {cond}")
    d = tcompare(m)
    res["modes"][cond] = {"metrics": m, "car_f1": d["car_f1"], "road_iou": d["road_iou"], "ID": d["ID"], "W": d["W"],
                          "ghost_ge5": d.get("ghost_ge5"), "recall": d.get("recall"), "flips": d.get("flips")}
    if cond == "hybrid":
        CMP = d

# 3. camera swap (hybrid)
swap = None
if not a.no_swap:
    rd = f"{a.wroot}/{a.name}_swap_dev"
    run([PY, "scripts/wan_rollout.py", "--ckpt", a.ckpt, "--split", "dev", "--seeds", "0,1", "--cond", "hybrid", "--swap", "--swap_at", str(a.swap_at),
         "--out", rd, "--ledger", ""] + roll_extra, "rollout swap")
    mf, md, mr = (f"{a.wroot}/m_{a.name}_swap_{k}.json" for k in ("full", "postdonor", "postrecv"))
    run([PY, "scripts/score_rgb.py", "--rgb", rd, "--lag_ref", a.a_roll, "--out", mf], "score swap full")
    run([PY, "scripts/score_rgb.py", "--rgb", rd, "--post_swap_only", "--out", md], "score swap donor")
    run([PY, "scripts/score_rgb.py", "--rgb", rd, "--post_swap_only", "--as_receiver", "--out", mr], "score swap receiver")
    full, dn, rc = (json.load(open(f))["drives"] for f in (mf, md, mr))
    by = lambda D: {k: statistics.mean(v) for k, v in _group(D).items()}

    def _group(D):
        g = {}
        for run_, v in D.items():
            g.setdefault(run_.rsplit("_s", 1)[0], []).append(v["car_f1"])
        return g
    # donor-keyed files are named by the donor; pair them back to their receivers through the full-run "other" fields
    pair = {}
    for run_ in full:
        R = torch.load(f"{rd}/{run_}.pt", map_location="cpu", weights_only=False, mmap=True)
        pair[R["other"] + "_s" + run_.rsplit("_s", 1)[1]] = run_
    dn_recv = {pair[k]: v for k, v in dn.items()}
    lag = [(v["lag_start"] if v.get("lag_recovered") else float("inf")) for v in full.values() if "lag_recovered" in v]
    swap = WP.donor_following(by(dn_recv), by(rc), lag)
    swap["boundaries_matched"] = all(bool(torch.load(f"{rd}/{r}.pt", map_location="cpu", weights_only=False, mmap=True).get("swap_boundaries_matched")) for r in full)
    swap["lag_excluded_runs"] = sum(1 for v in full.values() if v.get("lag_excluded"))
    res["swap"] = swap

# 4. decision
res["decision"] = WP.decide_s1(CMP, step, a.stage, swap["ok"] if swap else None)
res["timing_s"] = timing; res["panel_hours"] = round((time.time() - T0) / 3600, 3)
res["ledger_hours"] = round(LEDGER.hours(), 3); res["ledger_by_kind"] = LEDGER.by_kind()
json.dump(res, open(f"{a.wroot}/{a.name}_panel.json", "w"), indent=1, default=str)
print("WAN_PANEL", json.dumps({"step": step, "stage": a.stage, "verdict": res["decision"]["verdict"], "gates": res["decision"]["gates"],
                               "modes": {k: v["car_f1"].get("A") for k, v in res["modes"].items()}}, default=str), flush=True)
LEDGER.stop()
