"""S1 update forecast (Astra S1b item 5), run by slurm/wan_teacher.sh MODE=timing AFTER the 81-tick rollout and its RGB scoring.
Reads <timing_dir>/timing.json (trainer timing: production-loop wall per update incl. prefetch queue wait, startup, probe, save,
30-step window) + the rollout's own breakdown + the measured rollout / scorer wall times, the S1 ledger (all jobs so far) and
the dev tick count, and deducts from the budget: ledger hours, the pending overfit gate (2,000 updates + recurring fixed-noise
evals + its THREE sampled modes / decodes / scoring) and resume test (16 + 8 + 8 updates, 3 starts) unless the gate registry holds
a PASS for them (a failed gate is not finished), start-up scaled from the timing drive list to the production one, --n_panels full dev panels at the measured per-tick
rollout / scoring cost, numbered saves, recurring probes / saves per update and one trainer start per --job_h job
(mrv.wan_plan.forecast_s1). The scorer's model loading is counted per tick (conservative: no separate start measured).
Writes timing.json["forecast_final"] and prints WAN_FORECAST.
  python scripts/wan_forecast.py --timing_dir runs/wan_timing_<job> --rollout_s 412 --score_s 388
"""
import argparse, json
import torch
from mrv import wan_plan as WP

p = argparse.ArgumentParser()
p.add_argument("--timing_dir", required=True); p.add_argument("--rollout_s", type=float, required=True); p.add_argument("--score_s", type=float, required=True)
p.add_argument("--ticks", type=int, default=81, help="ticks in the timing rollout (x 1 seed)")
p.add_argument("--ledger", default="runs/wan_s1_ledger.jsonl"); p.add_argument("--lists", default="data/temporal_lists.json"); p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--budget_hours", type=float, default=24.0); p.add_argument("--n_panels", type=int, default=2)
p.add_argument("--job_h", type=float, default=12.0); p.add_argument("--val_every", type=int, default=500); p.add_argument("--save_every", type=int, default=500)
p.add_argument("--n_numbered", type=int, default=15)
p.add_argument("--gates", default="runs/wan_gates/gates.jsonl", help="gate registry: a gate counts as done only with a PASS record")
a = p.parse_args()
J = json.load(open(f"{a.timing_dir}/timing.json"))
br = J["rollout_breakdown"]
inner = br["encode_s"] + br["sample_s"] + br["decode_s"] + br["sd_s"]
roll_tick, roll_start = inner / a.ticks, max(0.0, a.rollout_s - inner)
score_tick = a.score_s / a.ticks
dev = json.load(open(a.lists))["dev"]
dev_ticks = sum(len(torch.load(f"{a.enc}/{d}.pt", map_location="cpu", weights_only=False, mmap=True)["ticks"]) for d in dev)
panel_base_h = WP.panel_cost_h(2 * dev_ticks, roll_tick, roll_start, score_tick, 0.0)
# shared-mode donors need one extra unswapped same-length rollout each (Astra S1c item 4): price them from the real donor plan
import os
_drv = {}
for d in dev:
    E_ = torch.load(f"{a.enc}/{d}.pt", map_location="cpu", weights_only=False, mmap=True)
    _drv[d] = {"ticks": [int(t) for t in E_["ticks"].tolist()], "ok": [bool(x) for x in E_["ok"].tolist()], "hard": [bool(x) for x in E_["hard"].tolist()]}
PLAN = WP.plan_donors(_drv, 5, 50, lambda sc, t: os.path.exists(f"data/sim/keys_tpv10/{sc}/t{int(t):04d}.jpg"))
# a total is only defined for a COMPLETE plan (six valid assignments); otherwise it is None and the trainer refuses (S1d item 4)
panel_h = WP.panel_with_shared_h(panel_base_h, PLAN["donors"], roll_tick, roll_start) if (PLAN["ok"] and len(PLAN["donors"]) == 6) else None
L = WP.Ledger(a.ledger, "forecast", write=False)
kinds = L.by_kind()
# overfit verdict (Astra / Neon 08:30): 8 clips sampled in THREE modes (hybrid, tokens-only, none; 30 steps, batch-1 window cost
# as a conservative bound), four decodes per clip (3 samples + round trip; mid-stream clips decode with context: x2), and ~20
# frames x (4 renders + 1 key) segmented per clip at the measured per-tick scorer cost
verdict_h = (3 * 8 * J["window30_rows1_s"] + 2 * 4 * 8 * J["wan_decode_5lat_s"] + 8 * 20 * 5 * score_tick) / 3600
# recurring fixed-noise overfit evals: every 100 updates + start + end, 3 sigmas x one forward of the 8 clips
fwd8 = J["micro"][str(J["best_micro"])]["fwd_s"] if str(J["best_micro"]) in J["micro"] else J["micro"][J["best_micro"]]["fwd_s"]
evals_h = (2000 // 100 + 2) * 3 * fwd8 / 3600
# start-up measured on the timing run's drive list, scaled to the production drive count (hashing / loading scale with drives)
prod_train = len(json.load(open(a.lists))["train"])
startup = J["startup_s"] * max(1.0, prod_train / max(1, J.get("train_drives", prod_train)))
# a gate is finished only with a PASS record (a failed or pending allocation will run again)
recs = WP.gate_records(a.gates)
GK = J.get("gate_key")  # the production gate key (timing runs the production model recipe and code): Astra S1c item 5
done = {k for k in ("overfit", "resume_test") if GK and any(r.get("kind") == k and r.get("status") == "PASS" and r.get("gate_key") == GK for r in recs)}
t = {"budget_h": a.budget_hours, "used_h": L.hours(live=False), "done_kinds": done, "wall_s": J["production_loop"]["wall_s"],
     "startup_s": startup, "probe_s": J["probe_s"], "val_every": a.val_every, "save_s": J["save_s"], "save_every": a.save_every,
     "n_numbered": a.n_numbered, "job_h": a.job_h, "panel_h": panel_h if panel_h is not None else panel_base_h, "n_panels": a.n_panels, "overfit_updates": 2000,
     "resume_updates": 32, "overfit_verdict_h": verdict_h, "overfit_evals_h": evals_h}
fc = WP.forecast_s1(t)
fc.update(inputs={k: (sorted(v) if isinstance(v, set) else v) for k, v in t.items()}, dev_ticks=dev_ticks, panel_h=round(panel_h, 3) if panel_h is not None else None,
          panel_plan_complete=panel_h is not None,
          gate_key=GK, panel_base_h=round(panel_base_h, 3), panel_h_total=round(panel_h, 3) if panel_h is not None else None,
          donor_plan=PLAN["donors"], donor_plan_ok=bool(PLAN["ok"] and len(PLAN["donors"]) == 6),
          startup_measured_s=J["startup_s"], startup_scaled_s=round(startup, 1), timing_train_drives=J.get("train_drives"), prod_train_drives=prod_train,
          roll_tick_s=round(roll_tick, 3), roll_start_s=round(roll_start, 1), score_tick_s=round(score_tick, 3), ledger_by_kind=kinds)
J["forecast_final"] = fc
json.dump(J, open(f"{a.timing_dir}/timing.json", "w"), indent=1)
print("WAN_FORECAST", json.dumps(fc))
