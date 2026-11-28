"""S1 update forecast (Astra S1b item 5), run by slurm/wan_teacher.sh MODE=timing AFTER the 81-tick rollout and its RGB scoring.
Reads <timing_dir>/timing.json (trainer timing: production-loop wall per update incl. prefetch queue wait, startup, probe, save,
30-step window) + the rollout's own breakdown + the measured rollout / scorer wall times, the S1 ledger (all jobs so far) and
the dev tick count, and deducts from the budget: ledger hours, the pending overfit gate (2,000 updates + its decode / scoring)
and resume test (16 + 8 + 8 updates, 3 starts) unless already charged, --n_panels full dev panels at the measured per-tick
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
a = p.parse_args()
J = json.load(open(f"{a.timing_dir}/timing.json"))
br = J["rollout_breakdown"]
inner = br["encode_s"] + br["sample_s"] + br["decode_s"] + br["sd_s"]
roll_tick, roll_start = inner / a.ticks, max(0.0, a.rollout_s - inner)
score_tick = a.score_s / a.ticks
dev = json.load(open(a.lists))["dev"]
dev_ticks = sum(len(torch.load(f"{a.enc}/{d}.pt", map_location="cpu", weights_only=False, mmap=True)["ticks"]) for d in dev)
panel_h = WP.panel_cost_h(2 * dev_ticks, roll_tick, roll_start, score_tick, 0.0)
L = WP.Ledger(a.ledger, "forecast", write=False)
kinds = L.by_kind()
verdict_h = (8 * J["window30_rows1_s"] + 8 * 20 * 2 * score_tick) / 3600  # 8 clips sampled + 2 x ~20 frames segmented per clip
done = {k.replace("alloc_", "") for k in kinds}  # allocation beats are labelled alloc_<mode>
t = {"budget_h": a.budget_hours, "used_h": L.hours(live=False), "done_kinds": done, "wall_s": J["production_loop"]["wall_s"],
     "startup_s": J["startup_s"], "probe_s": J["probe_s"], "val_every": a.val_every, "save_s": J["save_s"], "save_every": a.save_every,
     "n_numbered": a.n_numbered, "job_h": a.job_h, "panel_h": panel_h, "n_panels": a.n_panels, "overfit_updates": 2000,
     "resume_updates": 32, "overfit_verdict_h": verdict_h}
fc = WP.forecast_s1(t)
fc.update(inputs={k: (sorted(v) if isinstance(v, set) else v) for k, v in t.items()}, dev_ticks=dev_ticks, panel_h=round(panel_h, 3),
          roll_tick_s=round(roll_tick, 3), roll_start_s=round(roll_start, 1), score_tick_s=round(score_tick, 3), ledger_by_kind=kinds)
J["forecast_final"] = fc
json.dump(J, open(f"{a.timing_dir}/timing.json", "w"), indent=1)
print("WAN_FORECAST", json.dumps(fc))
