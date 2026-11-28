"""S1 update forecast + re-time rule (Astra S1b item 5; council/s1-timing-r1-CONVERGED.md items 1, 4, 7), run by
slurm/wan_teacher.sh MODE=timing after the trainer timing and the panel-loop rehearsal. CPU only after the width guard; can be
re-run at any time (e.g. once the 2080 Ti scorer parity has landed) to refresh retime_pass:
  python scripts/wan_forecast.py --timing_dir runs/wan_timing_<job>
Reads <timing_dir>/timing.json (trainer: production-loop wall per update = slowest of the full-run mean and the last two 32-update
block means, start-up measured from the ALLOCATION to the first update incl. the stage-in, probe, save, sweep / matched-update /
staged-vs-Lustre / memory records) + <timing_dir>/rehearsal/rehearsal.json (load-once rollout rate, one model load, byte-equality),
the S1 ledger (everything charged so far, both timings included), the gate registry and the dev tick count, and deducts from
the budget: ledger hours, the pending overfit gate (2,000 updates + recurring fixed-noise evals + its three sampled modes /
decodes / scoring) and resume test (3 starts + 32 updates) unless the registry holds a PASS for this gate key, --n_panels
(default 1, the gate; CONVERGED item 1) SLICE-ONLY panels (rendering with Wan loaded once, shared-mode references priced; the
signed scoring runs on 2080 Ti jobs under its own ledger and cap), numbered saves, recurring probes / saves and one start (from
allocation, incl. staging) per --job_h job (mrv.wan_plan.forecast_s1). The two-panel number is printed and recorded beside it
("no spare for a second read" if it is < 12,000). Start-up is never extrapolated 48 -> 144 when the timing ran on the
production list. Writes timing.json["forecast_final"] and ["retime_pass"] (every rule with pass / fail and its numbers; the
trainer refuses overfit / resume test / train unless all pass) and prints WAN_FORECAST / RETIME_PASS.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrv import wan_plan as WP

p = argparse.ArgumentParser()
p.add_argument("--timing_dir", required=True)
p.add_argument("--rehearsal", default="", help="default: <timing_dir>/rehearsal/rehearsal.json")
p.add_argument("--ledger", default="runs/wan_s1_ledger.jsonl"); p.add_argument("--lists", default="data/temporal_lists.json"); p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--budget_hours", type=float, default=24.0)
p.add_argument("--n_panels", type=int, default=1, help="panels charged inside the budget (CONVERGED item 1: ONE; the second belongs to the +12 h extension)")
p.add_argument("--job_h", type=float, default=12.0); p.add_argument("--val_every", type=int, default=500); p.add_argument("--save_every", type=int, default=500)
p.add_argument("--n_numbered", type=int, default=15)
p.add_argument("--gates", default="runs/wan_gates/gates.jsonl", help="gate registry: a gate counts as done only with a PASS record")
p.add_argument("--window_latents", type=int, default=5, help="rollout window size the S1 panels will use (prices the donor plan)")
p.add_argument("--swap_at", type=int, default=50, help="swap position the S1 panels will use")
p.add_argument("--score_parity", default="runs/wroll/score_parity_2080.json")
p.add_argument("--a_metrics", default="runs/troll/m_A20k_indep_dev.json"); p.add_argument("--a_roll", default="runs/troll/A20k_indep_dev")
p.add_argument("--overfit_score_tick_s", type=float, default=2.41,
               help="slice signed-scorer seconds / tick for the overfit verdict's segmentation (measured in timing 3756964)")
a = p.parse_args()
J = json.load(open(f"{a.timing_dir}/timing.json"))
RHP = a.rehearsal or f"{a.timing_dir}/rehearsal/rehearsal.json"
RH = json.load(open(RHP)) if os.path.exists(RHP) else None
if RH is None:
    raise SystemExit(f"no panel-loop rehearsal record {RHP}: the slice panel cost (load-once rate) is unmeasured; refused")
if RH.get("window_latents") != a.window_latents:  # Astra S1f 4b: rates measured at one width never price another
    raise SystemExit(f"rehearsal rollout measured at window {RH.get('window_latents')}, forecast asked for {a.window_latents}: refused "
                     f"(re-run MODE=timing with PANEL_WIN={a.window_latents})")
import torch  # after the width guard (CPU tests run the guard without torch)
roll_tick, load_s = float(RH["roll_tick_s"]), float(RH["load_s"])
L_ = json.load(open(a.lists))
dev = L_["dev"]
_drv = {}
for d in dev:
    E_ = torch.load(f"{a.enc}/{d}.pt", map_location="cpu", weights_only=False, mmap=True)
    _drv[d] = {"ticks": [int(t) for t in E_["ticks"].tolist()], "ok": [bool(x) for x in E_["ok"].tolist()], "hard": [bool(x) for x in E_["hard"].tolist()]}
dev_ticks = sum(len(v["ticks"]) for v in _drv.values())
# priced with the SAME window size / swap position the panel will use (Astra S1e item 4b); the panel and trainer refuse a total
# made for other settings; a total is only defined for a COMPLETE plan (six valid assignments)
PLAN = WP.plan_donors(_drv, a.window_latents, a.swap_at, lambda sc, t: os.path.exists(f"data/sim/keys_tpv10/{sc}/t{int(t):04d}.jpg"))
plan_ok = bool(PLAN["ok"] and len(PLAN["donors"]) == 6)
panel_base_h = WP.panel_render_h(2 * dev_ticks, roll_tick, load_s)
panel_h = WP.panel_render_h(2 * dev_ticks, roll_tick, load_s, donors=PLAN["donors"]) if plan_ok else None
L = WP.Ledger(a.ledger, "forecast", write=False)
kinds = L.by_kind()
# overfit verdict (Astra / Neon 08:30): 8 clips sampled in THREE modes (30 steps, batch-1 window cost as a bound), four decodes per
# clip (mid-stream clips decode with context: x2), ~20 frames x (4 renders + 1 key) segmented per clip on the slice
verdict_h = (3 * 8 * J["window30_rows1_s"] + 2 * 4 * 8 * J["wan_decode_5lat_s"] + 8 * 20 * 5 * a.overfit_score_tick_s) / 3600
SEL = J.get("selected_setting") or {}
fwd8 = (J.get("sweep") or {}).get(SEL.get("key", ""), {}).get("fwd_s")
if fwd8 is None:
    fwd8 = max(v["fwd_s"] for v in (J.get("sweep") or {}).values() if "fwd_s" in v)
evals_h = (2000 // 100 + 2) * 3 * fwd8 / 3600  # recurring fixed-noise overfit evals: every 100 updates + start + end, 3 sigmas
prod_train = len(L_["train"])
startup, st_info = WP.startup_for_forecast(J, prod_train)
if startup is None:
    raise SystemExit("timing.json has no measured start-up: re-run MODE=timing with the current trainer")
recs = WP.gate_records(a.gates)
GK = J.get("gate_key")  # the gate key of the SELECTED setting (what overfit / train will run): Astra S1c item 5
done = {k for k in ("overfit", "resume_test") if GK and any(r.get("kind") == k and r.get("status") == "PASS" and r.get("gate_key") == GK for r in recs)}
t = {"budget_h": a.budget_hours, "used_h": L.hours(live=False), "done_kinds": done, "wall_s": J["production_loop"]["wall_s"],
     "startup_s": startup, "probe_s": J["probe_s"], "val_every": a.val_every, "save_s": J["save_s"], "save_every": a.save_every,
     "n_numbered": a.n_numbered, "job_h": a.job_h, "panel_h": panel_h if panel_h is not None else panel_base_h, "n_panels": a.n_panels,
     "overfit_updates": 2000, "resume_updates": 32, "overfit_verdict_h": verdict_h, "overfit_evals_h": evals_h}
fc = WP.forecast_s1(t)
fc2 = WP.forecast_s1({**t, "n_panels": 2})
two = {"updates": fc2["updates"], "floor_12k_reachable": fc2["floor_12k_reachable"], "available_h": fc2["available_h"],
       "note": "no spare for a second read" if fc2["updates"] < WP.FLOOR_UPDATES else "a second read fits inside the 24 h"}
score = {}
if os.path.exists(a.score_parity):
    PR = json.load(open(a.score_parity))
    if PR.get("score_tick_s"):
        score = {"score_tick_s_2080": PR["score_tick_s"], "device": PR.get("device"),
                 "score_gpu_h_per_panel": round(WP.panel_score_h(2 * dev_ticks, float(PR["score_tick_s"])), 3),
                 "score_cap_h_per_panel": round(WP.score_cap_h(2 * dev_ticks, float(PR["score_tick_s"])), 3),
                 "note": "2080 Ti hours, own ledger runs/wan_score_ledger.jsonl; not slice hours"}
fc.update(inputs={k: (sorted(v) if isinstance(v, set) else v) for k, v in t.items()}, n_panels=a.n_panels, two_panels=two,
          dev_ticks=dev_ticks, panel_h=round(panel_h, 3) if panel_h is not None else None, panel_plan_complete=panel_h is not None,
          panel_cost_basis="render_only_load_once", shared_ref_start_s=0.0, gate_key=GK,
          panel_settings={"window_latents": a.window_latents, "swap_at": a.swap_at}, rollout_window_latents=RH["window_latents"],
          panel_base_h=round(panel_base_h, 3), panel_h_total=round(panel_h, 3) if panel_h is not None else None,
          donor_plan=PLAN["donors"], donor_plan_ok=plan_ok, startup_s=round(startup, 1), startup_info=st_info,
          timing_train_drives=J.get("train_drives"), prod_train_drives=prod_train, roll_tick_s=round(roll_tick, 4), roll_start_s=round(load_s, 1),
          rehearsal=RHP, wall_basis=J["production_loop"].get("basis"), wall_detail={k: J["production_loop"].get(k) for k in ("full_run_mean", "last_two_blocks")},
          scoring=score, ledger_by_kind=kinds)
try:
    SP = WP.score_parity_ok(a.score_parity, WP.score_binding(a.a_metrics, a.a_roll))
except Exception as e:  # noqa: BLE001
    SP = (False, f"cannot check ({type(e).__name__}: {e})")
RT = WP.retime_rules(J, fc, prod_train, SP, RH)
J["forecast_final"] = fc
J["retime_pass"] = RT
json.dump(J, open(f"{a.timing_dir}/timing.json", "w"), indent=1, default=str)
print("WAN_FORECAST", json.dumps({k: v for k, v in fc.items() if k not in ("donor_plan", "inputs")}, default=str))
print(f"FORECAST one panel: {fc['updates']} updates (floor 12k {'reachable' if fc['floor_12k_reachable'] else 'NOT reachable'}, target "
      f"{WP.UPDATE_TARGET}); two panels: {two['updates']} ({two['note']})")
for k, v in RT["rules"].items():
    print(f"RETIME {'PASS' if v['pass'] else 'FAIL'} {k} {json.dumps({x: y for x, y in v.items() if x != 'pass'}, default=str)}")
print("RETIME_PASS", "ALL PASS" if RT["all_pass"] else f"FAIL {RT['failing']}")
