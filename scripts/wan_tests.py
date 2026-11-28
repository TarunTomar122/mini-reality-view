"""CPU tests of the S1 planning rules (mrv/wan_plan.py; Astra items 3, 11, 12). Plain python3, no torch:
  python3 scripts/wan_tests.py      -> 'WAN_TESTS FAILS=0'
Soft reject (ok False, hard False, manifest seq changes) stays inside one codec stream but no training clip covers it; a hard
tick and a tick-id gap restart the stream; a short tail gets an overlapping window; candidates are counted; window borders /
swap snapping; the S1 decision rules (12k floor, extension band, final bar); the ledger charges preempted work.
"""
import json, os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrv import wan_plan as WP

FAILS = 0


def check(name, cond, info=""):
    global FAILS
    print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info and not cond else ""))
    FAILS += int(not cond)


# --- segments ---------------------------------------------------------------------------------------------------------
n = 100
ticks = list(range(n)); hard = [False] * n; ok = [True] * n
ok[20] = False                       # soft reject (offset-only): stream continues
hard[70] = True                      # hard break
ticks2 = ticks[:85] + [t + 3 for t in ticks[85:]]  # tick-id gap between positions 84 and 85
segs = WP.segments(ticks2, hard)
check("soft reject stays inside one segment", segs[0] == list(range(0, 70)), segs[0][:3])
check("hard tick breaks and is excluded", 70 not in sum(segs, []) and segs[1][0] == 71)
check("tick gap breaks", segs[1] == list(range(71, 85)) and segs[2] == list(range(85, 100)), [s[:2] for s in segs])

# --- streams / candidates -------------------------------------------------------------------------------------------
specs = WP.stream_specs(segs, phases=4, clip_latents=5)
check("streams only for segments with >= 1 + 5 latents", all(sp["seg"] == 0 for sp in specs) and len(specs) == 4, specs)
check("stream frames 4L-3 inside the segment", all(sp["pos0"] + sp["n_frames"] <= 70 for sp in specs))
mids = WP.mid_candidates(specs, ok, hard, 5)
cover = lambda sp, j0: range(sp["pos0"] + 4 * j0 - 3, sp["pos0"] + 4 * j0 + 17)
check("no mid clip covers the soft reject", all(20 not in cover(specs[k], j0) for k, j0 in mids), mids[:4])
check("mid clips exist and start at j0 >= 1", len(mids) > 0 and all(j0 >= 1 for _, j0 in mids))
check("every mid clip covers 20 ticks", all(len(cover(specs[k], j0)) == 20 for k, j0 in mids))
starts = WP.start_candidates(segs, ok, hard, frames=17, stride=4)
check("true starts avoid the soft reject and fit in a segment", all(not (s <= 20 < s + 17) for s in starts)
      and all(any(s in sg and s + 16 in sg for sg in segs) for s in starts), starts)
check("true starts on the stride grid", all(s % 4 == 0 for s in starts))
# brute-force count: phase streams of segment 0 (70 ticks): L = 18, 17, 17, 17; clips j0 = 1..L-5 whose 20 ticks avoid tick 20
bf = sum(1 for sp_ in specs for j0 in range(1, sp_["L"] - 4) if 20 not in cover(sp_, j0))
check("candidate count matches brute force", len(mids) == bf and bf > 0, (len(mids), bf))
check("true-start count", starts == [0, 24, 28, 32, 36, 40, 44, 48, 52], starts)

# --- spans / slots -----------------------------------------------------------------------------------------------------
sp = WP.latent_spans(5, True)
check("true-start spans: 17 distinct frames", sorted(set(sum(sp, []))) == list(range(17)) and sp[0] == [0, 0, 0, 0])
check("mid spans: 20 frames", sum(WP.latent_spans(5, False), []) == list(range(20)))
check("token slots", WP.token_slots([1, 2, 3, 4], "last") == [4] and WP.token_slots([1, 2, 3, 4], "ends") == [1, 4] and WP.token_slots([0, 0, 0, 0], "ends") == [0, 0])

# --- windows / short tail / swap snapping --------------------------------------------------------------------------
w = WP.plan_windows(13, 5)
check("tail window overlaps and keeps only new latents", w == [(0, 5, 0), (5, 10, 5), (8, 13, 10)], w)
check("short stream: one window", WP.plan_windows(3, 5) == [(0, 3, 0)])
check("whole-segment window", WP.plan_windows(9, 0) == [(0, 9, 0)])
check("every latent produced exactly once", sorted(sum([list(range(n0, w1)) for _, w1, n0 in WP.plan_windows(23, 5)], [])) == list(range(23)))
seg_short = list(range(7))  # short tail segment of 7 ticks -> 3 latents (frames 9, two padded)
check("short tail: latents cover all ticks", 4 * (1 + -(-(len(seg_short) - 1) // 4)) - 3 >= len(seg_short))
ff = WP.window_first_frames(13, 5)
check("window first frames", ff == [0, 17, 37], ff)
s = WP.snap_swap([list(range(100))], 50, 5)
check("swap snapped to the next window start", s == 57, s)

# --- decisions ---------------------------------------------------------------------------------------------------------
def cmp(car, road=0.50, roadA=0.49, idr=-0.2, wr=-0.2, cp=True):
    return {"drives": 6, "runs": 12, "unpaired_drives": [], "cold_ref": "B", "car_f1": {"A": car, "B": 0.4456},
            "road_iou": {"A": road, "B": roadA}, "ID": {"rel": idr, "can_pass": cp}, "W": {"rel": wr, "can_pass": cp}}
PROD = WP.production_ok({"timing": False, "recipe": {"batch": 8, "overfit": 0}, "args": {"kind": ""}})
MODES = {"ok": True, "why": []}
D = lambda c, st, stage, sw=True, prod=PROD, modes=MODES, prior=None: WP.decide_s1(c, st, stage, {"ok": sw, "blocked": [], "donor_f1": 0.5,
                                                                                                  "receiver_f1": 0.2}, prod, modes, prior)
check("production checkpoint ok", PROD["ok"], PROD)
check("batch 4 is not production", not WP.production_ok({"timing": False, "recipe": {"batch": 4, "overfit": 0}, "args": {}})["ok"])
check("timing / overfit / resume_test kinds are not production",
      not any(WP.production_ok(m)["ok"] for m in ({"timing": True, "recipe": {"batch": 8}}, {"recipe": {"batch": 8, "overfit": 8}},
                                                  {"recipe": {"batch": 8}, "args": {"kind": "resume_test"}})))
check("below 12k: unresolved", D(cmp(0.50), 11999, "24h")["verdict"].startswith("unresolved"))
check("24h pass", D(cmp(0.46), 12000, "24h")["verdict"].startswith("PASS"))
check("24h extension band", D(cmp(0.43), 14000, "24h")["verdict"].startswith("EXTEND"))
check("24h extension band lower edge 0.4156", D(cmp(0.4156), 14000, "24h")["verdict"].startswith("EXTEND"))
check("24h below band: stop", D(cmp(0.41), 14000, "24h")["verdict"].startswith("STOP"))
check("24h band but W fails: stop", D(cmp(0.43, wr=-0.05), 14000, "24h")["verdict"].startswith("STOP"))
check("24h band but swap fails: stop", D(cmp(0.43), 14000, "24h", sw=False)["verdict"].startswith("STOP"))
check("24h read only once", D(cmp(0.46), 14000, "24h", prior="EXTEND 12 h")["verdict"].startswith("INVALID"))
check("24h needs a production checkpoint", D(cmp(0.46), 14000, "24h", prod={"ok": False, "why": ["batch 4"]})["verdict"].startswith("INVALID"))
check("24h needs the four full-length modes", D(cmp(0.46), 14000, "24h", modes={"ok": False, "why": ["tokens missing"]})["verdict"].startswith("INVALID"))
check("final without EXTEND: invalid", D(cmp(0.46), 20000, "final")["verdict"].startswith("INVALID"))
check("final without EXTEND (prior STOP): invalid", D(cmp(0.46), 20000, "final", prior="STOP -> re-council")["verdict"].startswith("INVALID"))
check("final in band: stop (bar stays 0.4456)", D(cmp(0.44), 20000, "final", prior="EXTEND 12 h")["verdict"].startswith("STOP"))
check("final pass", D(cmp(0.45), 20000, "final", prior="EXTEND 12 h")["verdict"].startswith("PASS"))
check("road below A fails", not D(cmp(0.46, road=0.48), 12000, "24h")["all_pass"])
check("ineligible ID fails", not D(cmp(0.46, cp=False), 12000, "24h")["all_pass"])
check("NaN car F1 fails", not D(cmp(float("nan")), 12000, "24h")["all_pass"])
check("incomplete panel fails", not D({**cmp(0.46), "runs": 11}, 12000, "24h")["all_pass"])
check("check stage gives no verdict", D(cmp(0.46), 12000, "check")["verdict"].startswith("informational"))
SW_OK = {"ok": True, "blocked": [], "donor_f1": 0.5, "receiver_f1": 0.2}
DS = lambda c, st, stage, sw, prior=None: WP.decide_s1(c, st, stage, sw, PROD, MODES, prior)
check("passing quality + ZERO swap runs at 12k -> INVALID, not STOP (Astra repro)",
      DS(cmp(0.46), 12000, "24h", {"ok": False, "blocked": ["run ids: 0 runs over 0 drives"], "donor_f1": float("nan"), "receiver_f1": float("nan")})["verdict"].startswith("INVALID"))
check("blocked swap evidence -> INVALID", DS(cmp(0.41), 14000, "24h", {**SW_OK, "ok": False, "blocked": ["windows differ"]})["verdict"].startswith("INVALID"))
check("missing swap (None) -> INVALID", DS(cmp(0.46), 14000, "24h", None)["verdict"].startswith("INVALID"))
check("complete swap evidence that misses the rule -> STOP", DS(cmp(0.46), 14000, "24h", {**SW_OK, "ok": False})["verdict"].startswith("STOP"))
check("complete evidence below thresholds -> STOP", DS(cmp(0.40), 14000, "24h", SW_OK)["verdict"].startswith("STOP"))
check("incomplete compare panel -> INVALID", DS({**cmp(0.46), "runs": 11}, 14000, "24h", SW_OK)["verdict"].startswith("INVALID"))
check("final with blocked swap -> INVALID", DS(cmp(0.46), 20000, "final", {**SW_OK, "blocked": ["x"]}, "EXTEND 12 h")["verdict"].startswith("INVALID"))


def swaprun(df=0.5, rf=0.2, frames=20, wm=True, excl=False, rec=True, start=0):
    return {"donor_f1": df, "receiver_f1": rf, "donor_frames": frames, "receiver_frames": frames, "windows_match": wm,
            "lag": {"excluded": excl, "recovered": rec, "start": start if rec else None}}
RUNS = {f"scene-{d:04d}_s{sd}": swaprun() for d in range(6) for sd in (0, 1)}
check("donor following on full paired evidence", WP.donor_following(RUNS)["ok"], WP.donor_following(RUNS))
check("swap: missing run id blocks", not WP.donor_following({k: v for k, v in list(RUNS.items())[1:]})["ok"])
check("swap: one drive and one lag value does not pass", not WP.donor_following({"scene-0000_s0": swaprun()})["ok"])
check("swap: mismatched windows block", not WP.donor_following({**RUNS, "scene-0003_s1": swaprun(wm=False)})["ok"])
check("swap: unequal frames block", not WP.donor_following({**RUNS, "scene-0002_s0": {**swaprun(), "receiver_frames": 19}})["ok"])
check("swap: zero frames block", not WP.donor_following({**RUNS, "scene-0002_s0": swaprun(frames=0)})["ok"])
check("swap: a drive without lag evidence blocks",
      not WP.donor_following({**RUNS, "scene-0004_s0": swaprun(excl=True), "scene-0004_s1": swaprun(excl=True)})["ok"])
r_ = WP.donor_following({**RUNS, "scene-0004_s0": swaprun(excl=True)})
check("swap: exclusions reported, other seed carries the drive", r_["ok"] and r_["lag_excluded_runs"] == ["scene-0004_s0"], r_)
nr = {k: (swaprun(rec=False) if i < 7 else v) for i, (k, v) in enumerate(RUNS.items())}
check("swap: non-recoveries count as infinity", WP.donor_following(nr)["median_lag_start"] == float("inf") and not WP.donor_following(nr)["ok"])
check("swap: margin 0.10 not met", not WP.donor_following({k: swaprun(df=0.29, rf=0.2) for k in RUNS})["ok"])
check("swap: margin met with room", WP.donor_following({k: swaprun(df=0.31, rf=0.20) for k in RUNS})["ok"])

# --- Astra S1c repro fixtures ----------------------------------------------------------------------------------------
c_inel = {**cmp(0.46), "ID": {"rel": -0.2, "can_pass": False}}
check("S1c repro 1: complete swap, passing F1, finite ID gain but ID can_pass False -> INVALID, not STOP",
      DS(c_inel, 12000, "24h", SW_OK)["verdict"].startswith("INVALID") and "eligible" in DS(c_inel, 12000, "24h", SW_OK)["verdict"])
check("S1c: W can_pass False -> INVALID", DS({**cmp(0.40), "W": {"rel": -0.2, "can_pass": False}}, 14000, "24h", SW_OK)["verdict"].startswith("INVALID"))
rcv = {"name": "r", "ticks": list(range(150)), "ok": [True] * 150, "hard": [False] * 150}
dnr = {"name": "d", "ticks": list(range(150)), "ok": [i >= 77 for i in range(150)], "hard": [False] * 150}
pfr = WP.swap_preflight(rcv, dnr, W=5, swap_at=50)
check("S1c repro 2: swap at 57, donor accepted only from the next restart (77) -> 0 scorer lag windows, preflight fails",
      pfr["swap_pos"] == 57 and pfr["first_reset"] == 77 and pfr["lag_windows"] == 0 and not pfr["ok"], pfr)
pfo = WP.swap_preflight(rcv, {**dnr, "ok": [True] * 150}, W=5, swap_at=50)
check("S1c: scorer lag range = (swap, first restart) exclusive of restarts: 15 windows between 57 and 77", pfo["lag_windows"] == 15 and pfo["ok"], pfo)
pfa = WP.swap_preflight({**rcv, "ok": [i < 57 for i in range(150)]}, {**dnr, "ok": [True] * 150}, W=5, swap_at=50)
check("S1c: receiver not accepted after the swap -> no common frames -> fails", pfa["common_frames"] == 0 and not pfa["ok"], pfa)

# --- Astra S1d repro fixtures ----------------------------------------------------------------------------------------
import re as _re
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wan_rollout.py")).read()
check("rollout option mirror == scripts/wan_rollout.py options", _re.findall(r'add_argument\("--([a-z_0-9]+)"', _src) == WP.ROLLOUT_OPTIONS,
      _re.findall(r'add_argument\("--([a-z_0-9]+)"', _src))
w1, ex1 = WP.parse_rollout_extra(["--window=1"])
check("S1d repro 3: '--window=1' resolves to window_latents 1 for BOTH preflight and rollout", w1 == 1 and ex1 == ["--window_latents", "1"], (w1, ex1))
check("S1d: preflight at the parsed width (1) has the rollout's lag windows, not width 5's",
      WP.swap_preflight(rcv, {**dnr, "ok": [True] * 150}, W=w1, swap_at=50)["lag_windows"] != pfo["lag_windows"])
w5, ex5 = WP.parse_rollout_extra(["--steps", "20", "--cfg=2"])
check("rollout extras canonicalised with an explicit default width", w5 == 5 and ex5 == ["--steps", "20", "--cfg", "2.0", "--window_latents", "5"], ex5)
for bad_ in (["--max_t=10"], ["--sw"], ["--cond", "guide"], ["--drive", "x"], ["--seeds", "0"], ["--donors", "x"], ["--out", "y"]):
    try:
        WP.parse_rollout_extra(bad_); ok_ = False
    except SystemExit:
        ok_ = True
    check(f"rollout extras: forbidden / abbreviated {bad_} refused", ok_)
FC_OK = {"donor_plan_ok": True, "donor_plan": {f"r{i}": {"donor": "d", "mode": "full", "n": None} for i in range(6)}, "panel_h_total": 2.2, "panel_h": 2.0,
         "panel_settings": {"window_latents": 5, "swap_at": 50}, "rollout_window_latents": 5}
check("S1d: reserve = measured total x panels", abs(WP.train_reserve_h(FC_OK, 1) - 2.2) < 1e-12)
for name_, fc_ in (("S1d repro 4: legacy panel_h only (2.0 h vs a 2.2 h panel)", {"panel_h": 2.0}),
                   ("S1d: plan with an omitted receiver", {**FC_OK, "donor_plan": {**FC_OK["donor_plan"], "r5": None}}),
                   ("S1d: plan of five receivers", {**FC_OK, "donor_plan": {f"r{i}": {"donor": "d"} for i in range(5)}}),
                   ("S1d: plan not ok", {**FC_OK, "donor_plan_ok": False}),
                   ("S1d: no measured total", {**FC_OK, "panel_h_total": None})):
    try:
        WP.train_reserve_h(fc_, 1); ok_ = False
    except SystemExit:
        ok_ = True
    check(f"{name_} -> train / extend refused", ok_)

# --- Astra S1e repro fixtures ----------------------------------------------------------------------------------------
FC5 = {**FC_OK, "panel_settings": {"window_latents": 5, "swap_at": 50}}
for name_, args_ in (("S1e repro 4a: --reserve_panels 0", (FC5, 0, None)), ("S1e: --reserve_panels -1", (FC5, -1, None)),
                     ("S1e repro 4a: --reserve_hours 0 below the measured 2.2 h", (FC5, 1, 0.0)),
                     ("S1e: --reserve_hours 1.0 below the measured 2.2 h", (FC5, 1, 1.0))):
    try:
        WP.production_reserve_h(*args_); ok_ = False
    except SystemExit:
        ok_ = True
    check(f"{name_} -> production refused", ok_)
check("S1e: an explicit reserve above the measured total is kept", abs(WP.production_reserve_h(FC5, 1, 3.0) - 3.0) < 1e-12)
check("S1e: default production reserve = measured total", abs(WP.production_reserve_h(FC5, 1, None) - 2.2) < 1e-12)
mkA = lambda n_, rej=(): {"ticks": list(range(n_)), "ok": [i not in rej for i in range(n_)], "hard": [False] * n_}
DRVE = {"A": mkA(200), "B": mkA(200, rej=range(50, 57)), "C": mkA(180), "D": mkA(180), "E": mkA(180), "F": mkA(180)}
p5, p2 = WP.plan_donors(DRVE, 5, 50), WP.plan_donors(DRVE, 2, 50)
sh = lambda pl: [r for r, v in pl["donors"].items() if v and v["mode"] == "shared"]
check("S1e repro 4b: both plans pass; width 5 needs no shared reference, width 2 needs one 180-tick reference",
      p5["ok"] and p2["ok"] and sh(p5) == [] and len(sh(p2)) == 1 and p2["donors"][sh(p2)[0]]["n"] == 180, (p5["donors"], p2["donors"]))
check("S1e: the shared reference is priced (2 seeds x 180 ticks + start)",
      abs(WP.panel_with_shared_h(2.0, p2["donors"], 0.5, 60) - (2.0 + (60 + 2 * 180 * 0.5) / 3600)) < 1e-9)
for w_ in (2, 1):
    try:
        WP.production_reserve_h(FC5, 1, None, window_latents=w_); ok_ = False
    except SystemExit:
        ok_ = True
    check(f"S1e: reserve priced for width 5 refused for a width-{w_} panel", ok_)
try:
    WP.production_reserve_h(FC5, 1, None, swap_at=60); ok_ = False
except SystemExit:
    ok_ = True
check("S1e: reserve priced for swap at 50 refused for swap at 60", ok_)
try:
    WP.production_reserve_h({k: v for k, v in FC_OK.items() if k != "panel_settings"}, 1, None); ok_ = False
except SystemExit:
    ok_ = True
check("S1e: legacy forecast without panel_settings refused", ok_)

# --- swap donor preflight (Astra / Neon 08:30) ---------------------------------------------------------------------------
mk = lambda n_, hard_at=(): {"ticks": list(range(n_)), "ok": [True] * n_, "hard": [i in hard_at for i in range(n_)]}
DRV = {"scene-0043": mk(109), "scene-0393": mk(200), "scene-0234": mk(200), "scene-0396": mk(200), "scene-0285": mk(190), "scene-0468": mk(200)}
pf = WP.swap_preflight({"name": "scene-0393", **DRV["scene-0393"]}, {"name": "scene-0043", **DRV["scene-0043"]})
check("preflight: shorter donor (0043 for 0393) fails", not pf["ok"] and not pf["full_length"], pf)
plan = WP.plan_donors(DRV)
check("donor plan: every receiver gets a donor at least as long (full mode here)", plan["ok"] and all(
      v["mode"] == "full" and len(DRV[v["donor"]]["ticks"]) >= len(DRV[r]["ticks"]) and v["donor"] != r for r, v in plan["donors"].items()), plan["donors"])
check("donor plan: chosen pairs pass windows / frames / lag", all(plan["checks"][r]["ok"] and plan["checks"][r]["windows_identical"]
                                                             and plan["checks"][r]["common_frames"] > 0 and plan["checks"][r]["lag_windows"] > 0 for r in DRV))
pf2 = WP.swap_preflight({"name": "a", **mk(200)}, {"name": "b", **mk(200, hard_at=(120,))})
check("preflight: donor hard tick after the swap changes windows -> fails", not pf2["windows_identical"] and not pf2["ok"], pf2)
pf3 = WP.swap_preflight({"name": "a", **mk(200)}, {"name": "b", **mk(200)}, has_key=lambda sc, t: sc != "b")
check("preflight: no donor keys -> no common frames / lag -> fails", pf3["common_frames"] == 0 and not pf3["ok"], pf3)
pl2 = WP.plan_donors({"x": mk(300), "y": mk(100), "z": mk(100)})
check("donor plan: longest drive falls back to SHARED mode at the longest donor's length",
      pl2["ok"] and pl2["donors"]["x"] == {"donor": "y", "mode": "shared", "n": 100} and pl2["donors"]["y"]["mode"] == "full", pl2["donors"])
check("donor plan: impossible when no donor passes", not WP.plan_donors({"x": mk(300), "y": mk(30)})["ok"])

# --- forecast (S1b item 5) --------------------------------------------------------------------------------------------
T_ = {"budget_h": 24.0, "used_h": 1.0, "done_kinds": {"timing"}, "wall_s": 2.0, "startup_s": 300.0, "probe_s": 30.0, "val_every": 500,
      "save_s": 20.0, "save_every": 500, "n_numbered": 10, "job_h": 12.0, "panel_h": 1.0, "n_panels": 2, "overfit_updates": 2000,
      "resume_updates": 32, "overfit_verdict_h": 0.2}
fc = WP.forecast_s1(T_)
exp_pend = (300 + 2000 * 2) / 3600 + 0.2 + (900 + 64) / 3600 + 2.0
exp_avail = 24 - 1 - exp_pend - 10 * 20 / 3600
check("forecast deducts overfit / resume test / panels / numbered saves", abs(fc["available_h"] - exp_avail) < 1e-3, fc)
exp_ups = int((exp_avail - 2 * 300 / 3600) * 3600 / (2.0 + 30 / 500 + 20 / 500))
check("forecast updates after job starts and recurring probes / saves", fc["updates"] == exp_ups and fc["training_jobs"] == 2, (fc, exp_ups))
check("forecast charges recurring overfit evals", abs(WP.forecast_s1({**T_, "overfit_evals_h": 0.5})["pending"]["overfit_h"] - (exp_pend - 2.0 - (900 + 64) / 3600 + 0.5)) < 1e-3)
check("forecast skips charged kinds", "overfit_h" not in WP.forecast_s1({**T_, "done_kinds": {"overfit", "resume_test"}})["pending"])
check("forecast floor flag", WP.forecast_s1({**T_, "wall_s": 7.0})["floor_12k_reachable"] is False)
check("panel cost", abs(WP.panel_cost_h(1000, 0.5, 60, 1.0, 120) - (5 * (60 + 500) + 8 * (120 + 1000)) / 3600) < 1e-9)

# --- joint dropout / overfit reading (Fable 07:30) ----------------------------------------------------------------------
import random as _r
_r.seed(0); U = [_r.random() for _ in range(200000)]
gd, td = WP.drop_masks(U, 0.10, 0.10, 0.10)
both = sum(1 for g_, t_ in zip(gd, td) if g_ and t_) / len(U); gm = sum(gd) / len(U); tm = sum(td) / len(U)
check("joint dropout: both 10 %, guide 20 %, tokens 20 %", abs(both - 0.10) < 0.005 and abs(gm - 0.20) < 0.005 and abs(tm - 0.20) < 0.005, (both, gm, tm))
gd0, td0 = WP.drop_masks(U[:1000], 0, 0, 0)
check("joint dropout: all zero shares drop nothing", not any(gd0) and not any(td0))
try:
    WP.drop_masks([0.5], 0.5, 0.4, 0.2); ok_ = False
except ValueError:
    ok_ = True
check("joint dropout: shares over 1 refused", ok_)
H = [(i * 100, 1.0 - 0.01 * i) for i in range(21)]
OS = lambda loss_ok, f1_ok, rt=True, fh=0.6, fn=0.1, hist=H, prior=0: WP.overfit_status(loss_ok, f1_ok, rt, fh, fn, hist, prior)
check("overfit: both gates -> PASS", OS(True, True) == "PASS")
check("overfit: loss missed, still falling -> NOT_REACHED", OS(False, False).startswith("NOT_REACHED"))
check("overfit: loss missed, placement passing -> NOT_REACHED", OS(False, True, hist=[(0, 1.0), (100, 1.0)]).startswith("NOT_REACHED"))
check("overfit: second NOT_REACHED for the same key -> FAIL", OS(False, True, prior=1).startswith("FAIL (second"))
check("overfit: loss flat and placement failing -> FAIL", OS(False, False, hist=[(i * 100, 0.5) for i in range(10)]) == "FAIL")
check("overfit: loss ok but placement failing -> FAIL", OS(True, False) == "FAIL")
check("overfit: no round-trip cars -> FAIL", OS(True, True, rt=False).startswith("FAIL"))
check("overfit: hybrid no better than no-inputs (+0.02) -> FAIL even with both gates", OS(True, True, fh=0.30, fn=0.29).startswith("FAIL (hybrid"))
check("overfit: hybrid exactly none + 0.02 -> FAIL", OS(True, True, fh=0.32, fn=0.30).startswith("FAIL (hybrid"))
check("overfit: hybrid clearly above none -> PASS", OS(True, True, fh=0.40, fn=0.30) == "PASS")
with tempfile.TemporaryDirectory() as td:
    gp = f"{td}/gates.jsonl"
    rc_ = {"rank": 32, "tok_ticks": "last", "use_tokens": True, "shift": 2.0, "sigma_dist": "shifted_uniform", "lr": 1e-4, "warmup": 500,
           "prompt_embed": "", "wan": "w", "batch": 8, "lists": "a", "p_drop_both": 0.1}
    k1 = WP.gate_key("c1", rc_, "b", "v", {"g": 1})
    check("gate key ignores lists / dropout", k1 == WP.gate_key("c1", {**rc_, "lists": "b", "p_drop_both": 0.0}, "b", "v", {"g": 1}))
    check("gate key binds code / recipe / guide version", len({k1, WP.gate_key("c2", rc_, "b", "v", {"g": 1}), WP.gate_key("c1", {**rc_, "rank": 16}, "b", "v", {"g": 1}),
                                                              WP.gate_key("c1", rc_, "b", "v", {"g": 2})}) == 4)
    check("no gate records -> train refused", not WP.gates_ok(gp, k1)[0])
    WP.gate_append(gp, {"kind": "overfit", "gate_key": k1, "status": "NOT_REACHED (x)"})
    check("one NOT_REACHED counted", WP.prior_not_reached(gp, k1) == 1 and WP.prior_not_reached(gp, "other") == 0)
    WP.gate_append(gp, {"kind": "overfit", "gate_key": k1, "status": "PASS"})
    check("overfit PASS alone is not enough", not WP.gates_ok(gp, k1)[0])
    WP.gate_append(gp, {"kind": "resume_test", "gate_key": "otherkey", "status": "PASS"})
    check("resume PASS for another key does not count", not WP.gates_ok(gp, k1)[0])
    WP.gate_append(gp, {"kind": "resume_test", "gate_key": k1, "status": "PASS"})
    check("overfit + resume PASS for the key -> train allowed", WP.gates_ok(gp, k1)[0])

# --- split membership (S1b item 2) --------------------------------------------------------------------------------------
LS = {"train": ["scene-0001", "scene-0002"], "dev": ["scene-0010"], "decision": ["scene-0100"], "final": ["scene-0200"]}
check("splits: train + dev accepted", WP.check_split_membership(["scene-0001", "scene-0010"], LS) == ["scene-0001", "scene-0010"])
for bad_, why in ((["scene-0100"], "sealed decision"), (["scene-0200"], "sealed final"), (["scene-9999"], "unknown scene")):
    try:
        WP.check_split_membership(bad_, LS); ok_ = False
    except SystemExit:
        ok_ = True
    check(f"splits: {why} refused", ok_)
try:
    WP.check_split_membership(["scene-0001"], LS, allowed=("dev",)); ok_ = False
except SystemExit:
    ok_ = True
check("splits: train drive refused where only dev is allowed", ok_)
check("splits: sealed allowed with unseal", WP.check_split_membership(["scene-0100"], LS, allowed=("dev",), unseal=True) == ["scene-0100"])

# --- ledger --------------------------------------------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    path = f"{td}/ledger.jsonl"
    now = time.time()
    with open(path, "w") as f:  # j1 preempted (no end, stale beat), j2 finished, j4 still beating, torn last line
        f.write(json.dumps({"job": "j1", "kind": "train", "t0": now - 7200, "t": now - 7200, "beat_s": 60}) + "\n")
        f.write(json.dumps({"job": "j1", "kind": "train", "t0": now - 7200, "t": now - 3600, "beat_s": 60}) + "\n")
        f.write(json.dumps({"job": "j2", "kind": "panel", "t0": now - 1800, "t": now - 900, "beat_s": 60, "note": "end"}) + "\n")
        f.write(json.dumps({"job": "j4", "kind": "alloc_timing", "t0": now - 600, "t": now - 30, "beat_s": 60}) + "\n")
        f.write('{"job": "j3", "ki')
    L = WP.Ledger(path, "train", write=False)
    h = L.hours(live=False)
    exp = (3600 + 60) / 3600 + 0.25 + 600 / 3600
    check("ledger: preempted job charged its last interval, live job up to now, finished job exactly", abs(h - exp) < 2e-3, (h, exp))
    bk = L.by_kind()
    check("ledger by kind", abs(bk["train"] - (3660 / 3600)) < 1e-6 and abs(bk["panel"] - 0.25) < 1e-6 and abs(bk["alloc_timing"] - 600 / 3600) < 2e-3, bk)
    W_ = WP.Ledger(path, "panel", write=True); W_.start(); W_.stop()
    own = [json.loads(x) for x in open(path) if x.strip().startswith("{") and x.strip().endswith("}") and W_.job in x]
    check("ledger appends own beats after a torn line", len(own) >= 2 and own[-1].get("note") == "end", own)
    os.environ["WAN_LEDGER_OUTER"] = "1"
    R_ = WP.Ledger(path, "train"); n0 = sum(1 for _ in open(path)); R_.start(); R_.stop()
    check("ledger: inner processes under an outer allocation beat only read", sum(1 for _ in open(path)) == n0)
    del os.environ["WAN_LEDGER_OUTER"]


import subprocess as _sp, tempfile as _tf
# ---- Astra S1f 4a: production enforcement follows the run mode, not the --kind label
_src = open("scripts/train_wan_teacher.py").read()
check("S1f 4a: no enforcement keyed on the KIND label", 'KIND == "train"' not in _src and _src.count("if PROD") >= 6)
check("S1g 4a: PROD from the run mode only", "PROD = not (a.timing or a.overfit or a.resume_test)" in _src)
_tsh = open("slurm/wan_teacher.sh").read()
check("S1g 4a: train / extend pass --production after forwarded args", _tsh.count('--resume "$@" --production;') == 2)
check("S1g 4a: resume test uses the bounded flag", "--resume_test --kind resume_test --steps 16" in _tsh)
for _args, _name in ((["--out", "runs/wan_timing_followup", "--production", "--timing"], "--production + --timing"),
                     (["--out", "runs/wan_ctest_x", "--resume_test", "--steps", "40000"], "--resume_test with 40,000 steps"),
                     (["--out", "runs/wan_s1", "--resume_test", "--steps", "16"], "--resume_test outside a ctest dir")):
    _r = _sp.run([sys.executable, "scripts/train_wan_teacher.py", *_args], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    _msg = _r.stderr + _r.stdout
    if "No module named" in _msg:
        print(f"SKIP S1g 4a repro: {_name} (trainer imports unavailable here; runs in every slurm job)"); continue
    check(f"S1g 4a repro: {_name} refused", _r.returncode != 0 and ("cannot be combined" in _msg or "bounded 16-update" in _msg), _msg[-160:])
for _w, _ok in ((5, True), (2, False), (None, False)):
    try:
        WP.train_reserve_h({**FC_OK, "rollout_window_latents": _w}, 1); _got = True
    except SystemExit:
        _got = False
    check(f"S1g 4b: stored forecast measured at width {_w} for a width-5 panel -> {'accepted' if _ok else 'refused'}", _got == _ok)
# ---- Astra S1f 4b (rates now from the load-once rehearsal): a forecast width different from the measured one is refused
_d = _tf.mkdtemp()
os.makedirs(f"{_d}/rehearsal")
json.dump({"production_loop": {"wall_s": 4.0}}, open(f"{_d}/timing.json", "w"))
for _w, _ask in ((5, 2), (None, 5)):
    json.dump({"window_latents": _w, "roll_tick_s": 0.7, "load_s": 380, "pass": True}, open(f"{_d}/rehearsal/rehearsal.json", "w"))
    _r = _sp.run([sys.executable, "scripts/wan_forecast.py", "--timing_dir", _d, "--window_latents", str(_ask)], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    check(f"S1f 4b: measured width {_w} vs forecast {_ask} refused", _r.returncode != 0 and "refused" in (_r.stderr + _r.stdout), _r.stderr[-200:])
os.remove(f"{_d}/rehearsal/rehearsal.json")
_r = _sp.run([sys.executable, "scripts/wan_forecast.py", "--timing_dir", _d], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
check("CONVERGED 4/7: no rehearsal record (slice panel cost unmeasured) -> forecast refused", _r.returncode != 0 and "rehearsal" in (_r.stderr + _r.stdout), _r.stderr[-200:])
_sh = open("slurm/wan_teacher.sh").read()
check("S1f 4b: the rehearsal renders at PANEL_WIN and the forecast prices PANEL_WIN",
      '--rollout_args "--window_latents $PANEL_WIN"' in _sh and "wan_forecast.py --timing_dir $O --ledger $LEDGER --window_latents $PANEL_WIN" in _sh)

# ---- Neon 09:30 P0: cache provenance must not hash the whole policy module
for _f in ("scripts/wan_cache.py", "scripts/wan_tokens.py"):
    _c = [l for l in open(_f) if l.startswith("CODE_SHA") or l.startswith("                    + ")]
    check(f"Neon 0930: {_f} code sha excludes mrv/wan_plan.py", _c and not any("wan_plan.py" in l for l in _c), _c)

# =========================================================================================================================
# council/s1-timing-r1-CONVERGED.md: F1 staging, F2 manifest, F3 off-slice scoring, F5 explicit micro / grad_ckpt, re-time rule
# =========================================================================================================================
from mrv import wan_stage as WS
import shutil as _sh_


def _refused(fn, *args, **kw):
    try:
        fn(*args, **kw); return False
    except SystemExit:
        return True


def _fixture(root, drives=("scene-0001", "scene-0002"), dev=("scene-0010",)):
    """tiny data tree: cache / tokens / guides / enc per drive with sidecars as the real builders write them, a lists file and a
    fake Wan snapshot (transformer/, vae/, model_index.json)"""
    R = {"cache": f"{root}/data/wan_cache_v1", "tokens": f"{root}/data/wan_tok_v1", "guides": f"{root}/data/guides_v1", "enc": f"{root}/data/enc_tpv10"}
    for d in R.values():
        os.makedirs(d, exist_ok=True)
    for sc in list(drives) + list(dev):
        for k, sz in (("guides", 300), ("enc", 200), ("cache", 500), ("tokens", 900)):
            open(f"{R[k]}/{sc}.pt", "wb").write((sc + k).encode() * sz)
        sha = lambda k: WS.sha256_file(f"{R[k]}/{sc}.pt")
        json.dump({"complete": True, "ticks": 3}, open(f"{R['guides']}/{sc}.json", "w"))
        json.dump({"scene": sc, "sha256": sha("cache"), "guide_sha": sha("guides"), "enc_sha": sha("enc"), "guide_version": {"v": 1}, "complete": True},
                  open(f"{R['cache']}/{sc}.json", "w"))
        json.dump({"scene": sc, "sha256": sha("tokens"), "guide_sha": sha("guides"), "enc_sha": sha("enc"), "guide_version": {"v": 1}, "complete": True,
                   "shared_pass_ok": True, "weights_sha": "w"}, open(f"{R['tokens']}/{sc}.json", "w"))
    json.dump({"train": list(drives), "dev": list(dev), "decision": [], "final": []}, open(f"{root}/lists.json", "w"))
    W = f"{root}/hf/snap"
    for sub, n in (("transformer", 3), ("vae", 2)):
        os.makedirs(f"{W}/{sub}", exist_ok=True)
        for i in range(n):
            open(f"{W}/{sub}/f{i}.bin", "wb").write(f"{sub}{i}".encode() * 1000)
    json.dump({"_class_name": "WanPipeline"}, open(f"{W}/model_index.json", "w"))
    return R, W


def _vm(root, R, W, out, extra=()):
    return _sp.run([sys.executable, os.path.abspath("scripts/verify_manifest.py"), "--out", out, "--lists", f"{root}/lists.json", "--cache", R["cache"],
                    "--tokens", R["tokens"], "--guides", R["guides"], "--enc", R["enc"], "--wan_dir", W, "--nproc", "2", *extra],
                   capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})


with _tf.TemporaryDirectory() as td:
    R, W = _fixture(td)
    mp = f"{td}/manifest.json"
    r_ = _vm(td, R, W, mp)
    check("F2: verify_manifest PASS on a consistent tree writes ONE manifest + its sha", r_.returncode == 0 and "VERIFY_MANIFEST PASS" in r_.stdout
          and os.path.exists(mp) and open(mp + ".sha256").read().split()[0] == WS.sha256_file(mp), r_.stderr[-300:])
    check("F2: the manifest is read-only", not os.access(mp, os.W_OK) or os.stat(mp).st_mode & 0o222 == 0)
    man, msha = WS.load_manifest(mp)
    n_exp = 3 * 7  # 3 drives x (cache, cache json, tokens, tokens json, guide, guide json, enc)
    check("F2: every cache / token / guide / enc file (+ sidecars) of train + dev and the Wan snapshot is in it",
          len(man["files"]) == n_exp and len(man["wan"]["files"]) == 6 and all(len(e["sha256"]) == 64 for e in man["files"].values()), (len(man["files"]), len(man["wan"]["files"])))
    check("F2: an existing manifest is never overwritten", _vm(td, R, W, mp).returncode != 0)
    # repro: one corrupted token byte -> refused, no manifest
    b = bytearray(open(f"{R['tokens']}/scene-0002.pt", "rb").read()); b[5] ^= 1; open(f"{R['tokens']}/scene-0002.pt", "wb").write(bytes(b))
    r2 = _vm(td, R, W, f"{td}/m2.json")
    check("F2 repro: token file != its sidecar sha -> VERIFY_MANIFEST FAIL and no manifest", r2.returncode != 0 and not os.path.exists(f"{td}/m2.json")
          and "mismatch" in (r2.stderr + r2.stdout), r2.stderr[-300:])
    b[5] ^= 1; open(f"{R['tokens']}/scene-0002.pt", "wb").write(bytes(b))
    gj = json.load(open(f"{R['tokens']}/scene-0001.json")); gj["guide_sha"] = "0" * 64; json.dump(gj, open(f"{R['tokens']}/scene-0001.json", "w"))
    r3 = _vm(td, R, W, f"{td}/m3.json")
    check("F2 repro: token cache built from another guide than the latent cache -> refused", r3.returncode != 0 and not os.path.exists(f"{td}/m3.json"), r3.stderr[-300:])
    gj["guide_sha"] = WS.sha256_file(f"{R['guides']}/scene-0001.pt"); json.dump(gj, open(f"{R['tokens']}/scene-0001.json", "w"))
    # the manifest's own sha sidecar: an edited manifest is refused
    os.chmod(mp, 0o644); m_ = json.load(open(mp)); m_["created"] = 0; json.dump(m_, open(mp, "w")); os.chmod(mp, 0o444)
    check("F2 repro: a manifest edited after writing (sha != .sha256) is refused", _refused(WS.load_manifest, mp))
    # restore a valid manifest for the staging tests
    _sh_.rmtree(td + "/mfresh", ignore_errors=True); os.makedirs(td + "/mfresh")
    mp = f"{td}/mfresh/manifest.json"; r_ = _vm(td, R, W, mp); man, msha = WS.load_manifest(mp)
    roots = {k: WS.norm(v) for k, v in R.items()}
    st = f"{td}/stage"
    jobs, lay = WS.stage_plan(man, ["scene-0001", "scene-0002"], roots, st)
    srcs = [j[0] for j in jobs]
    check("F1: stage plan = cache + token .pt / .json of the job's drives + the Wan files; guide / enc never staged",
          len(jobs) == 2 * 4 + 6 and not any("/guides_v1/" in x or "/enc_tpv10/" in x for x in srcs), srcs)
    rec = WS.stage_files(jobs, streams=8)
    check("F1/F2: staged with 8 streams, every hash == manifest, bytes counted", rec["hash_equals_manifest"] and rec["files"] == len(jobs)
          and rec["bytes"] == sum(j[3] for j in jobs) and all(WS.sha256_file(j[1]) == j[2] for j in jobs), rec)
    check("F1: staged files land under the stage dir layout (cache/, tokens/, wan/)", os.path.exists(f"{lay['cache']}/scene-0001.pt")
          and os.path.exists(f"{lay['tokens']}/scene-0002.json") and os.path.exists(f"{lay['wan']}/transformer/f0.bin"))
    # repro: a same-size corruption after the manifest -> the hash-on-copy refuses (a size / mtime match is not proof)
    p0 = f"{R['cache']}/scene-0001.pt"; st0 = os.stat(p0)
    b = bytearray(open(p0, "rb").read()); b[0] ^= 1; open(p0, "wb").write(bytes(b)); os.utime(p0, (st0.st_atime, st0.st_mtime))
    check("F2 repro: same size + same mtime but different bytes -> no change alarm (alarm is not proof) ...", WS.change_alarm(man, [p0]) == [])
    check("F2 repro: ... and the hash-on-copy refuses the stage-in", _refused(WS.stage_files, WS.stage_plan(man, ["scene-0001"], roots, st + "2")[0]))
    b[0] ^= 1; open(p0, "wb").write(bytes(b)); os.utime(p0, (st0.st_atime, st0.st_mtime))
    g0 = f"{R['guides']}/scene-0002.pt"; os.utime(g0, (time.time(), time.time() + 5))
    check("F2 repro: a guide file touched after the manifest -> change alarm (trainer refuses: re-run verify_manifest)", len(WS.change_alarm(man, [g0])) == 1)
    open(f"{R['tokens']}/scene-0002.pt", "ab").write(b"x")
    check("F1 repro: a staged source whose size differs from the manifest is refused before copying",
          _refused(WS.stage_files, WS.stage_plan(man, ["scene-0002"], roots, st + "3")[0]))
    check("F2: a drive missing from the manifest is refused", _refused(WS.stage_plan, man, ["scene-9999"], roots, st + "4"))
    sb = WS.staged_bytes(man, ["scene-0001", "scene-0002"], roots)
    check("F1: memory need = staged bytes + 48 GB; --mem rounded up in G", WS.need_bytes(sb) == sb + 48 * 10 ** 9 and WS.mem_request_g(sb) * (1 << 30) >= WS.need_bytes(sb))

# cgroup readers (fake v2 / v1 trees) and the memory rules
with _tf.TemporaryDirectory() as td:
    os.makedirs(f"{td}/sys/fs/cgroup/job/step/task0")
    open(f"{td}/sys/fs/cgroup/job/memory.max", "w").write(str(150 * 10 ** 9))
    open(f"{td}/sys/fs/cgroup/job/step/task0/memory.max", "w").write("max")
    open(f"{td}/sys/fs/cgroup/job/step/task0/memory.peak", "w").write(str(120 * 10 ** 9))
    open(f"{td}/sys/fs/cgroup/job/step/task0/memory.current", "w").write(str(100 * 10 ** 9))
    WS_cg = WS.cgroup_mem
    # the reader walks up only inside /sys/fs/cgroup: emulate by pointing at the fixture (prefix check on the real path)
    rec = WS.cgroup_mem([("v2", f"{td}/sys/fs/cgroup/job/step/task0")])
    check("F1: cgroup v2 peak read from memory.peak", rec["peak"] == 120 * 10 ** 9 and rec["version"] == "v2", rec)
    os.makedirs(f"{td}/v1")
    open(f"{td}/v1/memory.limit_in_bytes", "w").write(str(96 * 10 ** 9)); open(f"{td}/v1/memory.max_usage_in_bytes", "w").write(str(90 * 10 ** 9))
    open(f"{td}/v1/memory.usage_in_bytes", "w").write("1")
    r1 = WS.cgroup_mem([("v1", f"{td}/v1")])
    check("F1: cgroup v1 limit / max_usage read", r1["limit"] == 96 * 10 ** 9 and r1["peak"] == 90 * 10 ** 9, r1)
_vals = iter([5, 9, 7, None, 3])
_ps = WS.PeakSampler(0.01, reader=lambda: next(_vals, None)).start(); time.sleep(0.2)
check("F1: the sampled peak keeps the maximum of the cgroup usage (memory.peak missing on old kernels)", _ps.stop() == 9 and _ps.n >= 4, (_ps.max, _ps.n))
ok_, msg_ = WS.mem_check(90 * 10 ** 9, 96 * 10 ** 9)
check("F1 repro: the timing job's 96 GB cannot hold ~90 GB staged + 48 GB -> refused with the --mem to request", ok_ is False and "--mem=" in msg_, msg_)
check("F1: a limit >= staged + 48 GB passes; an unreadable limit is UNKNOWN (not refused)",
      WS.mem_check(90 * 10 ** 9, 140 * 10 ** 9)[0] is True and WS.mem_check(90 * 10 ** 9, None)[0] is None)
check("re-time: peak incl. staged files <= 85 % of the request", WS.peak_rule(85, 100) == (True, 0.85) and WS.peak_rule(86, 100)[0] is False
      and WS.peak_rule(None, 100) == (None, None))
_tsh = open("slurm/wan_teacher.sh").read()
check("F1: wan_teacher.sh EXIT trap removes the stage dir; memcheck refuses (exit 9) before work",
      'cleanup() { stop_beat; [ -n "$STAGE" ] && rm -rf "$STAGE"; }' in _tsh and "trap cleanup EXIT" in _tsh and "mrv.wan_stage memcheck" in _tsh and "exit 9" in _tsh)
check("F1/F2/F5: every trainer call gets --manifest / --micro / --grad_ckpt / --stage_dir; MICRO / GRAD_CKPT unset refuses",
      "--manifest $MANIFEST --micro $MICRO --grad_ckpt $GRAD_CKPT --stage_dir $STAGE" in _tsh and '[ -n "$MICRO" ] && [ -n "$GRAD_CKPT" ] ||' in _tsh)
check("F2: the per-start re-hash (--verify_cache) is gone", "verify_cache" not in _tsh.split("--verify_cache) is gone")[0] and "--verify_cache 1" not in _tsh)
check("CONVERGED 7: start-up measured from the allocation (MRV_ALLOC_T0 exported first)", "export MRV_ALLOC_T0=$(date +%s.%N)" in _tsh.split("\n")[2])

# production-loop blocks and the forecast wall
rows = [{"wall_s": 4.0, "gpu_s": 3.9, "queue_wait_s": 0.01}] * 160 + [{"wall_s": 4.6, "gpu_s": 4.0, "queue_wait_s": 0.5}] * 64
bl = WS.block_stats(rows, 32)
wf, wd = WS.forecast_wall(rows, 32)
check("re-time: 224 updates -> 7 full blocks; per-block queue / wall and wall / gpu", len(bl) == 7 and bl[-1]["queue_frac"] == round(0.5 / 4.6, 4)
      and bl[0]["wall_over_gpu"] == round(4.0 / 3.9, 4), bl[-1])
check("re-time repro: a slowdown in the last two blocks prices the forecast (slowest of full mean / last two block means)",
      abs(wf - 4.6) < 1e-9 and wd["full_run_mean"] < 4.6, (wf, wd))
check("re-time: a partial tail block is dropped", len(WS.block_stats(rows[:200], 32)) == 6)

# F5: matched-update check and the selection
fits = {"gc1_m1": {"micro": 1, "grad_ckpt": 1, "gpu_s": 3.63}, "gc0_m1": {"micro": 1, "grad_ckpt": 0, "gpu_s": 2.80},
        "gc0_m2": {"micro": 2, "grad_ckpt": 0, "gpu_s": 2.70}, "gc1_m8": {"micro": 8, "grad_ckpt": 1, "gpu_s": 4.00}}
mrows = {"gc1_m1": {"loss": 0.500, "grad_norm": 1.00}, "gc0_m1": {"loss": 0.5004, "grad_norm": 1.01}, "gc0_m2": {"loss": 0.53, "grad_norm": 1.0},
         "gc1_m8": {"loss": 0.501, "grad_norm": 0.99}}
mu = WP.matched_updates(mrows, "gc1_m1", 0.02, 0.05)
sel = WP.select_setting(fits, mu)
check("F5 repro: a faster setting whose update does NOT match (loss +6 %) is never selected; fastest matched wins",
      not mu["rows"]["gc0_m2"]["ok"] and sel == {"key": "gc0_m1", "micro": 1, "grad_ckpt": 0, "gpu_s": 2.80}, (mu, sel))
check("F5: NaN loss never matches", not WP.matched_updates({"a": {"loss": 0.5, "grad_norm": 1.0}, "b": {"loss": float("nan"), "grad_norm": 1.0}}, "a")["rows"]["b"]["ok"])
check("F5: no matched setting -> no selection", WP.select_setting(fits, {"rows": {}}) is None)
rc0 = {"rank": 32, "tok_ticks": "last", "use_tokens": True, "shift": 2.0, "sigma_dist": "shifted_uniform", "lr": 1e-4, "warmup": 500, "prompt_embed": "",
       "wan": "w", "batch": 8, "micro": 1, "grad_ckpt": 1}
k0 = WP.gate_key("c", rc0, "b", "v", {"g": 1}, "m1")
check("F5 / F2: the gate key binds micro, grad_ckpt and the manifest sha", len({k0, WP.gate_key("c", {**rc0, "micro": 2}, "b", "v", {"g": 1}, "m1"),
      WP.gate_key("c", {**rc0, "grad_ckpt": 0}, "b", "v", {"g": 1}, "m1"), WP.gate_key("c", rc0, "b", "v", {"g": 1}, "m2")}) == 4)
_trs = open("scripts/train_wan_teacher.py").read()
check("F5: --micro / --grad_ckpt have no trainer default and are refused when missing; both in the recipe",
      'p.add_argument("--micro", type=int, default=None' in _trs and '"--grad_ckpt", type=int, default=None' in _trs
      and "must be passed explicitly in every mode" in _trs and '"micro", "grad_ckpt")}' in _trs)
check("F2: trainer requires --manifest, stages, loads Wan from the staged copy, records manifest sha beside DATA_SHA",
      "--manifest is required in every mode" in _trs and "WC.build(WAN_SRC," in _trs and 'from_pretrained(WAN_SRC, subfolder="vae"' in _trs
      and '"manifest_sha": MANIFEST_SHA, "guide_version"' in _trs and '("manifest_sha", MANIFEST_SHA)' in _trs and "verify_cache" not in _trs.split('"""', 2)[2])
check("F5: timing sweeps grad_ckpt x micro, matched-update check before selecting, --use_tokens 0 reading diagnostic only",
      "for gc in GCS:" in _trs and "WP.matched_updates(" in _trs and "WP.select_setting(" in _trs and '"diagnostic_only": True' in _trs)
check("CONVERGED 7: timing production loop >= 200 updates default, staged == Lustre batches, cgroup peak recorded",
      'p.add_argument("--timing_prod_updates", type=int, default=200' in _trs and "staged_vs_lustre" in _trs and "WS.cgroup_mem()" in _trs)
for _args, _want, _name in (([], "must be passed explicitly", "--micro / --grad_ckpt missing"),
                            (["--micro", "1", "--grad_ckpt", "1"], "--manifest is required", "--manifest missing")):
    with _tf.TemporaryDirectory() as _td:
        _r = _sp.run([sys.executable, "scripts/train_wan_teacher.py", "--out", f"{_td}/wan_timing_t", "--timing", *_args], capture_output=True, text=True,
                     env={**os.environ, "PYTHONPATH": os.getcwd()})
    _msg = _r.stderr + _r.stdout
    if "No module named" in _msg:
        print(f"SKIP CONVERGED repro: {_name} (trainer imports unavailable here; runs in every slurm job)"); continue
    check(f"CONVERGED repro: trainer {_name} -> refused", _r.returncode != 0 and _want in _msg, _msg[-200:])

# re-time rule (CONVERGED item 7)
def _J(**kw):
    J = {"train_drives": 144, "stage": {"cold": True, "hash_equals_manifest": True, "mismatches": [], "total_files": 600, "total_bytes": 9e10},
         "startup": {"alloc_to_first_update_s": 900.0, "process_to_first_update_s": 880.0, "stage_in_s": 780.0},
         "production_loop": {"updates": 224, "blocks": WS.block_stats([{"wall_s": 4.0, "gpu_s": 3.9, "queue_wait_s": 0.01}] * 224, 32)},
         "selected_setting": {"micro": 1, "grad_ckpt": 0}, "matched_update": {"all_ok": True}, "token_geometry": {"pass": True},
         "init_parity": {"pass": True}, "update0_sensitivity": {"pass": True}, "staged_vs_lustre": {"all_equal": True, "batches": 8},
         "memory": {"peak_le_85pct": True, "peak_over_request": 0.7}, "manifest_sha": "m"}
    J.update(kw); return J
FCP = {"donor_plan_ok": True, "donor_plan": {f"r{i}": {} for i in range(6)}, "floor_12k_reachable": True, "updates": 14000, "startup_info": {"extrapolated": False}}
RHP_ = {"pass": True, "why": "byte-equal"}
RTa = WP.retime_rules(_J(), FCP, 144, (True, "ok"), RHP_)
check("re-time: a complete passing record -> all 13 rules pass", RTa["all_pass"] and len(RTa["rules"]) == 13, RTa["failing"])
for name_, J_, fc_, sp_, rh_, rule_ in (
        ("timing on the 48-drive list", _J(train_drives=48), FCP, (True, ""), RHP_, "full_production_list_cold_allocation"),
        ("a warm (reused) stage", _J(stage={**_J()["stage"], "cold": False}), FCP, (True, ""), RHP_, "full_production_list_cold_allocation"),
        ("start-up not measured from allocation", _J(startup={"process_to_first_update_s": 800.0}), FCP, (True, ""), RHP_, "full_production_list_cold_allocation"),
        ("only 64 production updates (2 x 32)", _J(production_loop={"updates": 64, "blocks": _J()["production_loop"]["blocks"][:2]}), FCP, (True, ""), RHP_, "production_updates_ge_200"),
        ("one block with 6 % queue wait", _J(production_loop={"updates": 224, "blocks": _J()["production_loop"]["blocks"][:6] + [{"queue_frac": 0.06, "wall_over_gpu": 1.05}]}),
         FCP, (True, ""), RHP_, "queue_wait_le_5pct_each_block"),
        ("one block at wall 1.12 x gpu", _J(production_loop={"updates": 224, "blocks": _J()["production_loop"]["blocks"][:6] + [{"queue_frac": 0.01, "wall_over_gpu": 1.12}]}),
         FCP, (True, ""), RHP_, "wall_le_1.10_gpu_each_block"),
        ("no matched setting", _J(selected_setting=None), FCP, (True, ""), RHP_, "matched_update_selected"),
        ("update-0 sensitivity failed", _J(update0_sensitivity={"pass": False}), FCP, (True, ""), RHP_, "gates_final_recipe"),
        ("token geometry missing", _J(token_geometry=None), FCP, (True, ""), RHP_, "gates_final_recipe"),
        ("a staged hash mismatch", _J(stage={**_J()["stage"], "mismatches": ["x"]}), FCP, (True, ""), RHP_, "staged_hashes_eq_manifest"),
        ("7 of 8 batches equal", _J(staged_vs_lustre={"all_equal": False, "batches": 8}), FCP, (True, ""), RHP_, "staged_batches_byte_equal_lustre"),
        ("peak 0.9 of the request", _J(memory={"peak_le_85pct": False, "peak_over_request": 0.9}), FCP, (True, ""), RHP_, "peak_memory_le_85pct"),
        ("peak unreadable", _J(memory={"peak_le_85pct": None}), FCP, (True, ""), RHP_, "peak_memory_le_85pct"),
        ("no 2080 Ti scorer parity yet", _J(), FCP, (False, "no record"), RHP_, "scorer_parity_off_slice"),
        ("five valid donors", _J(), {**FCP, "donor_plan_ok": False}, (True, ""), RHP_, "six_valid_donors"),
        ("rehearsal not byte-equal", _J(), FCP, (True, ""), {"pass": False, "why": "RGB differs"}, "panel_rehearsal_byte_equal"),
        ("no rehearsal", _J(), FCP, (True, ""), None, "panel_rehearsal_byte_equal"),
        ("12,900 updates (floor met, < 13,200)", _J(), {**FCP, "updates": 12900}, (True, ""), RHP_, "forecast_ge_13200_one_panel")):
    rt_ = WP.retime_rules(J_, fc_, 144, sp_, rh_)
    check(f"re-time repro: {name_} -> rule {rule_} fails, overall FAIL", not rt_["all_pass"] and rule_ in rt_["failing"], rt_["failing"])
su, si_ = WP.startup_for_forecast(_J(), 144)
check("CONVERGED 7: start-up on the production list is used as measured from allocation (no 48 -> 144 extrapolation)",
      su == 900.0 and not si_["extrapolated"] and si_["from_allocation"], (su, si_))
su48, si48 = WP.startup_for_forecast(_J(train_drives=48), 144)
check("CONVERGED 7: a 48-drive timing scales ONLY the stage-in by drives and is flagged extrapolated",
      abs(su48 - (900 + 780 * 2)) < 1e-9 and si48["extrapolated"], (su48, si48))
TJok = {"selected_setting": {"micro": 1, "grad_ckpt": 0}, "retime_pass": {"all_pass": True}}
check("CONVERGED 7: overfit / train allowed only with the selected setting and an all-pass re-time", WP.retime_gate(TJok, 1, 0)[0])
for nm_, tj_, mc_, gc_ in (("micro mismatch", TJok, 8, 0), ("grad_ckpt mismatch", TJok, 1, 1),
                           ("re-time failed", {**TJok, "retime_pass": {"all_pass": False, "failing": ["x"]}}, 1, 0),
                           ("legacy timing.json (no selection)", {"retime_pass": {"all_pass": True}}, 1, 0), ("no retime_pass block", {"selected_setting": TJok["selected_setting"]}, 1, 0)):
    check(f"CONVERGED 7 repro: {nm_} -> overfit / resume test / train refused", not WP.retime_gate(tj_, mc_, gc_)[0])
check("CONVERGED 7: the trainer gates every non-timing mode on the re-time and the timing's gate key",
      "WP.retime_gate(TJ, a.micro, a.grad_ckpt)" in _trs and 'TJ.get("gate_key") != GATE_KEY' in _trs)

# panel cost: rendering only, one model load; scoring priced separately with its cap
check("F3: slice panel cost = ONE load + 5 x ticks x 2 seeds at the load-once rate (+ shared references)",
      abs(WP.panel_render_h(2092, 0.7, 381) - (381 + 5 * 2092 * 0.7) / 3600) < 1e-12
      and abs(WP.panel_render_h(2092, 0.7, 381, donors={"r": {"mode": "shared", "n": 100}, "s": {"mode": "full", "n": None}}) - (381 + (5 * 2092 + 200) * 0.7) / 3600) < 1e-12)
check("F3 repro: the measured 13.9 h panel (scoring on the slice, 5 loads) becomes ~2.1 h of slice time",
      abs(WP.panel_cost_h(2092, 0.70, 381, 2.41, 0.0) - 13.9) < 0.3 and WP.panel_render_h(2092, 0.70, 381) < 2.2)
check("F3: score GPU-h per panel = 7 scorings x ticks x rate; cap = 1.25 x", abs(WP.panel_score_h(2092, 3.0) - 7 * 2092 * 3 / 3600) < 1e-12
      and abs(WP.score_cap_h(2092, 3.0) - 1.25 * WP.panel_score_h(2092, 3.0)) < 1e-12)
_fcs = open("scripts/wan_forecast.py").read()
check("CONVERGED 1: wan_forecast --n_panels defaults to ONE; the two-panel number is recorded with 'no spare for a second read'",
      'p.add_argument("--n_panels", type=int, default=1' in _fcs and '"no spare for a second read"' in _fcs and "two_panels=two" in _fcs)
fc1 = WP.forecast_s1({**T_, "n_panels": 1}); fc2_ = WP.forecast_s1({**T_, "n_panels": 2})
check("CONVERGED 1: one panel in the 24 h buys more updates than two", fc1["updates"] > fc2_["updates"])

# off-slice scoring: parity record, binding, shard merge
with _tf.TemporaryDirectory() as td:
    B_ = {"a_metrics": "1", "a_roll": "2", "score_rgb": "3", "temporal_metrics": "4", "sd_vae": "5", "wan_score": "6"}
    good = {"pass": True, "replay_pass": True, "sharded_equal": True, "device": "NVIDIA GeForce RTX 2080 Ti", "binding": B_, "score_tick_s": 3.0}
    pr_ = f"{td}/p.json"
    json.dump(good, open(pr_, "w"))
    check("F3: bound 2080 Ti parity record accepted", WP.score_parity_ok(pr_, B_)[0])
    for nm_, rec_ in (("run on the H200 slice", {**good, "device": "NVIDIA H200 MIG 3g.71gb"}), ("sharded != unsharded", {**good, "sharded_equal": False}),
                      ("A@20k replay failed", {**good, "replay_pass": False}), ("scorer changed since", {**good, "binding": {**B_, "temporal_metrics": "X"}}),
                      ("wrapper changed since", {**good, "binding": {**B_, "wan_score": "X"}})):
        json.dump(rec_, open(pr_, "w"))
        check(f"F3 repro: parity record {nm_} -> MODE=train / panel decision refused", not WP.score_parity_ok(pr_, B_)[0])
    check("F3: no parity record -> refused", not WP.score_parity_ok(f"{td}/none.json", B_)[0])
PRr = {"device": "NVIDIA GeForce RTX 2080 Ti", "binding": {"score_rgb": "3", "temporal_metrics": "4", "sd_vae": "5"}}
bnd = lambda **kw: {"device": "NVIDIA GeForce RTX 2080 Ti", "score_rgb": "3", "temporal_metrics": "4", "sd_vae": "5", "rgb_manifest_sha": "M", **kw}
P1 = {"drives": {"a_s0": {"car_f1": 0.4}, "a_s1": {"car_f1": 0.5}}, "binding": bnd()}
P2 = {"drives": {"b_s0": {"car_f1": 0.3}}, "binding": bnd()}
mg, why_ = WP.merge_score_shards([("p1", P1), ("p2", P2)], "M", PRr)
check("F3: shards of THIS render on the parity device merge into one run set", not why_ and sorted(mg["drives"]) == ["a_s0", "a_s1", "b_s0"], why_)
for nm_, parts_ in (("a shard scored another render's RGB", [("p1", P1), ("p2", {**P2, "binding": bnd(rgb_manifest_sha="OLD")})]),
                    ("a shard scored on another GPU type", [("p1", P1), ("p2", {**P2, "binding": bnd(device="NVIDIA RTX A6000")})]),
                    ("a shard scored with another scorer", [("p1", P1), ("p2", {**P2, "binding": bnd(temporal_metrics="X")})]),
                    ("a run scored twice", [("p1", P1), ("p1b", P1)])):
    check(f"F3 repro: {nm_} -> the panel decision refuses", bool(WP.merge_score_shards(parts_, "M", PRr)[1]))
U_ = {"drives": {"a_s0": {"car_f1": 0.4, "placement_ticks": [0, 5]}, "b_s0": {"car_f1": float("nan"), "placement_ticks": [5]}},
      "binding": {"inputs": {"a_s0.pt": "x", "b_s0.pt": "y"}}}
S1_ = {"drives": {"a_s0": {"car_f1": 0.4, "placement_ticks": [0, 5]}}, "binding": {"inputs": {"a_s0.pt": "x"}}}
S2_ = {"drives": {"b_s0": {"car_f1": float("nan"), "placement_ticks": [5]}}, "binding": {"inputs": {"b_s0.pt": "y"}}}
check("F3: sharded == unsharded (NaN == NaN, same accepted ticks, same inputs)", WP.compare_sharded(U_, [S1_, S2_])[0])
check("F3 repro: a shard with different accepted ticks -> sharded != unsharded",
      not WP.compare_sharded(U_, [S1_, {**S2_, "drives": {"b_s0": {"car_f1": float("nan"), "placement_ticks": [5, 10]}}}])[0])
check("F3 repro: a missing shard / different input bytes -> not equal", not WP.compare_sharded(U_, [S1_])[0]
      and not WP.compare_sharded(U_, [S1_, {**S2_, "binding": {"inputs": {"b_s0.pt": "z"}}}])[0])
fl = [f"d/scene-000{i}_s{s}.pt" for i in range(6) for s in (0, 1)]
sh2 = [WP.shard_by_drive(fl, f"{i}/2") for i in range(2)]
check("F3: shards by drive partition the runs and keep both seeds of a drive together",
      sorted(sh2[0] + sh2[1]) == sorted(fl) and not set(sh2[0]) & set(sh2[1]) and all(f.replace("_s0", "_s1") in sh2[0] for f in sh2[0] if "_s0" in f))
_scr = open("scripts/score_rgb.py").read(); _wsc = open("scripts/wan_score.py").read(); _pan = open("scripts/wan_panel.py").read()
check("F3: score_rgb verifies every input against the slice's RGB manifest and records the binding",
      "RGB files differ from the slice's sha manifest" in _scr and 'OUTJ["binding"]' in _scr and "shard_by_drive(" in _scr)
check("F3: scoring has its own ledger file with a 1.25 x cap, refusal / termination at the cap",
      'default="runs/wan_score_ledger.jsonl"' in _wsc and "WP.score_cap_h(" in _wsc and "SCORE_CAP_EXCEEDED" in _wsc and "sys.exit(10)" in _wsc)
check("F3: wan_score.sh charges the 2080 Ti allocation to the score ledger, never the S1 slice ledger",
      "SL=runs/wan_score_ledger.jsonl" in open("slurm/wan_score.sh").read() and "wan_s1_ledger" not in open("slurm/wan_score.sh").read())
check("F3: the panel renders every run in ONE rollout process and the decision reads only shards of its RGB manifest",
      'plan_run(runs, f"{a.wroot}/{a.name}_plan"' in _pan and "WP.merge_score_shards(" in _pan and "score_parity_ok(PREC" in _pan
      and "temporal_compare.py" in _pan and '--phase", default=""' in _pan)
_rol = open("scripts/wan_rollout.py").read()
check("F3: load-once rollout resets per-run state (sensitivity, RNG) and uses a fresh solver per window",
      "SENS = None\n    torch.manual_seed(0)" in _rol and "sch = make_unipc()" in _rol and "SCHED" not in _rol and '"rgb_sha": hashlib.sha256(rgb.numpy().tobytes())' in _rol)
check("F3: rehearsal = load-once (after a different run) vs fresh-load byte-equality on RGB and SD latents",
      'roll_cmd(f"{a.out}/loop_tokens", "tokens"' in _pan and 'torch.equal(A_["rgb"], B_["rgb"])' in _pan and 'torch.equal(A_["lat"], B_["lat"])' in _pan)
print(f"WAN_TESTS FAILS={FAILS}")
sys.exit(1 if FAILS else 0)
