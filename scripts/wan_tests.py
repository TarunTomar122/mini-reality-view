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
    check(f"S1g 4a repro: {_name} refused", _r.returncode != 0 and ("overlapping run modes" in _msg or "bounded 16-update" in _msg), _msg[-160:])
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
      'cleanup() { stop_beat; kill $MEMS 2>/dev/null; [ -n "$STAGE" ] && rm -rf "$STAGE" "$GEOM"; }' in _tsh and "trap cleanup EXIT" in _tsh and "mrv.wan_stage memcheck" in _tsh and "exit 9" in _tsh)
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

# re-time rule (CONVERGED item 7 + Astra t1 items 1-9, 11)
CAN = {"train": [f"scene-{i:04d}" for i in range(144)], "dev": [f"scene-{i:04d}" for i in range(900, 906)]}
def _sw():
    return {f"gc{g}_m{m}": {"micro": m, "grad_ckpt": g, "gpu_s": 3.0 + m / 10, "fwd_s": 1.0} for g in (1, 0) for m in (1, 2, 4, 8)}
def _J(**kw):
    J = {"train_drives": 144, "dev_drives": 6, "train_ids_sha": WP.ids_sha(CAN["train"]), "dev_ids_sha": WP.ids_sha(CAN["dev"]),
         "batch": 8, "use_tokens": True, "sweep": _sw(), "evidence_sha": "E", "ckpt_sha16": "C" * 16, "job": "3760001",
         "stage": {"cold": True, "hash_equals_manifest": True, "mismatches": [], "total_files": 600, "total_bytes": 9e10},
         "startup": {"components": {k: 10.0 for k in WP.STARTUP_PARTS}},
         "production_loop": {"updates": 224, "block": 32, "micro": 1, "grad_ckpt": 0,
                             "blocks": WS.block_stats([{"wall_s": 4.0, "gpu_s": 3.9, "queue_wait_s": 0.01}] * 224, 32)},
         "selected_setting": {"key": "gc0_m1", "micro": 1, "grad_ckpt": 0},
         "matched_update": {"method": "per_tensor_adam_delta", "rows": {"gc0_m1": {"ok": True, "delta": {"ok": True}}}},
         "token_geometry": {"pass": True, "manifest_verified": True}, "init_parity": {"pass": True},
         "init_inertness": {"pass": True, "opt_step": 0}, "early_sensitivity": {"pass": True, "opt_step": WP.SENS_STEP, "micro": 1, "grad_ckpt": 0},
         "staged_vs_lustre": {"all_equal": True, "batches": 8},
         "memory": {"rule": {"pass": True, "ratio": 0.7}, "sampled_from_allocation": True}, "manifest_sha": "m"}
    J.update(kw); return J
FCP = {"donor_plan_ok": True, "donor_plan": {f"r{i}": {} for i in range(6)}, "floor_12k_reachable": True, "updates": 14000, "n_panels": 1,
       "panel_plan_complete": True, "panel_h_total": 2.1, "inputs": {"n_panels": 1, "budget_h": 24.0, "job_h": 12.0, "panel_h": 2.1},
       "startup_info": {"complete": True}, "ledger_evidence": {"ok": True, "timing_allocations": ["3756964", "3760001"]}, "evidence_sha": "E"}
RHP_ = {"pass": True, "why": "byte-equal", "fresh_all_ticks": True, "ckpt_sha": "C" * 16}
RTa = WP.retime_rules(_J(), FCP, CAN, (True, "ok"), RHP_)
check("re-time: a complete passing record -> all 17 rules pass", RTa["all_pass"] and len(RTa["rules"]) == 17, RTa["failing"])
L48 = {"train": CAN["train"][:48], "dev": CAN["dev"]}
J48 = _J(train_drives=48, train_ids_sha=WP.ids_sha(L48["train"]))
check("T1 item 3 repro (Astra): a 48-drive timing with a 48-drive forecast list -> exact_production_lists FAILS",
      "exact_production_lists" in WP.retime_rules(J48, FCP, CAN, (True, ""), RHP_)["failing"]
      and "exact_production_lists" in WP.retime_rules(J48, FCP, {**L48, "dev": CAN["dev"]}, (True, ""), RHP_)["failing"])
J100 = _J(production_loop={"updates": 200, "block": 100, "micro": 1, "grad_ckpt": 0,
                           "blocks": WS.block_stats([{"wall_s": 4.0, "gpu_s": 3.9, "queue_wait_s": 0.01}] * 200, 100)})
check("T1 item 4 repro (Astra): two 100-update blocks -> blocks_32_consecutive FAILS",
      "blocks_32_consecutive" in WP.retime_rules(J100, FCP, CAN, (True, ""), RHP_)["failing"])
FC0 = {**FCP, "n_panels": 0, "inputs": {**FCP["inputs"], "n_panels": 0}, "updates": 16000}
check("T1 item 5 repro (Astra): a zero-panel forecast with >= 13,200 updates -> forecast rule FAILS",
      "forecast_ge_13200_one_panel" in WP.retime_rules(_J(), FC0, CAN, (True, ""), RHP_)["failing"])
for name_, J_, fc_, sp_, rh_, rule_ in (
        ("T1 item 1: inertness not measured at optimizer step 0", _J(init_inertness={"pass": True, "opt_step": 250}), FCP, (True, ""), RHP_, "gates_final_recipe"),
        ("T1 item 1 repro: 'update-0' sensitivity read after 250 Adam steps", _J(early_sensitivity={"pass": True, "opt_step": 250, "micro": 1, "grad_ckpt": 0}), FCP, (True, ""), RHP_, "gates_final_recipe"),
        ("T1 item 1: early sensitivity on another setting than the selection", _J(early_sensitivity={"pass": True, "opt_step": WP.SENS_STEP, "micro": 8, "grad_ckpt": 1}), FCP, (True, ""), RHP_, "gates_final_recipe"),
        ("T1 item 2 repro: scalar-only matched check (no per-tensor Adam deltas)", _J(matched_update={"rows": {"gc0_m1": {"ok": True}}, "all_ok": True}), FCP, (True, ""), RHP_, "matched_update_selected"),
        ("T1 item 2: selected setting's per-tensor delta failed", _J(matched_update={"method": "per_tensor_adam_delta", "rows": {"gc0_m1": {"ok": True, "delta": {"ok": False}}}}), FCP, (True, ""), RHP_, "matched_update_selected"),
        ("T1 item 3: a duplicated ID padding a 143-drive list to 144", _J(train_ids_sha=WP.ids_sha(CAN["train"][:143] + CAN["train"][:1])), FCP, (True, ""), RHP_, "exact_production_lists"),
        ("T1 item 4: blocks not starting at the first production update", _J(production_loop={**_J()["production_loop"], "blocks": [{**b, "first": b["first"] + 5} for b in _J()["production_loop"]["blocks"]]}), FCP, (True, ""), RHP_, "blocks_32_consecutive"),
        ("T1 item 4: a sweep cell missing (gc0_m8 never measured, no OOM)", _J(sweep={k: v for k, v in _sw().items() if k != "gc0_m8"}), FCP, (True, ""), RHP_, "declared_recipe_and_sweep"),
        ("T1 item 4: global batch 4", _J(batch=4), FCP, (True, ""), RHP_, "declared_recipe_and_sweep"),
        ("T1 item 4: tokens off", _J(use_tokens=False), FCP, (True, ""), RHP_, "declared_recipe_and_sweep"),
        ("T1 item 4: production loop on another setting than the selection", _J(production_loop={**_J()["production_loop"], "micro": 8}), FCP, (True, ""), RHP_, "declared_recipe_and_sweep"),
        ("T1 item 5: budget 30 h", _J(), {**FCP, "inputs": {**FCP["inputs"], "budget_h": 30.0}}, (True, ""), RHP_, "forecast_ge_13200_one_panel"),
        ("T1 item 5: 24 h jobs", _J(), {**FCP, "inputs": {**FCP["inputs"], "job_h": 24.0}}, (True, ""), RHP_, "forecast_ge_13200_one_panel"),
        ("T1 item 5: no ledger evidence for both timing allocations", _J(), {**FCP, "ledger_evidence": {"ok": False}}, (True, ""), RHP_, "forecast_ge_13200_one_panel"),
        ("T1 item 5: panel cost unmeasured", _J(), {**FCP, "panel_h_total": None}, (True, ""), RHP_, "forecast_ge_13200_one_panel"),
        ("T1 item 6 repro: a rehearsal of ANOTHER checkpoint", _J(), FCP, (True, ""), {**RHP_, "ckpt_sha": "D" * 16}, "panel_rehearsal_byte_equal"),
        ("T1 item 6 repro: a rehearsal that reused existing RGB", _J(), FCP, (True, ""), {**RHP_, "fresh_all_ticks": False}, "panel_rehearsal_byte_equal"),
        ("T1 item 7: start-up components incomplete (no frozen snapshot / guard)", _J(startup={"components": {"pre_trainer_s": 1.0, "process_to_ready_s": 800.0}}), FCP, (True, ""), RHP_, "cold_allocation_startup_measured"),
        ("T1 item 7: start-up not from the allocation", _J(startup={"components": {**{k: 10.0 for k in WP.STARTUP_PARTS}, "pre_trainer_s": None}}), FCP, (True, ""), RHP_, "cold_allocation_startup_measured"),
        ("T1 item 8: peak not sampled from the allocation start", _J(memory={"rule": {"pass": True, "ratio": 0.7}, "sampled_from_allocation": False}), FCP, (True, ""), RHP_, "peak_memory_le_85pct_of_request"),
        ("T1 item 8: peak over 85 % of the request", _J(memory={"rule": {"pass": False, "ratio": 0.97}, "sampled_from_allocation": True}), FCP, (True, ""), RHP_, "peak_memory_le_85pct_of_request"),
        ("T1 item 9: geometry read unhashed inputs", _J(token_geometry={"pass": True}), FCP, (True, ""), RHP_, "gates_final_recipe"),
        ("T1 item 11: timing and forecast produced by different evidence code", _J(evidence_sha="E2"), FCP, (True, ""), RHP_, "evidence_code_identity"),
        ("one block with 6 % queue wait", _J(production_loop={**_J()["production_loop"], "blocks": _J()["production_loop"]["blocks"][:6] + [{**_J()["production_loop"]["blocks"][6], "queue_frac": 0.06}]}),
         FCP, (True, ""), RHP_, "queue_wait_le_5pct_each_block"),
        ("one block at wall 1.12 x gpu", _J(production_loop={**_J()["production_loop"], "blocks": _J()["production_loop"]["blocks"][:6] + [{**_J()["production_loop"]["blocks"][6], "wall_over_gpu": 1.12}]}),
         FCP, (True, ""), RHP_, "wall_le_1.10_gpu_each_block"),
        ("a warm (reused) stage", _J(stage={**_J()["stage"], "cold": False}), FCP, (True, ""), RHP_, "cold_allocation_startup_measured"),
        ("only 64 production updates", _J(production_loop={**_J()["production_loop"], "updates": 64, "blocks": _J()["production_loop"]["blocks"][:2]}), FCP, (True, ""), RHP_, "production_updates_ge_200"),
        ("a staged hash mismatch", _J(stage={**_J()["stage"], "mismatches": ["x"]}), FCP, (True, ""), RHP_, "staged_hashes_eq_manifest"),
        ("7 of 8 batches equal", _J(staged_vs_lustre={"all_equal": False, "batches": 8}), FCP, (True, ""), RHP_, "staged_batches_byte_equal_lustre"),
        ("no 2080 Ti scorer parity yet", _J(), FCP, (False, "no record"), RHP_, "scorer_parity_off_slice"),
        ("five valid donors", _J(), {**FCP, "donor_plan_ok": False}, (True, ""), RHP_, "six_valid_donors"),
        ("rehearsal not byte-equal", _J(), FCP, (True, ""), {**RHP_, "pass": False}, "panel_rehearsal_byte_equal"),
        ("12,900 updates (floor met, < 13,200)", _J(), {**FCP, "updates": 12900}, (True, ""), RHP_, "forecast_ge_13200_one_panel")):
    rt_ = WP.retime_rules(J_, fc_, CAN, sp_, rh_)
    check(f"re-time repro: {name_} -> rule {rule_} fails, overall FAIL", not rt_["all_pass"] and rule_ in rt_["failing"], rt_["failing"])
# item 1: init inertness / early sensitivity definitions
check("T1 item 1: init inert = guide exactly 0 and tokens within 1e-3 |v| at optimizer step 0",
      WP.init_inert({"trained": {"finite": True, "guide": 0.0, "tokens": 1e-6, "v_abs": 1.0}})
      and not WP.init_inert({"trained": {"finite": True, "guide": 1e-5, "tokens": 0.0, "v_abs": 1.0}})
      and not WP.init_inert({"trained": {"finite": True, "guide": 0.0, "tokens": 0.01, "v_abs": 1.0}}))
check("T1 item 1: timing probes the untouched init BEFORE any update, restores init weights + a fresh optimizer, then requires "
      "check_sensitivity after exactly SENS_STEP optimizer steps", _trs.index("S0 = sens_probe()") < _trs.index("for gc in GCS:")
      and "model.load_trainable(INIT_W); opt.load_state_dict(copy.deepcopy(INIT_OPT))" in _trs and "prec = probe(WP.SENS_STEP)" in _trs
      and '"opt_step": opt_step(), "required_step": WP.SENS_STEP' in _trs and "update0_sensitivity" not in _trs)
# item 2: per-tensor Adam deltas
_rows = {"a": {"ref": 1.0, "diff": 0.01, "norm": 1.0}, "b": {"ref": 0.5, "diff": 0.001, "norm": 0.5}, "z": {"ref": 0.0, "diff": 0.0, "norm": 0.0}}
check("T1 item 2: matching per-tensor Adam deltas pass", WP.delta_check(_rows)["ok"])
check("T1 item 2 repro: a REVERSED update on one tensor (same loss / grad norm) fails", not WP.delta_check({**_rows, "b": {"ref": 0.5, "diff": 1.0, "norm": 0.5}})["ok"])
check("T1 item 2 repro: a PERMUTED update (same norm, 30 % different) fails", not WP.delta_check({**_rows, "a": {"ref": 1.0, "diff": 0.3, "norm": 1.0}})["ok"])
check("T1 item 2: a moved tensor whose reference did not move fails", not WP.delta_check({**_rows, "z": {"ref": 0.0, "diff": 0.1, "norm": 0.1}})["ok"])
_mu2 = WP.matched_updates({"gc1_m1": {"loss": 0.5, "grad_norm": 1.0}, "gc0_m1": {"loss": 0.5, "grad_norm": 1.0}}, "gc1_m1")
_mu2["rows"]["gc0_m1"]["ok"] = _mu2["rows"]["gc0_m1"]["ok"] and False  # delta failed
check("T1 item 2: identical loss / grad norm but a failed delta -> not selectable",
      WP.select_setting({"gc1_m1": {"micro": 1, "grad_ckpt": 1, "gpu_s": 3.6}, "gc0_m1": {"micro": 1, "grad_ckpt": 0, "gpu_s": 2.8}}, _mu2)["key"] == "gc1_m1")
check("T1 item 2: timing compares per-tensor deltas after ONE Adam step from restored weights + optimizer state, releases the batch",
      "opt.load_state_dict(copy.deepcopy(SNAP_OPT))" in _trs and "WP.delta_check(rows_t)" in _trs and "del BFIX, DREF, SNAP_W, SNAP_OPT" in _trs)
# item 3: list identity
check("T1 item 3: exact canonical 144 + 6 accepted", WP.check_lists(CAN["train"], CAN["dev"], CAN, "full")[0])
for nm_, tr_, dv_, md_ in (("48 train drives (full mode)", CAN["train"][:48], CAN["dev"], "full"),
                           ("143 + one duplicate", CAN["train"][:143] + CAN["train"][:1], CAN["dev"], "full"),
                           ("a dev ID in the train list", CAN["train"][:143] + CAN["dev"][:1], CAN["dev"][1:], "full"),
                           ("a gate subset with a dev drive among train", CAN["train"][:47] + CAN["dev"][:1], CAN["dev"], "gate")):
    check(f"T1 item 3 repro: {nm_} -> refused", not WP.check_lists(tr_, dv_, CAN, md_)[0])
check("T1 item 3: a gate subset inside its original splits is allowed", WP.check_lists(CAN["train"][:48], CAN["dev"], CAN, "gate")[0])
check("T1 item 3: the trainer checks the lists against data/temporal_lists.json and the manifest split",
      'WP.check_lists(list(LISTS["train"]), list(LISTS["dev"]), CANON' in _trs and '(MAN.get("split") or {}).get("train") != sorted(CANON["train"])' in _trs)
# item 4: sweep evidence
_swo = {**_sw(), "gc0_m4": {"micro": 4, "grad_ckpt": 0, "oom": "CUDA out of memory"}, "gc0_m8": {"micro": 8, "grad_ckpt": 0, "skipped_after_oom": "gc0_m4"}}
check("T1 item 4: an OOM row with the larger micro marked skipped-after-OOM is complete evidence", WP.sweep_complete(_swo)[0])
check("T1 item 4 repro: 'skipped' without an OOM at a smaller micro is not evidence",
      not WP.sweep_complete({**_sw(), "gc0_m8": {"micro": 8, "grad_ckpt": 0, "skipped_after_oom": "gc1_m4"}})[0])
check("T1 item 4: blocks_ok on 7 x 32 consecutive", WP.blocks_ok(_J()["production_loop"])[0])
# item 5: forecast inputs
check("T1 item 5: ledger evidence needs both timing allocations, this job included",
      WP.ledger_evidence({"alloc-3756964-1-1": {"kind": "alloc_timing"}, "alloc-3760001-2-2": {"kind": "alloc_timing"}}, "3760001")["ok"]
      and not WP.ledger_evidence({"alloc-3760001-2-2": {"kind": "alloc_timing"}}, "3760001")["ok"]
      and not WP.ledger_evidence({"alloc-3756964-1-1": {"kind": "alloc_timing"}, "alloc-3760001-2-2": {"kind": "alloc_timing"}}, "3770000")["ok"])
_fsrc = open("scripts/wan_forecast.py").read()
check("T1 item 5: a missing ledger refuses the forecast (never zero spend)", "a missing ledger is never zero spend" in _fsrc and "WP.ledger_evidence(" in _fsrc)
# item 6
check("T1 item 6: plan_fresh catches a skipped (reused) run", WP.plan_fresh([{"ticks_rendered": 174}, {"ticks_rendered": 0}], [174, 174])[0] is False
      and WP.plan_fresh([{"ticks_rendered": 174}], [174])[0])
_psrc = open("scripts/wan_panel.py").read()
check("T1 item 6: rehearsal needs an empty folder, fresh ticks in both paths and the timing checkpoint's digest; render refuses reuse",
      "a rehearsal must render fresh" in _psrc and "WP.plan_fresh(PF[\"runs\"], [n_r])" in _psrc and "CK_SHA16" in _psrc and "panel not rendered fresh" in _psrc)
check("T1 item 6: a failed rehearsal fails the timing job", '[ $REH -ne 0 ] && { echo "REHEARSAL FAILED' in _tsh)
# item 7
su, si_ = WP.startup_for_forecast(_J())
check("T1 item 7: start-up = sum of the selected recipe's measured production start components", su == 10.0 * len(WP.STARTUP_PARTS) and si_["complete"], (su, si_))
check("T1 item 7: incomplete components -> no start-up (the forecast refuses)", WP.startup_for_forecast(_J(startup={"components": {"pre_trainer_s": 5.0}}))[0] is None)
check("T1 item 7: timing measures frozen snapshot / first update incl. Adam / guard verify with production's own functions; geometry excluded",
      "snap = frozen_snapshot()" in _trs and "guard_verify(0, snap)" in _trs and "SNAP0 = frozen_snapshot()" in _trs and "guard_verify(step, SNAP0)" in _trs
      and "MRV_TIMING_ONLY_S" in _trs and "export MRV_TIMING_ONLY_S" in _tsh)
# item 8
check("T1 item 8: Slurm request from SLURM_MEM_PER_NODE / per-CPU / scontrol", WS.slurm_requested_bytes({"SLURM_MEM_PER_NODE": "148480"}) == 145 * (1 << 30)
      and WS.slurm_requested_bytes({"SLURM_MEM_PER_CPU": "1024", "SLURM_CPUS_ON_NODE": "8"}) == 8 * (1 << 30)
      and WS.slurm_requested_bytes({}, "JobId=1 MinMemoryNode=145G") == 145 * (1 << 30) and WS.slurm_requested_bytes({}, "") is None)
mr_ = WP.memory_rule([140e9, None, 120e9], 145 * (1 << 30) * 0 + 145e9)
check("T1 item 8 repro: 140 GB peak under a 2 TB ancestor limit but a 145 GB request -> FAIL (ratio over the request)",
      WS.peak_rule(140e9, 2e12)[0] is True and mr_["pass"] is False and mr_["ratio"] > 0.85, mr_)
check("T1 item 8: the largest of the allocation sampler, memory.peak and the trainer sampler is used; no request -> unknown",
      WP.memory_rule([100e9, 120e9, None], 145e9)["peak"] == 120e9 and WP.memory_rule([100e9], None)["pass"] is None)
with _tf.TemporaryDirectory() as td:
    check("T1 item 8: job-level cgroup = the job_<id> ancestor", WS.job_cgroup_dir([("v2", "/sys/fs/cgroup/system.slice/slurmstepd.scope/job_77/step_batch/user/task_0")])
          == ("v2", "/sys/fs/cgroup/system.slice/slurmstepd.scope/job_77"))
check("T1 item 8: slurm starts the job-level sampler at allocation start, before staging; trainer divides by the request",
      _tsh.index("mrv.wan_stage sample") < _tsh.index("run_fg $T") and "MEM_REQUEST = WS.slurm_requested_bytes()" in _trs
      and "WP.memory_rule([AS.get(\"peak\"), AS.get(\"max_current\"), CG[\"peak\"], SAMP], MEM_REQUEST)" in _trs)
# item 9: reused staged files are rehashed
with _tf.TemporaryDirectory() as td:
    R9, W9 = _fixture(td)
    mp9 = f"{td}/m/manifest.json"; os.makedirs(f"{td}/m"); _vm(td, R9, W9, mp9); man9, msha9 = WS.load_manifest(mp9)
    roots9 = {k: WS.norm(v) for k, v in R9.items()}
    st9 = f"{td}/stage"; os.makedirs(st9)
    jobs9, _ = WS.stage_plan(man9, ["scene-0001"], roots9, st9)
    r1_ = WS.stage_reuse(jobs9, f"{st9}/STAGED.json", "J1", msha9)
    tgt = [j for j in jobs9 if j[1].endswith("tokens/scene-0001.pt")][0][1]
    b9 = bytearray(open(tgt, "rb").read()); b9[3] ^= 1; open(tgt, "wb").write(bytes(b9))  # same-size edit between resume legs
    r2_ = WS.stage_reuse(jobs9, f"{st9}/STAGED.json", "J1", msha9)
    check("T1 item 9 repro: a same-size edit of a staged file between legs is re-hashed, detected and recopied",
          r1_["cold"] and r2_["rehashed_files"] == len(jobs9) and r2_["recopied_after_rehash"] == 1 and r2_["reused_files"] == len(jobs9) - 1
          and all(WS.sha256_file(j[1]) == j[2] for j in jobs9), (r2_["rehashed_files"], r2_["recopied_after_rehash"]))
    r3_ = WS.stage_reuse(jobs9, f"{st9}/STAGED.json", "J2", msha9)
    check("T1 item 9: another job id never reuses (full copy + hash)", r3_["reused_files"] == 0 and r3_["files"] == len(jobs9))
    _r = _sp.run([sys.executable, "-m", "mrv.wan_stage", "copy", "--manifest", mp9, "--dst", f"{td}/geom", f"{R9['guides']}/scene-0001.pt", f"{R9['tokens']}/scene-0001.pt"],
                 capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    check("T1 item 9: geometry inputs copied + hashed against the manifest", _r.returncode == 0 and os.path.exists(f"{td}/geom/wan_tok_v1/scene-0001.pt"), _r.stderr[-200:])
    b9 = bytearray(open(f"{R9['guides']}/scene-0002.pt", "rb").read()); b9[0] ^= 1; open(f"{R9['guides']}/scene-0002.pt", "wb").write(bytes(b9))
    _r = _sp.run([sys.executable, "-m", "mrv.wan_stage", "copy", "--manifest", mp9, "--dst", f"{td}/geom2", f"{R9['guides']}/scene-0002.pt"],
                 capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    check("T1 item 9 repro: a geometry guide file that differs from the manifest is refused", _r.returncode != 0)
check("T1 item 9: the trainer stages through stage_reuse (re-hash) and the geometry check reads the hashed copies",
      "WS.stage_reuse(_jobs, _RECEIPT" in _trs and "--guides $GEOM/guides_v1 --tokens $GEOM/wan_tok_v1" in _tsh)
# item 11 (gate): evidence identity + refreshed forecast
TJok = {"selected_setting": {"micro": 1, "grad_ckpt": 0}, "retime_pass": {"all_pass": True}, "evidence_sha": "E",
        "forecast_final": {"evidence_sha": "E", "inputs": {**T_, "n_panels": 1, "done_kinds": []}}}
check("CONVERGED 7: overfit / train allowed only with the selected setting and an all-pass re-time", WP.retime_gate(TJok, 1, 0, evidence="E")[0])
for nm_, tj_, mc_, gc_ in (("micro mismatch", TJok, 8, 0), ("grad_ckpt mismatch", TJok, 1, 1),
                           ("re-time failed", {**TJok, "retime_pass": {"all_pass": False, "failing": ["x"]}}, 1, 0),
                           ("legacy timing.json (no selection)", {"retime_pass": {"all_pass": True}}, 1, 0), ("no retime_pass block", {"selected_setting": TJok["selected_setting"]}, 1, 0)):
    check(f"CONVERGED 7 repro: {nm_} -> overfit / resume test / train refused", not WP.retime_gate(tj_, mc_, gc_)[0])
check("CONVERGED 7: the trainer gates every non-timing mode on the re-time and the timing's gate key",
      "WP.retime_gate(TJ, a.micro, a.grad_ckpt" in _trs and 'TJ.get("gate_key") != GATE_KEY' in _trs)

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
# panel intake (CONVERGED item 4; Astra t1 items 12-14)
WT = {"sd": "s", "m2f": "m", "dino": "d", "raft": "r"}
PRr = {"device": "NVIDIA GeForce RTX 2080 Ti", "binding": {"score_rgb": "3", "temporal_metrics": "4", "sd_vae": "5", "wan_score": "6", "weights": WT}}
MANX = {"dirs": {c: f"runs/wroll/x_{c}_dev" for c in ("hybrid", "guide", "tokens", "none")} | {"swap": "runs/wroll/x_swap_dev"},
        "runs": {f"runs/wroll/x_{c}_dev/{d}_s{s}.pt": {"sha256": f"{c}{d}{s}", "ticks": 100} for c in ("hybrid", "guide", "tokens", "none", "swap")
                 for d in ("scene-a", "scene-b") for s in (0, 1)}}
def bnd(c="hybrid", i=0, n=2, **kw):
    fl = WP.SCORE_FLAGS.get(c, (False, False, False)); src = MANX["dirs"]["swap" if c.startswith("swap") else c]
    fs = WP.shard_by_drive([f for f in MANX["runs"] if os.path.dirname(f) == src], f"{i}/{n}")
    b = {"device": "NVIDIA GeForce RTX 2080 Ti", "score_rgb": "3", "temporal_metrics": "4", "sd_vae": "5", "wan_score": "6", "weights": WT,
         "rgb_manifest_sha": "M", "src": src, "shard": f"{i}/{n}", "inputs": {os.path.basename(f): MANX["runs"][f]["sha256"] for f in fs},
         "post_swap_only": fl[0], "as_receiver": fl[1], "lag_ref": "runs/troll/A20k_indep_dev" if fl[2] else None,
         "lag_ref_inputs": {"r.pt": "L"} if fl[2] else None}
    b.update(kw); return b
def part(c="hybrid", i=0, n=2, drives=None, **kw):
    return (f"m_x_{c}_dev.shard{i}of{n}.json", {"drives": drives or {f"scene-{'ab'[i]}_s0": {"car_f1": 0.4}, f"scene-{'ab'[i]}_s1": {"car_f1": 0.5}},
                                               "binding": bnd(c, i, n, **kw)}, i, n)
RC = {f"m_x_{c}_dev.shard{i}of2.json": {"rc": 0, "over_cap": False} for c in WP.SCORE_FLAGS for i in (0, 1)}
mg, why_ = WP.merge_score_shards([part(), part(i=1)], "hybrid", MANX, "M", PRr, RC)
check("F3: complete, bound shards of THIS render merge into one run set", not why_ and len(mg["drives"]) == 4, why_)
mgs, whys = WP.merge_score_shards([part("swap_full"), part("swap_full", 1)], "swap_full", MANX, "M", PRr, RC, {"r.pt": "L"})
check("T1 item 13: a swap_full shard with the lag reference and the signed reference files merges", not whys, whys)
hyb0 = part("hybrid", 0)
for nm_, parts_, what_, lag_ in (
        ("F3: a shard scored another render's RGB", [part(rgb_manifest_sha="OLD"), part(i=1)], "hybrid", None),
        ("F3: a shard scored on another GPU type", [part(device="NVIDIA RTX A6000"), part(i=1)], "hybrid", None),
        ("F3: a shard scored with another scorer", [part(temporal_metrics="X"), part(i=1)], "hybrid", None),
        ("T1 item 12 repro: a shard scored with other Mask2Former weights", [part(weights={**WT, "m2f": "X"}), part(i=1)], "hybrid", None),
        ("T1 item 12 repro: a shard scored by another wrapper version", [part(wan_score="X"), part(i=1)], "hybrid", None),
        ("T1 item 13 repro (Astra): the HYBRID shard copied under the guide filename", [(f"m_x_guide_dev.shard0of2.json", hyb0[1], 0, 2), part("guide", 1)], "guide", None),
        ("T1 item 13: a shard that scored only part of its partition", [part(inputs={"scene-a_s0.pt": "hybridscene-a0"}), part(i=1)], "hybrid", None),
        ("T1 item 13: a shard file claiming another partition", [part(i=0, shard="1/2"), part(i=1)], "hybrid", None),
        ("T1 item 13: a post-donor scoring run without --post_swap_only", [part("swap_postdonor", post_swap_only=False), part("swap_postdonor", 1)], "swap_postdonor", None),
        ("T1 item 13: a post-receiver scoring run without --as_receiver", [part("swap_postrecv", as_receiver=False), part("swap_postrecv", 1)], "swap_postrecv", None),
        ("T1 item 13: swap_full scored without the lag reference", [part("swap_full", lag_ref=None), part("swap_full", 1)], "swap_full", {"r.pt": "L"}),
        ("T1 item 13: swap_full against other lag reference files", [part("swap_full"), part("swap_full", 1)], "swap_full", {"r.pt": "OTHER"}),
        ("T1 item 13: a missing shard (folder not covered)", [part()], "hybrid", None),
        ("F3: a run scored twice", [part(), part(i=1, drives={"scene-a_s0": {"car_f1": 0.4}})], "hybrid", None)):
    check(f"{nm_} -> the panel decision refuses", bool(WP.merge_score_shards(parts_, what_, MANX, "M", PRr, RC, lag_)[1]))
for nm_, rc_ in (("T1 item 14 repro: a shard whose receipt says it finished OVER the cap", {**RC, "m_x_hybrid_dev.shard1of2.json": {"rc": 0, "over_cap": True}}),
                 ("T1 item 12: a shard with a failed receipt", {**RC, "m_x_hybrid_dev.shard1of2.json": {"rc": 1}}),
                 ("T1 item 12: a shard without a receipt", {k: v for k, v in RC.items() if k != "m_x_hybrid_dev.shard1of2.json"})):
    check(f"{nm_} -> refused", bool(WP.merge_score_shards([part(), part(i=1)], "hybrid", MANX, "M", PRr, rc_)[1]))
# item 12: parity parts bound at execution
CUR = {"score_rgb": "3", "temporal_metrics": "4", "sd_vae": "5", "wan_score": "6", "weights": WT}
AR_ = {"a_s0.pt": "1", "a_s1.pt": "2", "b_s0.pt": "3", "b_s1.pt": "4"}
pb = lambda shard, ins, **kw: {"binding": {**CUR, "shard": shard, "inputs": ins, "parity_ref_sha": "SIG" if shard == "0/1" else None, **kw}}
PP = [("u", pb("0/1", AR_), {"rc": 0}), ("s0", pb("0/2", {"a_s0.pt": "1", "a_s1.pt": "2"}), {"rc": 0}), ("s1", pb("1/2", {"b_s0.pt": "3", "b_s1.pt": "4"}), {"rc": 0})]
check("T1 item 12: parity parts bound at execution to the current scorer / weights / references pass", WP.parity_parts_ok(PP, CUR, AR_, "SIG")[0])
for nm_, pp_, ar_, sig_ in (("a shard receipt failed (rc 1)", [PP[0], PP[1], ("s1", PP[2][1], {"rc": 1})], AR_, "SIG"),
                            ("scores made with weights that changed since (current stamped over old)", [PP[0], PP[1], ("s1", pb("1/2", {"b_s0.pt": "3", "b_s1.pt": "4"}, weights={**WT, "raft": "OLD"}), {"rc": 0})], AR_, "SIG"),
                            ("reference rollouts changed after scoring", PP, {**AR_, "b_s1.pt": "NEW"}, "SIG"),
                            ("unsharded replay against another signed metrics file", PP, AR_, "OTHER"),
                            ("wrapper changed after scoring", [("u", pb("0/1", AR_, wan_score="OLD"), {"rc": 0}), PP[1], PP[2]], AR_, "SIG")):
    check(f"T1 item 12 repro: {nm_} -> parity check FAILS", not WP.parity_parts_ok(pp_, CUR, ar_, sig_)[0])
_wl = WP.metric_weights()
check("T1 item 12: metric weights cover every model the signed scorer loads (SD VAE, Mask2Former, DINOv2, RAFT, AlexNet, LPIPS)",
      set(_wl) == set(WP.METRIC_HF) | set(WP.METRIC_HUB) | {"lpips_alex"})
# item 14: aggregate headroom across live workers
now_ = time.time()
JB = {"alloc-101-1-1": {"kind": "score_x", "t0": now_ - 1800, "t": now_ - 10, "ended": False, "beat_s": 60},   # live, 0.5 h in
      "alloc-102-1-1": {"kind": "score_x", "t0": now_ - 7200, "t": now_ - 3600, "ended": True, "beat_s": 60},  # finished
      "alloc-103-1-1": {"kind": "score_x", "t0": now_ - 9000, "t": now_ - 8000, "ended": False, "beat_s": 60}}  # dead
RV = [{"job": "101", "est_h": 2.0}, {"job": "102", "est_h": 1.0}, {"job": "103", "est_h": 3.0}, {"job": "104", "est_h": 1.0}]
ar_h = WP.active_reserved_h(RV, JB, "score_x", now_)
check("T1 item 14: only LIVE workers' remaining reservations count (2.0 - 0.5 h)", abs(ar_h - 1.5) < 1e-3, ar_h)
check("T1 item 14 repro: spent 3 h + live reservations 1.5 h + this shard 1 h > cap 5 h -> refused before scoring",
      not WP.score_headroom(3.0, ar_h, 1.0, 5.0)[0] and WP.score_headroom(3.0, ar_h, 0.4, 5.0)[0])
_wsc = open("scripts/wan_score.py").read(); _pan = open("scripts/wan_panel.py").read()
check("T1 item 14: wan_score reserves under a cross-node lock, re-checks after finishing and sets an over-cap output aside",
      "os.mkdir(LOCK)" in _wsc and "WP.score_headroom(" in _wsc and 'os.replace(out, out + ".overcap")' in _wsc and '"over_cap": over' in _wsc)
check("T1 item 14: the decision refuses publication on a missing or over-cap scoring ledger", "no publication on a missing or over-cap scoring ledger" in _pan)
check("T1 item 12: score_rgb binds wrapper / weights / signed reference / lag reference files at execution; parity_check verifies them",
      '"weights": metric_weights()' in _scr if (_scr := open("scripts/score_rgb.py").read()) else False)
check("T1 item 12: parity_check uses parity_parts_ok (receipts + execution bindings)", "WP.parity_parts_ok(" in _wsc)
# item 15: checkpoint / donor plan digests
MR15 = {"ckpt_sha256": "A" * 64, "donors_sha256": "D" * 64, "donor_plan": {"donors": {}}, "runs": {"f": {"ckpt_sha": "A" * 16}}}
check("T1 item 15: digests recorded at render and unchanged -> decision allowed", WP.render_binding_ok(MR15, "A" * 64, "D" * 64)[0])
for nm_, args_ in (("the rolling checkpoint changed between render and decision", (MR15, "B" * 64, "D" * 64)),
                   ("the donor plan was regenerated", (MR15, "A" * 64, "E" * 64)),
                   ("a rendered file from another checkpoint", ({**MR15, "runs": {"f": {"ckpt_sha": "B" * 16}}}, "A" * 64, "D" * 64)),
                   ("no donor plan stored at render", ({**MR15, "donor_plan": None}, "A" * 64, "D" * 64))):
    check(f"T1 item 15 repro: {nm_} -> decision refused", not WP.render_binding_ok(*args_)[0])
check("T1 item 15: render stores ckpt / donor digests + the plan; decide verifies and uses the stored plan",
      '"ckpt_sha256": CK_SHA256, "donors_sha256": WP._sha_file(DJ), "donor_plan": PLAN' in _pan and 'DON = M["donor_plan"]["donors"]' in _pan
      and "WP.render_binding_ok(" in _pan)
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

# ---- Astra t1 item 10: overlapping modes / unsigned overfit gate
for _flags, _ok in (((True, 8, False, False), False), ((False, 8, True, False), False), ((True, 0, False, True), False),
                    ((True, 0, False, False), True), ((False, 8, False, False), True), ((False, 0, False, False), True)):
    check(f"T1 item 10 repro: modes timing/overfit/resume/production = {_flags} -> {'accepted' if _ok else 'refused'}", (not _refused(WP.check_modes, *_flags)) == _ok)
check("T1 item 10: the signed overfit gate is accepted", WP.overfit_signed_ok(8, 2000, 0.25, 0.8)[0])
for _a, _nm in (((4, 2000, 0.25, 0.8), "4 clips"), ((8, 1000, 0.25, 0.8), "1,000 updates"), ((8, 2000, 0.5, 0.8), "loss ratio 0.5"),
                ((8, 2000, 0.25, 0.5), "F1 fraction 0.5"), ((8, 2000, 0.25, 0.8, 0.0), "none margin 0")):
    check(f"T1 item 10 repro: overfit gate with {_nm} -> no verdict can be registered", not WP.overfit_signed_ok(*_a)[0])
_trs = open("scripts/train_wan_teacher.py").read()
check("T1 item 10: the trainer rejects overlapping modes first and re-checks the signed gate before gate_append",
      _trs.index("MODE_ = WP.check_modes(") < _trs.index("TEST = a.timing or a.overfit") and "a PASS is registered only under the signed" in _trs)
# ---- Astra t1 item 11: evidence identity + refreshed forecast against the actual ledger
with _tf.TemporaryDirectory() as td:
    for f_ in WP.EVIDENCE_FILES:
        os.makedirs(os.path.dirname(f"{td}/{f_}"), exist_ok=True); _sh_.copyfile(f_, f"{td}/{f_}")
    e0 = WP.evidence_sha(td)
    open(f"{td}/mrv/wan_stage.py", "a").write("\n# edit\n"); e1 = WP.evidence_sha(td)
    open(f"{td}/scripts/wan_forecast.py", "a").write("\n# edit\n"); e2 = WP.evidence_sha(td)
check("T1 item 11 repro: staging / forecast code edits change the evidence identity (they were outside the timing identity)", len({e0, e1, e2}) == 3)
TJe = {"selected_setting": {"micro": 1, "grad_ckpt": 0}, "retime_pass": {"all_pass": True}, "evidence_sha": "E",
       "forecast_final": {"evidence_sha": "E", "inputs": {**T_, "budget_h": 24.0, "used_h": 1.0, "n_panels": 1, "panel_h": 2.0, "wall_s": 2.0, "done_kinds": []}}}
check("T1 item 11 repro: re-time evidence from older staging / forecast code -> overfit refused", not WP.retime_gate(TJe, 1, 0, evidence="E2")[0])
ok_a, r_a = WP.refresh_forecast(TJe["forecast_final"], 1.0, set())
ok_b, r_b = WP.refresh_forecast(TJe["forecast_final"], 15.0, set())
check("T1 item 11 repro: a stored PASS whose headroom was consumed (ledger 1 h -> 15 h) is refused at the next stage",
      ok_a and not ok_b and WP.retime_gate(TJe, 1, 0, evidence="E", used_h=1.0)[0] and not WP.retime_gate(TJe, 1, 0, evidence="E", used_h=15.0)[0], (r_a["updates"], r_b["updates"]))
ok_c, r_c = WP.refresh_forecast(TJe["forecast_final"], 15.0, set(), start_step=9000)
check("T1 item 11: a production resume counts the updates already done (start + affordable >= 12k)", ok_c == (9000 + r_c["updates"] >= 12000))
check("T1 item 11: the trainer refreshes the forecast with the live ledger and gate registry before every non-timing stage",
      "WP.retime_gate(TJ, a.micro, a.grad_ckpt, evidence=EVIDENCE_SHA, used_h=LEDGER.hours(), done_kinds=_done" in _trs)
print(f"WAN_TESTS FAILS={FAILS}")
sys.exit(1 if FAILS else 0)
