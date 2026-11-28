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

print(f"WAN_TESTS FAILS={FAILS}")
sys.exit(1 if FAILS else 0)
