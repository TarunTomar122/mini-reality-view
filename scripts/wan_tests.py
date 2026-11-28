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
D = lambda c, st, stage, sw=True, prod=PROD, modes=MODES, prior=None: WP.decide_s1(c, st, stage, {"ok": sw}, prod, modes, prior)
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
check("forecast skips charged kinds", "overfit_h" not in WP.forecast_s1({**T_, "done_kinds": {"overfit", "resume_test"}})["pending"])
check("forecast floor flag", WP.forecast_s1({**T_, "wall_s": 7.0})["floor_12k_reachable"] is False)
check("panel cost", abs(WP.panel_cost_h(1000, 0.5, 60, 1.0, 120) - (5 * (60 + 500) + 8 * (120 + 1000)) / 3600) < 1e-9)

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
