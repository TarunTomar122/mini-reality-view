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
check("below 12k: unresolved", WP.decide_s1(cmp(0.50), 11999, "24h", True)["verdict"].startswith("unresolved"))
check("24h pass", WP.decide_s1(cmp(0.46), 12000, "24h", True)["verdict"].startswith("PASS"))
check("24h extension band", WP.decide_s1(cmp(0.43), 14000, "24h", True)["verdict"].startswith("EXTEND"))
check("24h extension band lower edge 0.4156", WP.decide_s1(cmp(0.4156), 14000, "24h", True)["verdict"].startswith("EXTEND"))
check("24h below band: stop", WP.decide_s1(cmp(0.41), 14000, "24h", True)["verdict"].startswith("STOP"))
check("24h band but W fails: stop", WP.decide_s1(cmp(0.43, wr=-0.05), 14000, "24h", True)["verdict"].startswith("STOP"))
check("24h band but swap fails: stop", WP.decide_s1(cmp(0.43), 14000, "24h", False)["verdict"].startswith("STOP"))
check("final in band: stop (bar stays 0.4456)", WP.decide_s1(cmp(0.44), 20000, "final", True)["verdict"].startswith("STOP"))
check("final pass", WP.decide_s1(cmp(0.45), 20000, "final", True)["verdict"].startswith("PASS"))
check("road below A fails", not WP.decide_s1(cmp(0.46, road=0.48), 12000, "24h", True)["all_pass"])
check("ineligible ID fails", not WP.decide_s1(cmp(0.46, cp=False), 12000, "24h", True)["all_pass"])
check("NaN car F1 fails", not WP.decide_s1(cmp(float("nan")), 12000, "24h", True)["all_pass"])
check("incomplete panel fails", not WP.decide_s1({**cmp(0.46), "runs": 11}, 12000, "24h", True)["all_pass"])
check("check stage gives no verdict", WP.decide_s1(cmp(0.46), 12000, "check", True)["verdict"].startswith("informational"))
d = WP.donor_following({"a": 0.5, "b": 0.4}, {"a": 0.2, "b": 0.1}, [0, 1, 1])
check("donor following", d["ok"], d)
check("not donor following (lag)", not WP.donor_following({"a": 0.5}, {"a": 0.1}, [3, 4, float("inf")])["ok"])

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
    with open(path, "w") as f:  # a preempted job: two beats, no end record; plus a torn last line
        f.write(json.dumps({"job": "j1", "kind": "train", "t0": now - 7200, "t": now - 7200}) + "\n")
        f.write(json.dumps({"job": "j1", "kind": "train", "t0": now - 7200, "t": now - 3600}) + "\n")
        f.write(json.dumps({"job": "j2", "kind": "panel", "t0": now - 1800, "t": now - 900}) + "\n")
        f.write('{"job": "j3", "ki')
    L = WP.Ledger(path, "train")
    h = L.hours(live=False)
    check("ledger charges preempted work up to the last beat", abs(h - 1.25) < 1e-6, h)
    check("ledger by kind", {k: round(v, 3) for k, v in L.by_kind().items()} == {"train": 1.0, "panel": 0.25}, L.by_kind())
    L.start(); L.stop()
    own = [json.loads(x) for x in open(path) if x.strip().startswith("{") and x.strip().endswith("}") and L.job in x]
    check("ledger appends own beats after a torn line", len(own) >= 2, own)

print(f"WAN_TESTS FAILS={FAILS}")
sys.exit(1 if FAILS else 0)
