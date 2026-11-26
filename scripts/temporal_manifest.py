"""Temporal data, step 2 (council/PLAN-temporal.md): pair every dense 10 Hz game key with real camera sweeps.

  python scripts/temporal_manifest.py --keys data/sim/keys_tpv10 --out data/temporal_manifest.json

Tick time: keyframe k sits at tick 5k; ticks in between get a time interpolated linearly between the two real sample
timestamps. For each of the six cameras the nearest sweep of the same scene is taken; any offset above 45 ms rejects
the tick. Sequences break at rejected ticks and at repeated rig bundles (same six files as the previous tick).
Writes {scene: [{"tick", "t_us", "files": [6], "off_ms": [6], "ok", "seq"}...]} and prints reject / duration stats.
"""
import argparse, bisect, glob, json, os
import numpy as np
from nuscenes.nuscenes import NuScenes

p = argparse.ArgumentParser()
p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--dataroot", default="data/nuscenes_full")
p.add_argument("--out", default="data/temporal_manifest.json")
p.add_argument("--max_ms", type=float, default=45.0)
a = p.parse_args()
nusc = NuScenes("v1.0-trainval", a.dataroot, verbose=False)
by = {s["name"]: s for s in nusc.scene}
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
man, stats = {}, []
for tj in sorted(glob.glob(f"{a.keys}/*/ticks.json")):
    sc = os.path.basename(os.path.dirname(tj))
    T = json.load(open(tj))["ticks"]
    scene = by[sc]
    toks, t = [], scene["first_sample_token"]
    while t:
        toks.append(t); t = nusc.get("sample", t)["next"]
    kts = [nusc.get("sample", x)["timestamp"] for x in toks]
    s0 = nusc.get("sample", toks[0])
    chains = {}
    for c in CAMS:  # all sweeps of this camera in the scene, time-sorted
        sd, L = nusc.get("sample_data", s0["data"][c]), []
        while sd["prev"]:
            sd = nusc.get("sample_data", sd["prev"])
        while True:
            L.append((sd["timestamp"], sd["filename"]))
            if not sd["next"]:
                break
            sd = nusc.get("sample_data", sd["next"])
        chains[c] = (np.array([x[0] for x in L]), [x[1] for x in L])
    rows, prev_files, seq = [], None, 0
    for tick in sorted(int(x) for x in T):
        k = tick // 5
        frac = (tick - 5 * k) / 5
        t_us = kts[k] if k + 1 >= len(kts) else kts[k] + frac * (kts[k + 1] - kts[k])
        files, offs = [], []
        for c in CAMS:
            ts, fn = chains[c]
            j = int(np.argmin(np.abs(ts - t_us)))
            files.append(fn[j]); offs.append(round((ts[j] - t_us) / 1000, 1))
        ok = max(abs(o) for o in offs) <= a.max_ms and files != prev_files
        if not ok or (rows and not rows[-1]["ok"]):
            seq += 1
        rows.append({"tick": tick, "t_us": int(t_us), "files": files, "off_ms": offs, "ok": ok, "seq": seq,
                     "token": T[str(tick)]})
        prev_files = files
    man[sc] = rows
    good = sum(r["ok"] for r in rows)
    stats.append((sc, len(rows), good))
json.dump(man, open(a.out, "w"))
n, g = sum(s[1] for s in stats), sum(s[2] for s in stats)
print(f"MANIFEST {len(stats)} drives, {n} ticks, {g} usable ({100 * g / max(1, n):.1f}%), {g / 10 / 60:.1f} usable minutes")
for sc, nn, gg in stats[:10]:
    print(sc, nn, gg)
