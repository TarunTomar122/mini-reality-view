"""Temporal data, step 0 (council/PLAN-temporal.md): freeze the drive lists before any fitting.

  python scripts/temporal_lists.py   -> data/temporal_lists.json {train, dev, decision, final}

train + dev: E3's quarter list (tmp/tpv_quarter.txt, 150 train drives). dev = 6 drives spread over traffic density
(vehicles within 35 m at the middle keyframe) x turning (ego heading change over the drive), seed 0.
decision: the 30 judge drives (val scenes with >= 4 keyframe game keys, sorted, first 30; same as sd_render.py).
final: 24 further val drives with game keys, never viewed or used, chosen by seed 0.
"""
import glob, json, os, random
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import val
from pyquaternion import Quaternion

nusc = NuScenes("v1.0-trainval", "data/nuscenes_full", verbose=False)
by = {s["name"]: s for s in nusc.scene}


def feats(sc):
    toks, t = [], by[sc]["first_sample_token"]
    while t:
        toks.append(t); t = nusc.get("sample", t)["next"]
    yaw = []
    for x in toks:
        ep = nusc.get("ego_pose", nusc.get("sample_data", nusc.get("sample", x)["data"]["LIDAR_TOP"])["ego_pose_token"])
        yaw.append(Quaternion(ep["rotation"]).yaw_pitch_roll[0])
    turn = abs(np.degrees(np.angle(np.exp(1j * (yaw[-1] - yaw[0])))))
    smp = nusc.get("sample", toks[len(toks) // 2])
    ep = nusc.get("ego_pose", nusc.get("sample_data", smp["data"]["LIDAR_TOP"])["ego_pose_token"])
    veh = sum(1 for at in smp["anns"] if nusc.get("sample_annotation", at)["category_name"].startswith("vehicle")
              and np.linalg.norm(np.array(nusc.get("sample_annotation", at)["translation"][:2]) - np.array(ep["translation"][:2])) <= 35)
    return veh, turn


quarter = sorted(open("tmp/tpv_quarter.txt").read().split())
f = {sc: feats(sc) for sc in quarter}
rng = random.Random(0)
med_v = np.median([v for v, _ in f.values()])
cells = {(dv, dt): [s for s in quarter if (f[s][0] > med_v) == dv and (f[s][1] > 30) == dt] for dv in (0, 1) for dt in (0, 1)}
dev = []
for i, key in enumerate(sorted(cells)):  # 6 drives over the four density x turning cells (2 extra from the largest cells)
    pool = [s for s in cells[key] if s not in dev]
    rng.shuffle(pool)
    dev += pool[:1]
rest = sorted((s for s in quarter if s not in dev), key=lambda s: -len(cells[(f[s][0] > med_v, f[s][1] > 30)]))
pool = [s for s in quarter if s not in dev]; rng.shuffle(pool); dev += pool[:6 - len(dev)]
train = [s for s in quarter if s not in dev]
vals = sorted(s for s in set(val) if len(glob.glob(f"data/sim/keys_tpv/{s}/*.png")) >= 4 and os.path.exists(f"data/enc_full/{s}.pt"))
decision = vals[:30]
rest = vals[30:]; rng.shuffle(rest)
final = sorted(rest[:24])
out = {"train": train, "dev": sorted(dev), "decision": decision, "final": final,
       "dev_features": {s: {"vehicles": int(f[s][0]), "turn_deg": round(float(f[s][1]), 1)} for s in dev}}
json.dump(out, open("data/temporal_lists.json", "w"), indent=1)
print({k: len(v) for k, v in out.items() if isinstance(v, list)}, out["dev_features"])
