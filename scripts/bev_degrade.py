"""Car map P3 follow-up (council/PLAN-carmap.md): BEVFormer is better on drives it was trained on (runs/judge/bev_leak.json:
recall 0.85 vs 0.72, precision 0.79 vs 0.72), so the TRAINING box maps are degraded to the val error rates by distance
(drops, centre jitter, phantoms), seeded per frame so X1 and X2 see the same degraded boxes.

  python scripts/bev_degrade.py --leak runs/judge/bev_leak.json --inp 'tmp/bev_boxes_train_*.json' --out tmp/bev_boxes_trainD.json

Per distance bin (0-15, 15-30, 30-50 m; beyond 50 m uses 30-50):
  drop     d = 1 - recall_val / recall_train            (every box, any class, same rate)
  jitter   sigma per axis so the mean centre error grows from train to val: sqrt(e_v^2 - e_t^2) / sqrt(pi/2)
  phantoms extra false boxes per kept vehicle box so precision falls to val: copies of a vehicle box of the same frame
           moved 2.5-6 m (random direction) with a random yaw; score copied
Rates are computed from the car numbers and applied to every class (phantoms only for vehicle classes).
Check: re-run bev_leak.py with --train pointing at the output; train and val should then agree within the P3 thresholds.
"""
import argparse, glob, json, math, zlib
import numpy as np
from pyquaternion import Quaternion

p = argparse.ArgumentParser()
p.add_argument("--leak", required=True)
p.add_argument("--inp", required=True)
p.add_argument("--out", required=True)
p.add_argument("--thr", type=float, default=0.3)
a = p.parse_args()
L = json.load(open(a.leak))
BINS = ["0-15m", "15-30m", "30-50m"]
VEH = {"car", "truck", "bus", "trailer", "construction_vehicle"}
RATE = {}
for b in BINS:
    t, v = L["train"][b], L["val"][b]
    d = max(0.0, 1 - v["recall"] / t["recall"])
    tp, fp = t["tp"] * (1 - d), t["fp"] * (1 - d)
    need = tp * (1 / v["precision"] - 1)
    ph = max(0.0, need - fp) / max(1.0, (t["tp"] + t["fp"]) * (1 - d))  # per kept predicted car box
    sig = math.sqrt(max(0.0, v["centre_err_m"] ** 2 - t["centre_err_m"] ** 2)) / math.sqrt(math.pi / 2)
    RATE[b] = (d, sig, ph)
print("rates (drop, sigma m, phantoms per kept box):", {b: tuple(round(x, 3) for x in r) for b, r in RATE.items()}, flush=True)


def binof(r):
    return "0-15m" if r < 15 else "15-30m" if r < 30 else "30-50m"


B = {}
for f in sorted(glob.glob(a.inp)):
    B.update(json.load(open(f)))
out, n_in, n_keep, n_ph = {}, 0, 0, 0
for tok, v in B.items():
    rng = np.random.default_rng(zlib.crc32(f"degrade|{tok}".encode()))
    Rl, tl = Quaternion(v["lidar2ego_r"]), np.array(v["lidar2ego_t"])
    nb, ns, nl = [], [], []
    for bb, s, lab in zip(v["boxes"], v["scores"], v["labels"]):
        u_drop, jx, jy = rng.random(), rng.standard_normal(), rng.standard_normal()  # fixed draws per box
        u_ph, ang, dist, yaw = rng.random(), rng.uniform(0, 2 * math.pi), rng.uniform(2.5, 6.0), rng.uniform(-math.pi, math.pi)
        if s < a.thr:
            continue
        n_in += 1
        c = Rl.rotate(np.array(bb[:3])) + tl
        d, sig, ph = RATE[binof(float(np.linalg.norm(c[:2])))]
        if u_drop < d:
            continue
        b2 = list(bb); b2[0] += sig * jx; b2[1] += sig * jy
        nb.append(b2); ns.append(s); nl.append(lab); n_keep += 1
        if lab in VEH and u_ph < ph:
            b3 = list(bb); b3[0] += dist * math.cos(ang); b3[1] += dist * math.sin(ang); b3[6] = yaw
            nb.append(b3); ns.append(s); nl.append(lab); n_ph += 1
    out[tok] = {**v, "boxes": nb, "scores": ns, "labels": nl, "degraded": True}
json.dump(out, open(a.out, "w"))
print(f"frames {len(out)}; boxes >= thr {n_in} -> kept {n_keep} + phantoms {n_ph}")
