"""Car map preflight P3 (council/PLAN-carmap.md): is BEVFormer better on drives it was trained on?

  python scripts/bev_leak.py --train tmp/bev_boxes_train_*.json --train_scenes tmp/bev_leak_train30.txt \
      --val tmp/bev_boxes_otherval.json --out runs/judge/bev_leak.json

Predicted car boxes (score >= 0.3, same as the 0.417 hull test) vs nuScenes vehicle.car boxes with >= 1 lidar point,
in the ego frame, within 50 m: greedy one-to-one matching by centre distance (<= 2 m). Recall, precision and mean
centre error, overall and by distance (0-15, 15-30, 30-50 m), for 30 seeded TRAIN drives vs the unused official-VAL
drives. Rule: recall or precision differing by > 0.05, or centre error by > 25%, means the training maps must be
degraded to the val error rates before X1 / X2.
"""
import argparse, glob, json
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

p = argparse.ArgumentParser()
p.add_argument("--train", nargs="+", required=True)
p.add_argument("--train_scenes", required=True)
p.add_argument("--val", nargs="+", required=True)
p.add_argument("--thr", type=float, default=0.3)
p.add_argument("--out", required=True)
a = p.parse_args()
nusc = NuScenes("v1.0-trainval", "data/nuscenes_full", verbose=False)
BINS = [(0, 15), (15, 30), (30, 50)]


def load(files, scenes=None):
    B = {}
    for f in files:
        for g in glob.glob(f):
            for tok, v in json.load(open(g)).items():
                if scenes is None or v["scene"] in scenes:
                    B[tok] = v
    return B


def score(B):
    st = {b: [0, 0, 0, []] for b in BINS}  # tp, fp, fn, centre errors
    for tok, v in B.items():
        smp = nusc.get("sample", tok)
        ep = nusc.get("ego_pose", nusc.get("sample_data", smp["data"]["LIDAR_TOP"])["ego_pose_token"])
        Rw, tw = Quaternion(ep["rotation"]), np.array(ep["translation"])
        gt = []
        for at in smp["anns"]:
            an = nusc.get("sample_annotation", at)
            if an["category_name"].startswith("vehicle.car") and an["num_lidar_pts"] >= 1:
                c = Rw.inverse.rotate(np.array(an["translation"]) - tw)
                if np.linalg.norm(c[:2]) <= 50:
                    gt.append(c[:2])
        Rl, tl = Quaternion(v["lidar2ego_r"]), np.array(v["lidar2ego_t"])
        pr = []
        for bb, s, lab in zip(v["boxes"], v["scores"], v["labels"]):
            if lab == "car" and s >= a.thr:
                c = Rl.rotate(np.array(bb[:3])) + tl
                if np.linalg.norm(c[:2]) <= 50:
                    pr.append(c[:2])
        gt, pr = np.array(gt).reshape(-1, 2), np.array(pr).reshape(-1, 2)
        used_g, used_p, pairs = set(), set(), []
        if len(gt) and len(pr):
            D = np.linalg.norm(gt[:, None] - pr[None], axis=-1)
            for k in np.argsort(D, axis=None):
                i, j = divmod(int(k), D.shape[1])
                if D[i, j] > 2:
                    break
                if i not in used_g and j not in used_p:
                    used_g.add(i); used_p.add(j); pairs.append((i, j, D[i, j]))
        binof = lambda c: next((b for b in BINS if b[0] <= np.linalg.norm(c) < b[1]), None)
        for i, j, d in pairs:
            b = binof(gt[i]); st[b][0] += 1; st[b][3].append(d)
        for j in range(len(pr)):
            if j not in used_p and binof(pr[j]):
                st[binof(pr[j])][1] += 1
        for i in range(len(gt)):
            if i not in used_g and binof(gt[i]):
                st[binof(gt[i])][2] += 1
    out = {}
    for b, (tp, fp, fn, e) in st.items():
        out[f"{b[0]}-{b[1]}m"] = {"recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp), "centre_err_m": float(np.mean(e)) if e else None, "tp": tp, "fp": fp, "fn": fn}
    tp = sum(v[0] for v in st.values()); fp = sum(v[1] for v in st.values()); fn = sum(v[2] for v in st.values())
    out["all"] = {"recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp),
                  "centre_err_m": float(np.mean([x for v in st.values() for x in v[3]])), "frames": len(B)}
    return out


tr = score(load(a.train, set(open(a.train_scenes).read().split())))
va = score(load(a.val))
flag = (abs(tr["all"]["recall"] - va["all"]["recall"]) > 0.05 or abs(tr["all"]["precision"] - va["all"]["precision"]) > 0.05
        or abs(tr["all"]["centre_err_m"] - va["all"]["centre_err_m"]) > 0.25 * va["all"]["centre_err_m"])
res = {"train": tr, "val": va, "leak_flag": bool(flag)}
json.dump(res, open(a.out, "w"), indent=1)
print("LEAK", json.dumps({"train": tr["all"], "val": va["all"], "leak_flag": flag}))
