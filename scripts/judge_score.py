"""Geometry judge, part 2: score renders against the projected truth (council/PLAN-sim.md).

  python judge_score.py --renders runs/judge/simA_5k --judge m2f [--shift 0] --out runs/judge/simA_5k_m2f.json
  python judge_score.py --compare runs/judge/simB_5k_m2f.json runs/judge/simA_5k_m2f.json

Judges: m2f = Mask2Former Swin-L Cityscapes panoptic (car instances with score >= 0.5, road pixels);
        segformer = SegFormer-B5 Cityscapes (car = connected components >= 30 px, road pixels). Renders are upscaled 2x.
Cars: predicted instances whose bottom pixel lies on the ground within 30 m are matched one-to-one (Hungarian,
IoU >= 0.3) to the visible projected nuScenes car hulls -> pooled TP/FP/FN -> F1 = 2TP / (2TP + FP + FN).
Kerbs: road-region boundary pixels -> ground metres; symmetric mean nearest distance to the map's drivable-area
boundary within 30 m and inside the view, each direction capped at 2 m; no predicted edge = 2 m.
--compare: paired scene bootstrap (2000 draws) of the F1 and kerb-error differences, with 95% intervals.
"""
import argparse, glob, json, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--renders", default="")
p.add_argument("--gt", default="data/judge_gt")
p.add_argument("--judge", default="m2f", choices=["m2f", "segformer"])
p.add_argument("--shift", default="0")
p.add_argument("--out", default="")
p.add_argument("--compare", nargs=2, default=None)
a = p.parse_args()


def summarise(frames):
    tp, fp, fn = (sum(f[k] for f in frames) for k in ("tp", "fp", "fn"))
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return {"f1": f1, "tp": tp, "fp": fp, "fn": fn, "kerb_m": float(np.mean([f["kerb_m"] for f in frames])), "frames": len(frames)}


if a.compare:
    X, Y = (json.load(open(f))["frames"] for f in a.compare)
    key = lambda f: (f["scene"], f["k"], f["seed"])
    ym = {key(f): f for f in Y}
    pairs = [(f, ym[key(f)]) for f in X if key(f) in ym]
    scenes = sorted({f["scene"] for f, _ in pairs})
    by = {s: [(x, y) for x, y in pairs if x["scene"] == s] for s in scenes}
    rng = np.random.default_rng(0)
    d_f1, d_k = [], []
    for _ in range(2000):
        pick = rng.choice(scenes, len(scenes))
        xs = [x for s in pick for x, _ in by[s]]; ys = [y for s in pick for _, y in by[s]]
        sx, sy = summarise(xs), summarise(ys)
        d_f1.append(sx["f1"] - sy["f1"]); d_k.append(sx["kerb_m"] - sy["kerb_m"])
    sx, sy = summarise([x for x, _ in pairs]), summarise([y for _, y in pairs])
    res = {"A": a.compare[0], "B": a.compare[1], "f1": [sx["f1"], sy["f1"]], "kerb_m": [sx["kerb_m"], sy["kerb_m"]],
           "d_f1": sx["f1"] - sy["f1"], "d_f1_ci95": np.percentile(d_f1, [2.5, 97.5]).tolist(),
           "d_kerb_m": sx["kerb_m"] - sy["kerb_m"], "d_kerb_ci95": np.percentile(d_k, [2.5, 97.5]).tolist(),
           "kerb_rel": (sx["kerb_m"] - sy["kerb_m"]) / max(1e-9, sy["kerb_m"]), "scenes": len(scenes), "pairs": len(pairs)}
    print("COMPARE", json.dumps(res))
    raise SystemExit

import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label as cc_label, binary_erosion
from scipy.spatial import cKDTree
from mrv.nusc import CHASES, chase_camera

dev = "cuda"
W, H, CROP = 512, 288, 32
R, K = chase_camera(W, H, "low")
POS = CHASES["low"][0]
vv, uu = np.mgrid[0:256, 0:512]
pix = np.stack([uu.ravel() + 0.5, vv.ravel() + 0.5 + CROP, np.ones(uu.size)], 1)
d = (np.linalg.inv(K) @ pix.T).T @ R
with np.errstate(divide="ignore", invalid="ignore"):
    tg = -POS[2] / d[:, 2]
GROUND = np.where((tg > 0)[:, None], POS[:2] + d[:, :2] * tg[:, None], np.nan).reshape(256, 512, 2)
DIST = np.linalg.norm(GROUND, axis=-1)
NEAR = np.nan_to_num(DIST, nan=1e9) <= 30

if a.judge == "m2f":
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    name = "facebook/mask2former-swin-large-cityscapes-panoptic"
    proc = AutoImageProcessor.from_pretrained(name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(name).to(dev).eval()
else:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    proc = SegformerImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name).to(dev).eval()
lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
CAR, ROAD = lab["car"], lab["road"]


@torch.no_grad()
def segment(img):
    big = img.resize((1024, 512), Image.BICUBIC)
    inp = proc(images=big, return_tensors="pt").to(dev)
    out = model(**inp)
    if a.judge == "m2f":
        pan = proc.post_process_panoptic_segmentation(out, threshold=0.5, target_sizes=[(256, 512)])[0]
        seg = pan["segmentation"].cpu().numpy()
        cars = [seg == s["id"] for s in pan["segments_info"] if s["label_id"] == CAR]
        road = np.zeros((256, 512), bool)
        for s in pan["segments_info"]:
            if s["label_id"] == ROAD:
                road |= seg == s["id"]
    else:
        logits = torch.nn.functional.interpolate(out.logits, size=(256, 512), mode="bilinear")
        sem = logits.argmax(1)[0].cpu().numpy()
        lb, n = cc_label(sem == CAR)
        cars = [lb == i for i in range(1, n + 1) if (lb == i).sum() >= 30]
        road = sem == ROAD
    return cars, road


def near_car(m):
    ys, xs = np.nonzero(m)
    i = ys.argmax()
    return bool(NEAR[ys[i], xs[i]])


def kerb_error(road, edge_xy):
    pred_edge = road & ~binary_erosion(road, iterations=1) & NEAR
    pred_xy = GROUND[pred_edge]
    pred_xy = pred_xy[np.isfinite(pred_xy).all(1)]
    # truth edges: within 30 m and inside the view (project back by nearest ground pixel lookup)
    g = edge_xy[np.linalg.norm(edge_xy, axis=1) <= 30]
    in_view = cKDTree(GROUND[NEAR].reshape(-1, 2)).query(g, distance_upper_bound=0.3)[0] < 0.3 if len(g) else np.zeros(0, bool)
    g = g[in_view]
    if len(g) == 0:
        return None
    if len(pred_xy) == 0:
        return 2.0
    d1 = np.minimum(cKDTree(g).query(pred_xy)[0], 2.0).mean()
    d2 = np.minimum(cKDTree(pred_xy).query(g)[0], 2.0).mean()
    return float((d1 + d2) / 2)


idx = json.load(open(f"{a.renders}/index.json"))
frames = []
for it in idx["items"]:
    gt = np.load(f"{a.gt}/{it['scene']}/{it['k']:03d}_{it['token']}_sh{float(a.shift):g}.npz")
    cars_gt, road_gt, edge_xy = gt["cars"], gt["road"], gt["edge_xy"]
    cars, road = segment(Image.open(f"{a.renders}/{it['file']}").convert("RGB"))
    cars = [m for m in cars if near_car(m)]
    tp = 0
    if len(cars) and len(cars_gt):
        P = np.stack(cars).reshape(len(cars), -1).astype(np.float32); G = cars_gt.reshape(len(cars_gt), -1).astype(np.float32)
        inter = P @ G.T; union = P.sum(1)[:, None] + G.sum(1)[None] - inter
        iou = inter / np.maximum(union, 1)
        r, c = linear_sum_assignment(-iou)
        tp = int((iou[r, c] >= 0.3).sum())
    ke = kerb_error(road, edge_xy)
    frames.append({"scene": it["scene"], "k": it["k"], "seed": it["seed"], "tp": tp, "fp": len(cars) - tp,
                   "fn": len(cars_gt) - tp, "kerb_m": 2.0 if ke is None else ke, "kerb_valid": ke is not None})
s = summarise(frames)
s.update({"renders": a.renders, "judge": a.judge, "shift": float(a.shift)})
json.dump({"summary": s, "frames": frames}, open(a.out, "w"), indent=1)
print("JUDGE", json.dumps(s))
