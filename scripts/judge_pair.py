"""Experiment T judge: score renders against the exact game keys of the same moments (council/tpv-s1-fable.md).

  python scripts/judge_pair.py --renders runs/judge/tpv_E1_10k --keys data/sim/keys_tpv --judge m2f --out runs/judge/tpv_E1_10k_m2f.json
  python scripts/judge_pair.py --compare A.json B.json

The same segmenter (Mask2Former Swin-L Cityscapes panoptic, or SegFormer-B5) is run on the render and on the game key
(top 32 rows cropped, 512x256, upscaled 2x for the segmenter). Our own car (always at the same pixels in the
third-person view) is masked out of both. Cars: one-to-one Hungarian matching at IoU >= 0.3 between render cars and
key cars (>= 8 px tall) -> pooled F1. Road: IoU of the road masks. Pixels: L1 (0-255) and LPIPS (alex) vs the key.
--keys_self scores the keys against themselves through a VAE round trip (ceiling check, --vae).
--compare: paired scene bootstrap (2000 draws) of F1 / road IoU / LPIPS differences with 95% intervals.
"""
import argparse, glob, json, math, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--renders", default="")
p.add_argument("--keys", default="data/sim/keys_tpv")
p.add_argument("--judge", default="m2f", choices=["m2f", "segformer"])
p.add_argument("--out", default="")
p.add_argument("--vae", action="store_true", help="ceiling check: score VAE(key) against key instead of renders")
p.add_argument("--compare", nargs=2, default=None)
a = p.parse_args()


def summarise(fr):
    tp, fp, fn = (sum(f[k] for f in fr) for k in ("tp", "fp", "fn"))
    return {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "tp": tp, "fp": fp, "fn": fn,
            "road_iou": float(np.mean([f["road_iou"] for f in fr])), "l1": float(np.mean([f["l1"] for f in fr])),
            "lpips": float(np.mean([f["lpips"] for f in fr])), "frames": len(fr)}


if a.compare:
    X, Y = (json.load(open(f))["frames"] for f in a.compare)
    key = lambda f: (f["scene"], f["k"], f["seed"])
    ym = {key(f): f for f in Y}
    pairs = [(f, ym[key(f)]) for f in X if key(f) in ym]
    scenes = sorted({f["scene"] for f, _ in pairs})
    by = {s: [(x, y) for x, y in pairs if x["scene"] == s] for s in scenes}
    rng = np.random.default_rng(0)
    D = {m: [] for m in ("f1", "road_iou", "lpips")}
    for _ in range(2000):
        pick = rng.choice(scenes, len(scenes))
        sx = summarise([x for s in pick for x, _ in by[s]]); sy = summarise([y for s in pick for _, y in by[s]])
        for m in D:
            D[m].append(sx[m] - sy[m])
    sx, sy = summarise([x for x, _ in pairs]), summarise([y for _, y in pairs])
    res = {"A": a.compare[0], "B": a.compare[1], "pairs": len(pairs), "scenes": len(scenes)}
    for m in D:
        res[m] = [sx[m], sy[m]]; res["d_" + m] = sx[m] - sy[m]; res["d_" + m + "_ci95"] = np.percentile(D[m], [2.5, 97.5]).tolist()
    print("COMPARE", json.dumps(res))
    raise SystemExit

import torch
import lpips
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label as cc_label

dev = "cuda"
W, H, CROP = 512, 288, 32
# own car in the third-person view (sim_render.py --view tpv: camera 7.5 m behind the car centre, 2.4 m up, pitch -8 deg,
# 60 deg HFOV): project a generous box around the car and mask it out of every comparison
f = W / 2 / math.tan(math.radians(30))
cp, sp = math.cos(math.radians(8)), math.sin(math.radians(8))
pts = []
for x in (-1.05, 1.05):
    for y in (-2.5, 2.5):
        for z in (0.0, 1.6):
            Yf, Zu = y + 7.5, z - 2.4  # forward, up relative to the camera
            depth = Yf * cp - Zu * sp; up = Yf * sp + Zu * cp
            pts.append((W / 2 + f * x / depth, H / 2 - f * up / depth - CROP))
pts = np.array(pts)
from scipy.spatial import ConvexHull
from PIL import ImageDraw, ImageFilter
hull = pts[ConvexHull(pts).vertices]
_m = Image.new("L", (512, 256), 0); ImageDraw.Draw(_m).polygon([tuple(q) for q in hull], fill=255)
OWN = np.asarray(_m.filter(ImageFilter.MaxFilter(9))) > 0  # ~4 px margin
x0, x1, y0, y1 = int(hull[:, 0].min()), int(hull[:, 0].max()), int(hull[:, 1].min()), int(hull[:, 1].max())
print("own-car box x", x0, x1, "y", y0, y1, flush=True)

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
lp = lpips.LPIPS(net="alex").to(dev).eval()


@torch.no_grad()
def segment(img):
    inp = proc(images=img.resize((1024, 512), Image.BICUBIC), return_tensors="pt").to(dev)
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
        sem = torch.nn.functional.interpolate(out.logits, size=(256, 512), mode="bilinear").argmax(1)[0].cpu().numpy()
        lb, n = cc_label(sem == CAR)
        cars = [lb == i for i in range(1, n + 1) if (lb == i).sum() >= 30]
        road = sem == ROAD
    keep = []
    for m in cars:
        rows = np.where(m.any(1))[0]
        if (m & OWN).sum() > 0.5 * m.sum() or len(rows) == 0 or rows.max() - rows.min() + 1 < 8:
            continue
        keep.append(m & ~OWN)
    return keep, road & ~OWN


def to_t(img):
    return torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1


vae = None
if a.vae:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
idx = json.load(open(f"{a.renders}/index.json"))["items"]
frames = []
for it in idx:
    kf = glob.glob(f"{a.keys}/{it['scene']}/{it['k']:03d}_*.png")
    if not kf:
        continue
    key = Image.open(kf[0]).convert("RGB").resize((512, 288), Image.BILINEAR).crop((0, CROP, 512, 288))
    if a.vae:
        with torch.no_grad():
            z = vae.encode(to_t(key)).latent_dist.mean
            r = vae.decode(z).sample[0].clamp(-1, 1)
        ren = Image.fromarray(((r + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())
    else:
        ren = Image.open(f"{a.renders}/{it['file']}").convert("RGB")
    ck, rk = segment(key)
    cr, rr = segment(ren)
    tp = 0
    if ck and cr:
        P = np.stack(cr).reshape(len(cr), -1).astype(np.float32); G = np.stack(ck).reshape(len(ck), -1).astype(np.float32)
        inter = P @ G.T; iou = inter / np.maximum(P.sum(1)[:, None] + G.sum(1)[None] - inter, 1)
        r_, c_ = linear_sum_assignment(-iou); tp = int((iou[r_, c_] >= 0.3).sum())
    ri = float((rk & rr).sum() / max(1, (rk | rr).sum()))
    A, B = to_t(ren), to_t(key)
    with torch.no_grad():
        l1 = float((A - B).abs().mean() * 127.5); lpv = float(lp(A, B))
    frames.append({"scene": it["scene"], "k": it["k"], "seed": it["seed"], "tp": tp, "fp": len(cr) - tp, "fn": len(ck) - tp,
                   "key_cars": len(ck), "road_iou": ri, "l1": l1, "lpips": lpv})
s = summarise(frames)
s.update({"renders": a.renders, "judge": a.judge, "vae_ceiling": a.vae})
json.dump({"summary": s, "frames": frames}, open(a.out, "w"), indent=1)
print("JUDGE", json.dumps(s))
