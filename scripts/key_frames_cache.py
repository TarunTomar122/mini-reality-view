"""P3 hull F1 (PLAN-carmap v4): judge frames + cached Mask2Former key masks for drives that have game keys but no
renders. Frames = 4 evenly spaced keyed keyframes per drive (the sd_render.py --keys_like rule); masks with exactly
judge_pair.py's segmenter settings and cache format (runs/judge/keycache_m2f/<scene>_<k:03d>_<token>.npz).
Writes <out>/index.json {"items": [{scene, k, token, seed: 0}]} for bev_box_judge.py --like <out>.

  python scripts/key_frames_cache.py --scenes_file tmp/bev_otherval84.txt --out runs/judge/keyidx_val84
"""
import argparse, glob, json, math, os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import ConvexHull
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

p = argparse.ArgumentParser()
p.add_argument("--scenes_file", required=True)
p.add_argument("--keys", default="data/sim/keys_tpv")
p.add_argument("--frames", type=int, default=4)
p.add_argument("--out", required=True)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True); CACHE = "runs/judge/keycache_m2f"; os.makedirs(CACHE, exist_ok=True)
dev = "cuda"; W, H, CROP = 512, 288, 32
f = W / 2 / math.tan(math.radians(30)); cp, sp = math.cos(math.radians(8)), math.sin(math.radians(8)); pts = []
for x in (-1.05, 1.05):
    for y in (-2.5, 2.5):
        for z in (0.0, 1.6):
            Yf, Zu = y + 7.5, z - 2.4; depth = Yf * cp - Zu * sp; up = Yf * sp + Zu * cp
            pts.append((W / 2 + f * x / depth, H / 2 - f * up / depth - CROP))
pts = np.array(pts); hull = pts[ConvexHull(pts).vertices]
_m = Image.new("L", (512, 256), 0); ImageDraw.Draw(_m).polygon([tuple(q) for q in hull], fill=255)
OWN = np.asarray(_m.filter(ImageFilter.MaxFilter(9))) > 0
name = "facebook/mask2former-swin-large-cityscapes-panoptic"
proc = AutoImageProcessor.from_pretrained(name); model = Mask2FormerForUniversalSegmentation.from_pretrained(name).to(dev).eval()
lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}; CAR, ROAD = lab["car"], lab["road"]


@torch.no_grad()
def segment(img):  # = judge_pair.py segment() for --judge m2f
    inp = proc(images=img.resize((1024, 512), Image.BICUBIC), return_tensors="pt").to(dev)
    pan = proc.post_process_panoptic_segmentation(model(**inp), threshold=0.5, target_sizes=[(256, 512)])[0]
    seg = pan["segmentation"].cpu().numpy()
    cars = [seg == s["id"] for s in pan["segments_info"] if s["label_id"] == CAR]
    road = np.zeros((256, 512), bool)
    for s in pan["segments_info"]:
        if s["label_id"] == ROAD:
            road |= seg == s["id"]
    keep = []
    for m in cars:
        rows = np.where(m.any(1))[0]
        if (m & OWN).sum() > 0.5 * m.sum() or len(rows) == 0 or rows.max() - rows.min() + 1 < 8:
            continue
        keep.append(m & ~OWN)
    return keep, road & ~OWN


items = []
for sc in open(a.scenes_file).read().split():
    kk = sorted(glob.glob(f"{a.keys}/{sc}/*.png"))
    if len(kk) < a.frames:
        print("skip (too few keys)", sc); continue
    for i in np.linspace(0, len(kk) - 1, a.frames).round().astype(int):
        kp = kk[i]; b = os.path.basename(kp)[:-4]; k, tok = int(b[:3]), b[4:]
        c = f"{CACHE}/{sc}_{b}.npz"
        if not os.path.exists(c):
            cars, road = segment(Image.open(kp).convert("RGB").resize((512, 288), Image.BILINEAR).crop((0, CROP, 512, 288)))
            np.savez_compressed(c, cars=np.packbits(np.stack(cars), axis=-1) if cars else np.zeros((0, 256, 64), np.uint8), road=np.packbits(road, axis=-1))
        items.append({"scene": sc, "k": k, "token": tok, "seed": 0})
json.dump({"items": items}, open(f"{a.out}/index.json", "w"), indent=1)
print("KEYIDX", len(items), "frames ->", a.out)
