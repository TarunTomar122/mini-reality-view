"""Car map R (council/PLAN-carmap.md v4): key car masks for the loss. For every flat keyframe key of the given drives
(data/lat512_tpv sample tokens), Mask2Former (Cityscapes panoptic, same settings and own-car exclusion as
judge_pair.py) on the game key (top 32 rows cropped -> 512x256); union of car instances, own car excluded, area-pooled
8x8 -> 32x64 latent cells (fraction of car pixels per cell). Writes <out>/<scene>.pt {sample_tokens, mask (n,32,64) fp16}.

  python scripts/key_carmasks.py --lists data/temporal_lists.json --split train --out data/keycar_tpv
"""
import argparse, glob, json, math, os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import ConvexHull
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

p = argparse.ArgumentParser()
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--split", default="train")
p.add_argument("--lat", default="data/lat512_tpv")
p.add_argument("--keys", default="data/sim/keys_tpv")
p.add_argument("--shard", default="0/1")
p.add_argument("--out", required=True)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
W, H, CROP = 512, 288, 32
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
lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}; CAR = lab["car"]


@torch.no_grad()
def carmask(img):
    inp = proc(images=img.resize((1024, 512), Image.BICUBIC), return_tensors="pt").to(dev)
    pan = proc.post_process_panoptic_segmentation(model(**inp), threshold=0.5, target_sizes=[(256, 512)])[0]
    seg = pan["segmentation"].cpu().numpy(); out = np.zeros((256, 512), bool)
    for s in pan["segments_info"]:
        if s["label_id"] != CAR:
            continue
        m = seg == s["id"]; rows = np.where(m.any(1))[0]
        if (m & OWN).sum() > 0.5 * m.sum() or len(rows) == 0 or rows.max() - rows.min() + 1 < 8:
            continue
        out |= m & ~OWN
    return out


drives = json.load(open(a.lists))[a.split]
si, sn = map(int, a.shard.split("/"))
for n_, sc in enumerate(sorted(drives)):
    if n_ % sn != si or os.path.exists(f"{a.out}/{sc}.pt") or not os.path.exists(f"{a.lat}/{sc}.pt"):
        continue
    toks = torch.load(f"{a.lat}/{sc}.pt")["sample_tokens"]; masks = []
    for t in toks:
        kp = glob.glob(f"{a.keys}/{sc}/*_{t}.png")
        if not kp:
            masks.append(torch.full((32, 64), float("nan"))); continue
        im = Image.open(kp[0]).convert("RGB").resize((512, 288), Image.BILINEAR).crop((0, CROP, 512, 288))
        m = torch.from_numpy(carmask(im)).float()[None, None]
        masks.append(torch.nn.functional.avg_pool2d(m, 8)[0, 0])
    torch.save({"sample_tokens": toks, "mask": torch.stack(masks).half()}, f"{a.out}/{sc}.pt")
    print(sc, len(toks), "keys", flush=True)
print("KEYCAR_DONE")
