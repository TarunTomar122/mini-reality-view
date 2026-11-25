"""Geometry judge ceiling / resolution check: turn the answer keys of the judge frames into a "renders" folder that
judge_score.py can score (council/PLAN-bev.md change 2).

  python scripts/judge_keys.py --like runs/judge/sd10k_normal --keys data/targets_full_low --out runs/judge/keys_lidar

Keys are 512x288 RGBA drone views; the top 32 rows are cropped (as for the renders) and the colour is composited on
black by alpha (what the renderer is trained to imitate). Only seed 0 entries are written (keys have no seed).
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--like", required=True, help="renders folder whose index.json lists the judge frames")
p.add_argument("--keys", default="data/targets_full_low")
p.add_argument("--out", required=True)
a = p.parse_args()
idx = json.load(open(f"{a.like}/index.json"))
items = []
for it in idx["items"]:
    if it["seed"] != 0:
        continue
    f = glob.glob(f"{a.keys}/{it['scene']}/*_{it['token']}.png")
    if not f:
        continue
    t = np.asarray(Image.open(f[0]).convert("RGBA")).astype(np.float32)[32:]
    rgb = (t[..., :3] * t[..., 3:] / 255).round().astype(np.uint8)
    os.makedirs(f"{a.out}/{it['scene']}", exist_ok=True)
    Image.fromarray(rgb).save(f"{a.out}/{it['file']}")
    items.append(it)
json.dump({"keys": a.keys, "items": items}, open(f"{a.out}/index.json", "w"), indent=1)
print("KEYS_DONE", len(items))
