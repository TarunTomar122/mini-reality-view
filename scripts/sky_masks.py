"""Stage 6 helper: sky masks for drivestudio, with the same SegFormer-B5 Cityscapes model it expects,
loaded through transformers instead of the old mmcv stack. Cityscapes class 10 = sky.
Also writes coarse fine_dynamic_masks (human = 11,12; vehicle = 13..18) in drivestudio's layout.
"""
import argparse, glob, os
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

p = argparse.ArgumentParser()
p.add_argument("--data_root", required=True, help="drivestudio processed split dir, e.g. .../processed_10Hz/mini")
p.add_argument("--scenes", default="", help="comma list of scene ids (000, 001...), default all")
a = p.parse_args()
name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
proc = SegformerImageProcessor.from_pretrained(name)
model = SegformerForSemanticSegmentation.from_pretrained(name).cuda().half().eval()
scenes = a.scenes.split(",") if a.scenes else sorted(os.listdir(a.data_root))
for sc in scenes:
    files = sorted(glob.glob(f"{a.data_root}/{sc}/images/*.jpg"))
    dirs = {k: f"{a.data_root}/{sc}/{k}" for k in ["sky_masks", "fine_dynamic_masks/all", "fine_dynamic_masks/human", "fine_dynamic_masks/vehicle"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    for i in range(0, len(files), 8):
        chunk = files[i:i + 8]
        ims = [Image.open(f).convert("RGB") for f in chunk]
        with torch.no_grad():
            x = proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
            logits = model(pixel_values=x).logits
            H, W = ims[0].size[1], ims[0].size[0]
            seg = torch.nn.functional.interpolate(logits.float(), size=(H, W), mode="bilinear").argmax(1).cpu().numpy()
        for f, m in zip(chunk, seg):
            b = os.path.splitext(os.path.basename(f))[0]
            human, vehicle = np.isin(m, [11, 12]), np.isin(m, [13, 14, 15, 16, 17, 18])
            for k, v in [("sky_masks", m == 10), ("fine_dynamic_masks/human", human),
                         ("fine_dynamic_masks/vehicle", vehicle), ("fine_dynamic_masks/all", human | vehicle)]:
                Image.fromarray(v.astype(np.uint8) * 255).save(f"{dirs[k]}/{b}.png")
    print(sc, len(files), flush=True)
print("SKY_DONE")
