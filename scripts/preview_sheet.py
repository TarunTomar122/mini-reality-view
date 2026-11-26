"""Build the experiment-T preview sheets (local): per drive, rows = keyed frames;
columns = 6 real photos | game key | E1 | E2 | E3 | E1 with another drive's cameras.
usage: python scripts/preview_sheet.py <step> <pulled preview dir> <cams dir> <out prefix>"""
import glob, os, sys
from PIL import Image, ImageDraw
step, P, C, out = sys.argv[1:5]
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
names = {"scene-0915": "busy junction", "scene-0563": "moving traffic", "scene-0632": "quiet street"}
cols = [("key", "game picture (answer)"), (f"tpvE1_{step}", "E1: our model as start"), (f"tpvE2_{step}", "E2: plain SD start"),
        (f"tpvE3_{step}", "E3: quarter of data"), (f"tpvE1_{step}_swap", "E1, wrong street's cameras")]
W, H = 384, 192; cw, ch = 128, 72
for sc, nm in names.items():
    keys = sorted(glob.glob(f"{C}/{sc}_*_key.png"))
    S = Image.new("RGB", (3 * cw + 8 + len(cols) * W, 24 + len(keys) * H), "white"); d = ImageDraw.Draw(S)
    d.text((4, 6), f"{sc} ({nm}), step {step}", fill="black")
    for j, (_, t) in enumerate(cols):
        d.text((3 * cw + 8 + j * W + 4, 6), t, fill="black")
    for i, kp in enumerate(keys):
        b = os.path.basename(kp)[len(sc) + 1:-8]; k = b[:3]; y = 24 + i * H
        for c_, cam in enumerate(CAMS):
            im = Image.open(f"{C}/{sc}_{b}_{cam}.jpg").convert("RGB").resize((cw, ch))
            S.paste(im, ((c_ % 3) * cw, y + 20 + (c_ // 3) * ch))
        for j, (dd, _) in enumerate(cols):
            if dd == "key":
                im = Image.open(kp).convert("RGB").resize((512, 288)).crop((0, 32, 512, 288))
            else:
                f = f"{P}/{dd}/{sc}/{k}_{b[4:]}_s0.png"
                if not os.path.exists(f):
                    continue
                im = Image.open(f).convert("RGB")
            S.paste(im.resize((W, H)), (3 * cw + 8 + j * W, y))
    S.save(f"{out}_{sc}.jpg", quality=88)
print("ok")
