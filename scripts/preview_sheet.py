"""Build the experiment-T preview sheets (local): per drive, rows = keyed frames;
columns = 6 real photos | game key | E1 | E2 | E3 | E1 with another drive's cameras.
usage: python scripts/preview_sheet.py <step> <pulled preview dir> <cams dir> <out prefix>"""
import glob, os, sys
from PIL import Image, ImageDraw, ImageFont
F = ImageFont.truetype('/System/Library/Fonts/HelveticaNeue.ttc', 22)
step, P, C, out = sys.argv[1:5]
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
names = {"scene-0915": "busy junction", "scene-0563": "moving traffic", "scene-0632": "quiet street"}
cols = [("key", "answer"), (f"tpvE1_{step}", "E1"), (f"tpvE2_{step}", "E2"),
        (f"tpvE3_{step}", "E3"), (f"tpvE1_{step}_swap", "E1, wrong street")]
CHIP = {"answer": (60, 60, 55), "E1": (42, 120, 214), "E2": (235, 104, 52), "E3": (27, 175, 122), "E1, wrong street": (138, 137, 128)}
W, H = 384, 192; cw, ch = 128, 72
for sc, nm in names.items():
    keys = sorted(glob.glob(f"{C}/{sc}_*_key.png"))
    T = 44
    S = Image.new("RGB", (3 * cw + 8 + len(cols) * W, T + len(keys) * H), "white"); d = ImageDraw.Draw(S)
    def chip(x, t, c):
        w = d.textlength(t, font=F) + 24
        d.rounded_rectangle((x, 8, x + w, 38), radius=15, fill=c); d.text((x + 12, 11), t, font=F, fill="white")
    chip(4, "6 real cameras", (60, 60, 55))
    for j, (_, t) in enumerate(cols):
        chip(3 * cw + 8 + j * W + 6, t, CHIP[t])
    for i, kp in enumerate(keys):
        b = os.path.basename(kp)[len(sc) + 1:-8]; k = b[:3]; y = T + i * H
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
