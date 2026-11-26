"""Build the live project page (docs/index.html, served by GitHub Pages from /docs).

Everything the page shows is baked into the HTML, so it also works opened straight from disk:
  - training curves from pulled/site/tpvE*.jsonl (copies of runs/tpvE*/log.jsonl)
  - scores from pulled/site/scores/*.json (copies of runs/judge/tpv*_m2f.json), if any
  - preview sheets pulled/prev<step>_<scene>.jpg -> docs/site/img/ (resized)
  - the short news list in scripts/site_news.json
usage: python3 scripts/build_site.py
"""
import glob, json, os, re, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
STEPS_TOTAL = 10000
RUNS = [("E1", "our model", "all drives"),
        ("E2", "plain Stable Diffusion", "all drives"),
        ("E3", "our model", "¼ of the drives")]
DRIVES = [("scene-0915", "Busy junction", "rain, a left turn at a crossing, 26 vehicles around"),
          ("scene-0563", "Moving traffic", "queue at a traffic light, 10 of 20 vehicles moving"),
          ("scene-0632", "Quiet street", "industrial, 2 vehicles")]

curves = {}
for r, _, _ in RUNS:
    rows = [json.loads(l) for l in open(P("pulled", "site", f"tpv{r}.jsonl"))] if os.path.exists(P("pulled", "site", f"tpv{r}.jsonl")) else []
    curves[r] = [[d["step"], round(d["loss"], 4)] for d in rows]

scores = {}
for f in glob.glob(P("pulled", "site", "scores", "*.json")):
    s = json.load(open(f))["summary"]
    scores[os.path.basename(f)[:-5]] = {k: s[k] for k in ("f1", "road_iou", "lpips", "frames") if k in s}

os.makedirs(P("docs", "site", "img"), exist_ok=True)
pics = {}
for f in sorted(glob.glob(P("pulled", "prev*_scene-*.jpg"))):
    m = re.match(r"prev(\d+)_(scene-\d+)\.jpg", os.path.basename(f))
    if not m:
        continue
    step, sc = int(m.group(1)), m.group(2)
    dst = P("docs", "site", "img", f"step{step}_{sc}.jpg")
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(f):
        subprocess.run(["sips", "-Z", "1800", "-s", "formatOptions", "80", f, "--out", dst], capture_output=True)
    pics.setdefault(step, {})[sc] = f"site/img/step{step}_{sc}.jpg"

# hero: front camera | game answer | E1's drawing, for the moving-traffic drive at the latest preview step
hero = None
if pics:
    from PIL import Image, ImageDraw
    st = max(pics); sc, k = "scene-0563", "010"
    cam = glob.glob(P("pulled", "prev_cams", f"{sc}_{k}_*_CAM_FRONT.jpg"))
    key = glob.glob(P("pulled", "prev_cams", f"{sc}_{k}_*_key.png"))
    ren = glob.glob(P("pulled", "preview", f"tpvE1_{st}", sc, f"{k}_*_s0.png"))
    if cam and key and ren:
        h = 300; out = []
        for src, name, crop in ((cam[0], "cam", None), (key[0], "key", (0, 32, 512, 288)), (ren[0], "e1", None)):
            im = Image.open(src).convert("RGB")
            im = im.crop(crop) if crop else im
            if name == "cam":  # same 2:1 shape as the other two panels
                w, hh = im.size; nh = w // 2; im = im.crop((0, (hh - nh) // 2, w, (hh - nh) // 2 + nh))
            im.resize((h * 2, h)).save(P("docs", "site", "img", f"hero_{name}.jpg"), quality=85)
            out.append(f"site/img/hero_{name}.jpg")
        hero = {"cam": out[0], "key": out[1], "e1": out[2], "step": st}

news = json.load(open(P("scripts", "site_news.json")))
data = {"updated": time.strftime("%d %b %Y, %H:%M"), "total": STEPS_TOTAL, "runs": RUNS, "drives": DRIVES,
        "curves": curves, "scores": scores, "hero": hero, "pics": {str(k): v for k, v in sorted(pics.items())}, "news": news}
html = open(P("scripts", "site_template.html")).read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
open(P("docs", "index.html"), "w").write(html)
print("wrote docs/index.html", len(html) // 1024, "KB;", {k: len(v) for k, v in curves.items()}, "pictures at steps", sorted(pics))
