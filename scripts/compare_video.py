"""Side-by-side MP4 of saved 10 Hz rollouts (owner: "when can i see video of these new tries").
For one drive / seed: game key | each rollout folder given (decoded latents), 10 fps, labels burned in.

  python scripts/compare_video.py --scene scene-0234 --seed 0 --rolls "A 4k:runs/troll/A4k_indep_dev" "warm 0.7:runs/troll/warm07_dev" --out pulled/v.mp4
"""
import argparse, os, subprocess, tempfile
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL
p = argparse.ArgumentParser()
p.add_argument("--scene", required=True); p.add_argument("--seed", default="0")
p.add_argument("--rolls", nargs="+", required=True); p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--max_ticks", type=int, default=200); p.add_argument("--out", required=True)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
R = [(lab, torch.load(f"{d}/{a.scene}_s{a.seed}.pt")) for lab, d in (x.split(":", 1) for x in a.rolls)]
ticks = R[0][1]["ticks"].tolist()[: a.max_ticks]
try: font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception: font = ImageFont.load_default()
tmp = tempfile.mkdtemp(); W, H = 512, 256
for i, t in enumerate(ticks):
    panels = [("game key (answer)", Image.open(f"{a.keys}/{a.scene}/t{t:04d}.jpg").convert("RGB").resize((512, 288)).crop((0, 32, 512, 288)))]
    for lab, r in R:
        if i >= len(r["lat"]) or (r.get("hard") is not None and bool(r["hard"][i])):
            panels.append((lab + " (no frame)", Image.new("RGB", (W, H)))); continue
        with torch.no_grad():
            x = vae.decode(r["lat"][i][None].float().to(dev) / 0.18215).sample[0].clamp(-1, 1)
        panels.append((lab, Image.fromarray(((x + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())))
    cols = 2; rows = (len(panels) + 1) // 2
    sheet = Image.new("RGB", (W * cols, H * rows)); d = ImageDraw.Draw(sheet)
    for k, (lab, im) in enumerate(panels):
        x0, y0 = (k % cols) * W, (k // cols) * H; sheet.paste(im, (x0, y0))
        d.rectangle([x0, y0, x0 + 12 * len(lab) + 10, y0 + 28], fill=(0, 0, 0)); d.text((x0 + 6, y0 + 4), lab, fill="white", font=font)
    d.text((8, H * rows - 26), f"{a.scene}  tick {t}  ({t / 10:.1f} s)", fill="white", font=font)
    sheet.save(f"{tmp}/f{i:04d}.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", f"{tmp}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.out], check=True)
print("VIDEO", a.out, len(ticks), "frames")
