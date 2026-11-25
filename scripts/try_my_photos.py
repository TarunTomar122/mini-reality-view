"""Draw the drone view from 6 of your own photos.

  python scripts/try_my_photos.py --ckpt runs/sd_final/ckpt.pt \
      --front f.jpg --front_left fl.jpg --front_right fr.jpg --back_left bl.jpg --back b.jpg --back_right br.jpg \
      --out drone.png

Each photo is centre-cropped to 16:9, resized to 448x252 and turned into DINOv2 tokens. With no calibration we assume
the standard nuScenes 6-camera rig (assets/default_rays.pt: average viewing rays over all 850 scenes), so stand still,
hold the phone level at about chest height, and turn to face each direction (see README: "Try it on your own photos").
Runs on a CUDA GPU, Apple MPS or CPU (slow).
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
for c in ["front", "front_left", "front_right", "back_left", "back", "back_right"]:
    p.add_argument(f"--{c}", required=True)
p.add_argument("--out", default="drone.png")
p.add_argument("--rays", default="assets/default_rays.pt")
p.add_argument("--steps", type=int, default=25)
p.add_argument("--cfg", type=float, default=2.0)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dt = torch.float16 if dev != "cpu" else torch.float32
# nuScenes camera order used everywhere in this project
ORDER = ["front_left", "front", "front_right", "back_left", "back", "back_right"]


def load(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    W, H = im.size
    if W / H > 16 / 9:
        nw = int(H * 16 / 9); im = im.crop(((W - nw) // 2, 0, (W - nw) // 2 + nw, H))
    else:
        nh = int(W * 9 / 16); im = im.crop((0, (H - nh) // 2, W, (H - nh) // 2 + nh))
    return np.asarray(im.resize((448, 252), Image.BICUBIC))


from transformers import AutoModel
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
x = torch.from_numpy(np.stack([load(getattr(a, c)) for c in ORDER])).to(dev).permute(0, 3, 1, 2).float() / 255
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
with torch.no_grad():
    h = dino(pixel_values=(x - mean) / std).last_hidden_state[:, 1:]
    h = F.avg_pool2d(h.view(6, 18, 32, 384).permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2)
tok = h[None].float()  # (1, 6, 144, 384)
rays = torch.load(a.rays, map_location=dev)[None].float()

ck = torch.load(a.ckpt, map_location="cpu")
if "adapter" in ck:  # Stable Diffusion 1.5 renderer (train_sd.py)
    from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
    from peft import LoraConfig
    from mrv.sd import CameraAdapter, LORA_TARGETS
    SD = "sd-legacy/stable-diffusion-v1-5"
    r = ck["args"]["rank"]
    unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
    unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS))
    unet.load_state_dict(ck["lora"], strict=False)
    unet.to(dev, dt).eval()
    adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 2)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
    ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); ddim.set_timesteps(a.steps)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    z = torch.randn(1, 4, 32, 64, generator=g).to(dev)
    st = torch.ones(1, dtype=torch.long, device=dev)
    with torch.no_grad():
        c = adapter(tok, rays, st); u = adapter(tok, rays, st, torch.ones(1, dtype=torch.bool, device=dev))
        for t in ddim.timesteps:
            e = unet(torch.cat([z, z]).to(dt), t, encoder_hidden_states=torch.cat([c, u]).to(dt)).sample.float()
            ec, eu = e.chunk(2)
            z = ddim.step(eu + a.cfg * (ec - eu), t, z).prev_sample
        img = vae.decode(z / 0.18215).sample
else:  # from-scratch DiT renderer (train.py), small enough for a laptop
    from mrv.model import Renderer
    from diffusers import AutoencoderKL
    r = ck["args"]
    m = Renderer(d=r["d"], depth=r["depth"], heads=r["heads"], prev=r["prev"], n_styles=2 if r.get("enc2") else 0,
                 rays=bool(r.get("rays"))).to(dev).eval()
    m.load_state_dict(ck["ema"])
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    z = torch.randn(1, 4, 18, 32, generator=g).to(dev)
    kw = {"style": torch.ones(1, dtype=torch.long, device=dev)} if r.get("enc2") else {}
    if r["prev"]:
        kw.update(prev=torch.randn_like(z), s_noise=torch.ones(1, device=dev))
    ts = torch.linspace(1, 0, a.steps + 1, device=dev)
    with torch.no_grad():
        ctx = m.context(tok, rays if r.get("rays") else None)
        for i in range(a.steps):
            z = z + (ts[i + 1] - ts[i]) * m(z, ts[i].expand(1), None, ctx=ctx, **kw)
        img = vae.decode(z * ck["lat_std"] / 0.18215).sample
out = ((img[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
Image.fromarray(out).save(a.out)
print("saved", a.out, out.shape)
