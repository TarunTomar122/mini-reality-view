"""Phone-photo test of the final renderer on the geometry-judge frames.

For every judge frame (same list as a renders folder made by sd_render.py, seed 0 only) the six real nuScenes camera
pictures are turned into "phone photos" exactly as scripts/try_my_photos.py would see them: a random crop / zoom
(1-1.3x), colour, gamma, hue, blur and JPEG damage (mrv/aug.py), then centre-cropped to 16:9 and resized to 448x252.
They are encoded with DINOv2 and rendered with the DEFAULT rig rays (assets/default_rays.pt: no calibration), DDIM 25,
CFG 2, style 1 - the same path as try_my_photos.py. Output: a renders folder that judge_score.py can score.

  python scripts/phone_test.py --ckpt runs/sd_run1/ckpt.pt --like runs/judge/sd40k --out runs/judge/sd40k_phone
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--like", required=True)
p.add_argument("--out", required=True)
p.add_argument("--dataroot", default="data/nuscenes_full")
p.add_argument("--rays", default="assets/default_rays.pt")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
dev = "cuda"
from nuscenes.nuscenes import NuScenes
from transformers import AutoModel
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.aug import augment

nusc = NuScenes("v1.0-trainval", a.dataroot, verbose=False)
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
SD = "sd-legacy/stable-diffusion-v1-5"
ck = torch.load(a.ckpt, map_location="cpu")
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=ck["args"]["rank"], lora_alpha=ck["args"]["rank"], target_modules=LORA_TARGETS))
unet.load_state_dict(ck["lora"], strict=False); unet.to(dev, AMP).eval()
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 2)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler")
rays = torch.load(a.rays, map_location=dev)[None].float()
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def to169(im):  # the centre crop try_my_photos.py does
    W, H = im.size
    if W / H > 16 / 9:
        nw = int(H * 16 / 9); im = im.crop(((W - nw) // 2, 0, (W - nw) // 2 + nw, H))
    else:
        nh = int(W * 9 / 16); im = im.crop((0, (H - nh) // 2, W, (H - nh) // 2 + nh))
    return np.asarray(im.resize((448, 252), Image.BICUBIC))


idx = [it for it in json.load(open(f"{a.like}/index.json"))["items"] if it["seed"] == 0]
os.makedirs(a.out, exist_ok=True)
items = []
rng = np.random.default_rng(a.seed)
for n, it in enumerate(idx):
    smp = nusc.get("sample", it["token"])
    pics = []
    for c in CAMS:
        img = Image.open(os.path.join(a.dataroot, nusc.get("sample_data", smp["data"][c])["filename"])).convert("RGB")
        ph, _ = augment(img, rng)
        pics.append(to169(Image.fromarray(ph)))
    x = torch.from_numpy(np.stack(pics)).to(dev).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        h = dino(pixel_values=(x - mean) / std).last_hidden_state[:, 1:]
        tok = F.avg_pool2d(h.view(6, 18, 32, 384).permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2)[None].float()
        st = torch.ones(1, dtype=torch.long, device=dev)
        with torch.autocast("cuda", dtype=AMP):
            c = adapter(tok, rays, st); u = adapter(tok, rays, st, torch.ones(1, dtype=torch.bool, device=dev))
        g = torch.Generator(device="cpu").manual_seed(it["k"])  # same noise as sd_render.py seed 0
        z = torch.randn(1, 4, 32, 64, generator=g).to(dev)
        ddim.set_timesteps(25)
        for t in ddim.timesteps:
            with torch.autocast("cuda", dtype=AMP):
                e = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([c, u])).sample.float()
            ec, eu = e.chunk(2)
            z = ddim.step(eu + 2.0 * (ec - eu), t, z).prev_sample
        img = vae.decode(z / 0.18215).sample[0].clamp(-1, 1)
    os.makedirs(f"{a.out}/{it['scene']}", exist_ok=True)
    Image.fromarray(((img + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(f"{a.out}/{it['file']}")
    if n < 6:  # keep a few phone-style inputs for the README picture
        Image.fromarray(np.concatenate([np.concatenate(pics[:3], 1), np.concatenate(pics[3:], 1)], 0)).save(f"{a.out}/{it['scene']}/{it['k']:03d}_phone_inputs.jpg")
    items.append(it)
    print(n, it["scene"], flush=True)
json.dump({"ckpt": a.ckpt, "phone": True, "items": items}, open(f"{a.out}/index.json", "w"), indent=1)
print("PHONE_TEST_DONE", len(items))
