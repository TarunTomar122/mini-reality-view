"""~10 s of real camera sweeps (~12 Hz) -> chase-view frames, one by one.

  python scripts/video_render.py <ckpt> <scene> <seconds> <out_dir> [warm_strength]

warm_strength 0 (default): every frame starts from the same fixed noise (independent frames).
warm_strength s in (0, 1): from the second frame on, start from the PREVIOUS output latent noised to the point where a
fraction s of the 25 DDIM steps remains (SDEdit-style warm start, no retraining), with the same fixed noise. Lower s
keeps more of the previous frame (steadier, but slower to follow the cameras).
"""
import os, sys, bisect, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from nuscenes.nuscenes import NuScenes
from transformers import AutoModel
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
ck_path, sc, secs, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
WARM = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
os.makedirs(out, exist_ok=True)
nusc = NuScenes("v1.0-trainval", "data/nuscenes_full", verbose=False)
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
scene = [s for s in nusc.scene if s["name"] == sc][0]
s0 = nusc.get("sample", scene["first_sample_token"])
chains = {}
for c in CAMS:
    t = s0["data"][c]; L = []
    while t:
        sd = nusc.get("sample_data", t); L.append((sd["timestamp"], sd["filename"])); t = sd["next"]
    chains[c] = L
front = [x for x in chains["CAM_FRONT"] if x[0] - chains["CAM_FRONT"][0][0] <= secs * 1e6]
dev = "cuda"; AMP = torch.float16
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).half().eval()
SD = "sd-legacy/stable-diffusion-v1-5"
ck = torch.load(ck_path, map_location="cpu")
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=ck["args"]["rank"], lora_alpha=ck["args"]["rank"], target_modules=LORA_TARGETS))
unet.load_state_dict(ck["lora"], strict=False); unet.to(dev, AMP).eval()
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 2)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler")
ry = torch.load("data/calib_rays.pt")[sc][None].to(dev).float()
z0 = torch.randn(1, 4, 32, 64, generator=torch.Generator().manual_seed(0)).to(dev)
st = torch.full((1,), 2, dtype=torch.long, device=dev)
for i, (ts, ffront) in enumerate(front):
    files = []
    for c in CAMS:
        L = chains[c]; j = min(range(len(L)), key=lambda q: abs(L[q][0] - ts)); files.append(L[j][1])
    imgs = [np.asarray(Image.open("data/nuscenes_full/" + f).convert("RGB").resize((448, 252), Image.BICUBIC)) for f in files]
    x = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        h = dino(pixel_values=((x - MEAN) / STD).half()).last_hidden_state[:, 1:]
        tok = F.avg_pool2d(h.float().view(6, 18, 32, 384).permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2)[None]
        with torch.autocast("cuda", dtype=AMP):
            cc = adapter(tok, ry, st); uu = adapter(tok, ry, st, torch.ones(1, dtype=torch.bool, device=dev))
        ddim.set_timesteps(25); ts_all = ddim.timesteps
        if WARM > 0 and i > 0:  # warm start from the previous frame's latent
            k0 = int(round(len(ts_all) * (1 - WARM))); t0 = ts_all[k0]
            ab = ddim.alphas_cumprod[t0].to(dev)
            z = ab.sqrt() * z_prev + (1 - ab).sqrt() * z0; ts_run = ts_all[k0:]
        else:
            z = z0.clone(); ts_run = ts_all
        for t in ts_run:
            with torch.autocast("cuda", dtype=AMP):
                e = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([cc, uu])).sample.float()
            ec, eu = e.chunk(2); z = ddim.step(eu + 2.0 * (ec - eu), t, z).prev_sample
        z_prev = z.clone()
        img = vae.decode(z / 0.18215).sample[0].clamp(-1, 1)
    Image.fromarray(((img + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()).save(f"{out}/{i:04d}_gen.png")
    for c, f in zip(CAMS, files):
        if c in ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT") and not WARM:
            Image.open("data/nuscenes_full/" + f).convert("RGB").resize((640, 360), Image.BICUBIC).save(f"{out}/{i:04d}_{c}.jpg", quality=90)
    print(i, len(front), flush=True)
print("VIDEO_DONE", len(front))
