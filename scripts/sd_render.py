"""Render an SD renderer checkpoint on the geometry-judge set (council/PLAN-sim.md).

Judge set: official val scenes (with tokens in data/enc_full) sorted by name, positions 30-59, i.e. outside the
"final 30" used by evaluate.py. 4 evenly spaced keyframes per scene, 2 seeds, style 1 (splat look), DDIM 25, CFG 2.
--shuffle_cams renders with the six cameras rolled by 3 (front <-> back), the validation control.
Writes <out>/<scene>/<k>_<token>_s<seed>.png (512x256) and <out>/index.json.
"""
import argparse, glob, json, os
import numpy as np
import torch
from PIL import Image
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from nuscenes.utils.splits import val as VAL, train as TRAIN

SD = "sd-legacy/stable-diffusion-v1-5"
p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", required=True)
p.add_argument("--tokens", default="data/enc_full")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--first", type=int, default=30)
p.add_argument("--n_scenes", type=int, default=30)
p.add_argument("--frames", type=int, default=4)
p.add_argument("--seeds", default="0,1")
p.add_argument("--style", type=int, default=1)
p.add_argument("--steps", type=int, default=25)
p.add_argument("--cfg", type=float, default=2.0)
p.add_argument("--split", default="val", choices=["val", "train"], help="scenes to draw from (train = train-frame check)")
p.add_argument("--scenes", default="", help="comma list of scenes (overrides --first/--n_scenes)")
p.add_argument("--scene_list", default="", help="file of scene names to choose from (e.g. E3's quarter)")
p.add_argument("--keys_like", default="", help="game key folder (e.g. data/sim/keys_tpv): use val scenes that have keys there and pick frames among the keyed ones")
p.add_argument("--shuffle_cams", action="store_true", help="roll tokens AND rays by 3 (not a real control: geometry stays consistent)")
p.add_argument("--control", default="", choices=["", "roll_tokens", "swap_scene"],
               help="validation controls: roll_tokens = tokens rolled by 3 cameras, rays kept (front content placed behind); "
                    "swap_scene = tokens + rays of the judge scene half the list away (same frame position), i.e. the wrong street")
a = p.parse_args()
dev = "cuda"
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
ck = torch.load(a.ckpt, map_location="cpu")
ar = ck["args"]
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=ar["rank"], lora_alpha=ar["rank"], target_modules=LORA_TARGETS))
unet.load_state_dict(ck["lora"], strict=False)
unet.to(dev, AMP).eval()
adapter = CameraAdapter(n_styles=ar.get("n_styles", 2)).to(dev).eval()
adapter.load_state_dict(ck["adapter"])
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler")
RAYS = torch.load(a.rays)
SPLIT = set(VAL if a.split == "val" else TRAIN)
if a.scene_list:
    SPLIT &= set(open(a.scene_list).read().split())
names = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{a.tokens}/*.pt") if os.path.basename(f)[:-3] in SPLIT)
if a.keys_like:
    names = [n for n in names if len(glob.glob(f"{a.keys_like}/{n}/*.png")) >= a.frames]
judge = names[a.first:a.first + a.n_scenes]
if a.scenes:
    judge = a.scenes.split(",")
index = []
for si, sc in enumerate(judge):
    e = torch.load(f"{a.tokens}/{sc}.pt")
    if a.control == "swap_scene":
        osc = judge[(si + len(judge) // 2) % len(judge)]; eo = torch.load(f"{a.tokens}/{osc}.pt")
    n = len(e["sample_tokens"])
    ks = np.linspace(2, n - 3, a.frames).round().astype(int).tolist()
    if a.keys_like:  # frames that have a game key
        kk = sorted(int(os.path.basename(f)[:3]) for f in glob.glob(f"{a.keys_like}/{sc}/*.png"))
        ks = [kk[i] for i in np.linspace(0, len(kk) - 1, a.frames).round().astype(int)]
    os.makedirs(f"{a.out}/{sc}", exist_ok=True)
    for k in ks:
        tok = e["tokens"][k:k + 1].to(dev).float()
        ry = RAYS[sc][None].to(dev).float()
        if a.shuffle_cams:
            tok, ry = tok.roll(3, dims=1), ry.roll(3, dims=1)
        if a.control == "roll_tokens":
            tok = tok.roll(3, dims=1)
        if a.control == "swap_scene":
            ko = min(round(k / max(1, n - 1) * (len(eo["tokens"]) - 1)), len(eo["tokens"]) - 1)
            tok, ry = eo["tokens"][ko:ko + 1].to(dev).float(), RAYS[osc][None].to(dev).float()
        st = torch.full((1,), a.style, dtype=torch.long, device=dev)
        for seed in map(int, a.seeds.split(",")):
            g = torch.Generator(device="cpu").manual_seed(1000 * seed + k)
            x = torch.randn(1, 4, 32, 64, generator=g).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
                c = adapter(tok, ry, st); u = adapter(tok, ry, st, torch.ones(1, dtype=torch.bool, device=dev))
            ddim.set_timesteps(a.steps)
            with torch.no_grad():
                for t in ddim.timesteps:
                    with torch.autocast("cuda", dtype=AMP):
                        ep = unet(torch.cat([x, x]), t, encoder_hidden_states=torch.cat([c, u])).sample.float()
                    ec, eu = ep.chunk(2)
                    x = ddim.step(eu + a.cfg * (ec - eu), t, x).prev_sample
                img = vae.decode(x / 0.18215).sample[0].clamp(-1, 1)
            arr = ((img + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            fn = f"{sc}/{k:03d}_{e['sample_tokens'][k]}_s{seed}.png"
            Image.fromarray(arr).save(f"{a.out}/{fn}")
            index.append({"scene": sc, "k": k, "token": e["sample_tokens"][k], "seed": seed, "file": fn})
    print(sc, len(ks), flush=True)
json.dump({"ckpt": a.ckpt, "style": a.style, "shuffle_cams": a.shuffle_cams, "control": a.control, "items": index}, open(f"{a.out}/index.json", "w"), indent=1)
print("RENDER_DONE", len(index))
