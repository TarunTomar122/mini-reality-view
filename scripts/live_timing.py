"""Throwaway live-latency measurement (council review 18:30, Fable + Astra): end-to-end time per frame of frozen A@20k on the
current GPU, batch one: 6 camera images (uint8 on GPU, 448x252) -> DINOv2-small fp16 -> 2x2 pool (144 tokens/camera) ->
CameraAdapter -> UNet x N DDIM steps (CFG 2 = 2 rows, or no CFG = 1 row) -> VAE decode 512x256. Reports p50 / p95 ms and
fps over 50 timed frames after 5 warm-up frames, for N in {25, 8, 4} x {CFG 2, no CFG}. Random camera pixels (timing only;
quality is a separate test). Also reports the share of time in each stage at N = 25 with CFG.

  python scripts/live_timing.py --ckpt runs/longA/ckpt_20000.pt
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from peft import LoraConfig
from transformers import AutoModel
from mrv.sd import CameraAdapter, LORA_TARGETS

p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--frames", type=int, default=50)
a = p.parse_args()
dev = "cuda"; SD = "sd-legacy/stable-diffusion-v1-5"
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
ck = torch.load(a.ckpt, map_location="cpu"); r = ck["args"]["rank"]
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); unet.load_state_dict(ck["lora"], strict=False)
unet.to(dev, AMP).eval()
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).half().eval()
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev, AMP).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
rays = torch.load("data/calib_rays.pt"); ray = next(iter(rays.values()))[None].to(dev).float()
st = torch.full((1,), 2, dtype=torch.long, device=dev)
imgs = torch.randint(0, 255, (6, 252, 448, 3), dtype=torch.uint8, device=dev)
sync = torch.cuda.synchronize


def frame(n_steps, cfg, stages=None):
    ddim = SCHED[n_steps]; T = [time.time()]
    with torch.no_grad():
        x = ((imgs.permute(0, 3, 1, 2).float() / 255 - MEAN) / STD).half()
        h = dino(pixel_values=x).last_hidden_state[:, 1:]  # (6, 576, 384)
        tok = F.avg_pool2d(h.view(6, 18, 32, 384).permute(0, 3, 1, 2).float(), 2).flatten(2).transpose(1, 2)[None]  # (1, 6, 144, 384)
        sync(); T.append(time.time())
        with torch.autocast("cuda", dtype=AMP):
            c = adapter(tok, ray, st)
            u = U_NULL
        z = torch.randn(1, 4, 32, 64, device=dev)
        sync(); T.append(time.time())
        for t in ddim.timesteps:
            with torch.autocast("cuda", dtype=AMP):
                if cfg:
                    e = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([c, u])).sample.float()
                    ec, eu = e.chunk(2); e = eu + 2.0 * (ec - eu)
                else:
                    e = unet(z, t, encoder_hidden_states=c).sample.float()
            z = ddim.step(e, t, z).prev_sample
        sync(); T.append(time.time())
        img = vae.decode((z / 0.18215).to(AMP)).sample.clamp(-1, 1)
        sync(); T.append(time.time())
    if stages is not None:
        stages.append(np.diff(T))
    return T[-1] - T[0]


with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
    U_NULL = adapter(torch.zeros(1, 6, 144, 384, device=dev), ray, st, torch.ones(1, dtype=torch.bool, device=dev))
SCHED = {}
for n in (25, 8, 4):
    s_ = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); s_.set_timesteps(n); SCHED[n] = s_
res = {"gpu": torch.cuda.get_device_name(0), "amp": str(AMP)}
for n in (25, 8, 4):
    for cfg in (True, False):
        for _ in range(5):
            frame(n, cfg)
        torch.cuda.reset_peak_memory_stats(); stg = []
        ts = np.array([frame(n, cfg, stg) for _ in range(a.frames)]) * 1000
        key = f"ddim{n}_{'cfg2' if cfg else 'nocfg'}"
        res[key] = {"p50_ms": round(float(np.median(ts)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1),
                    "fps": round(1000 / float(np.mean(ts)), 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "stage_ms": dict(zip(["camera_encoder", "adapter", "unet_loop", "vae_decode"], [round(float(v) * 1000, 1) for v in np.median(np.array(stg), 0)]))}
        print(key, res[key], flush=True)
print("LIVE_TIMING", json.dumps(res), flush=True)
