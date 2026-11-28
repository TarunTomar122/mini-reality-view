"""PLAN-video stage S0b: timing on the current GPU (run once on a 2080 Ti and once on the H200 slice).
- Wan2.1-T2V-1.3B transformer, one denoising call (batch 1, CFG off; CFG 2 = x2 rows), latent [1, 16, F, 32, 64] for a 512x256
  output (8x spatial), F in {1 (per-tick still), 2, 5 (17-tick clip)}, with a stand-in conditioning sequence of 256 tokens x 4096
  (text-embedding width; the real camera tokens replace it later). Reports ms per call (p50 over 20 after 3 warm-up), peak GB,
  and whether outputs are finite (fp16 on Turing: Wan was trained in bf16).
- Wan VAE decode of one latent frame (= 4 ticks, or 1 tick for the first / still) and encode of a 5-frame chunk.
- Depth Anything V2-Small (metric outdoor + plain): 6 camera views per tick, batch 6, at 518x294-ish (model's own processor).
Prints one JSON line "S0B_TIMING {...}".
  python scripts/s0b_timing.py
"""
import json, time
import torch
from diffusers import WanTransformer3DModel, AutoencoderKLWan
from transformers import AutoModelForDepthEstimation

dev = "cuda"; W = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
bf16 = torch.cuda.is_bf16_supported(including_emulation=False); DT = torch.bfloat16 if bf16 else torch.float16
res = {"gpu": torch.cuda.get_device_name(), "dtype": str(DT)}


def timed(fn, n=20, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    ts.sort(); return ts[len(ts) // 2], out


tf = WanTransformer3DModel.from_pretrained(W, subfolder="transformer", torch_dtype=DT).to(dev).eval()
res["transformer_params_M"] = round(sum(p.numel() for p in tf.parameters()) / 1e6, 1)
ctx = torch.randn(1, 256, 4096, device=dev, dtype=DT)
for F in (1, 2, 5):
    for rows in (1, 2):
        x = torch.randn(rows, 16, F, 32, 64, device=dev, dtype=DT); t = torch.full((rows,), 500, device=dev)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            ms, out = timed(lambda: tf(hidden_states=x, timestep=t, encoder_hidden_states=ctx.expand(rows, -1, -1), return_dict=False)[0])
        res[f"tf_F{F}_rows{rows}"] = {"ms": round(ms, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "finite": bool(torch.isfinite(out).all())}
        print(F, rows, res[f"tf_F{F}_rows{rows}"], flush=True)
del tf; torch.cuda.empty_cache()

vae = AutoencoderKLWan.from_pretrained(W, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
with torch.no_grad():
    z1 = torch.randn(1, 16, 1, 32, 64, device=dev); z2 = torch.randn(1, 16, 2, 32, 64, device=dev)
    res["vae_decode_1latent_ms"] = round(timed(lambda: vae.decode(z1).sample, n=10)[0], 1)
    res["vae_decode_2latents_ms"] = round(timed(lambda: vae.decode(z2).sample, n=10)[0], 1)
    v5 = torch.randn(1, 3, 5, 256, 512, device=dev)
    res["vae_encode_5frames_ms"] = round(timed(lambda: vae.encode(v5).latent_dist.mode(), n=10)[0], 1)
del vae; torch.cuda.empty_cache()

for name, rid in (("depth_metric_outdoor_small", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"), ("depth_small", "depth-anything/Depth-Anything-V2-Small-hf")):
    m = AutoModelForDepthEstimation.from_pretrained(rid, torch_dtype=torch.float16).to(dev).eval()
    x = torch.randn(6, 3, 294, 518, device=dev, dtype=torch.float16)  # 6 views, multiple of 14
    with torch.no_grad():
        res[name + "_6views_ms"] = round(timed(lambda: m(pixel_values=x).predicted_depth)[0], 1)
    del m; torch.cuda.empty_cache()
print("S0B_TIMING", json.dumps(res), flush=True)
