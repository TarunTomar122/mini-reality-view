"""PLAN-video stage S0a (council/PLAN-video.md): does the Wan2.1 video VAE alone lose cars?
For each dev drive (seed-0 layout of the signed A@20k rollout runs/troll/A20k_indep_dev/<drive>_s0.pt as the tick template):
load the game-render answer keys (same resize / crop as scripts/temporal_metrics.py: 512x288 -> crop to 512x256), and write two
pseudo-rollouts in the signed rollout format (SD-VAE latents, which the signed scorer decodes):
  --arm sd   : key -> SD-VAE encode                       (codec floor of the scorer's own path)
  --arm wan  : key video -> Wan VAE encode / decode (windows of 4k+1 ticks) -> SD-VAE encode
Then score both with the unchanged signed scorer (scripts/temporal_metrics.py --roll <dir>) and compare car F1 / road IoU with
A@20k's (runs/troll/m_A20k_indep_dev.json, seed 0). Kill rule (PLAN-video S0a): Wan round-trip car F1 below A@20k's.
  python scripts/s0_vae_roundtrip.py --arm wan --out runs/s0/wan_rt_dev
"""
import argparse, glob, json, os
import numpy as np, torch
from PIL import Image
from diffusers import AutoencoderKL, AutoencoderKLWan

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["sd", "wan", "wanstill"], required=True); p.add_argument("--out", required=True)
p.add_argument("--template", default="runs/troll/A20k_indep_dev"); p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--wan", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"); p.add_argument("--window", type=int, default=41, help="4k+1 ticks per Wan VAE call; 0 = the whole drive in one call (codec state continuous, padded to 4k+1)")
a = p.parse_args()
assert a.window == 0 or (a.window - 1) % 4 == 0
dev = "cuda"; os.makedirs(a.out, exist_ok=True)
sd = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
wan = AutoencoderKLWan.from_pretrained(a.wan, subfolder="vae", torch_dtype=torch.float32).to(dev).eval() if a.arm in ("wan", "wanstill") else None
to_t = lambda im: torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 127.5 - 1  # [3,H,W] in [-1, 1]


def key_img(sc, t):
    f = f"{a.keys}/{sc}/t{t:04d}.jpg"
    return Image.open(f).convert("RGB").resize((512, 288), Image.BILINEAR).crop((0, 32, 512, 288)) if os.path.exists(f) else None


stats = {}
for tf in sorted(glob.glob(f"{a.template}/*_s0.pt")):
    R = torch.load(tf, map_location="cpu", weights_only=False); sc = os.path.basename(tf).rsplit("_s", 1)[0]
    ticks = R["ticks"].tolist(); imgs = [key_img(sc, t) for t in ticks]
    have = [i for i, im in enumerate(imgs) if im is not None]
    X = torch.stack([to_t(imgs[i]) if imgs[i] is not None else torch.zeros(3, 256, 512) for i in range(len(ticks))])  # [T,3,256,512]
    if a.arm == "wanstill":  # Astra: every tick encoded as an independent still (1-frame video): the per-tick live option
        Y = X.clone()
        with torch.no_grad():
            for i in range(len(ticks)):
                lat = wan.encode(X[i][None, :, None].to(dev)).latent_dist.mode()
                Y[i] = wan.decode(lat).sample[0, :, 0].clamp(-1, 1).cpu()
        src = Y
    elif a.arm == "wan":
        Y = X.clone(); T = len(ticks); s = 0
        W = a.window if a.window else T + (-(T - 1)) % 4  # whole drive padded up to 4k+1
        with torch.no_grad():
            while s < T:  # consecutive windows of 4k+1 frames; the tail window is padded by repeating the last frame
                e = min(s + W, T); seg = X[s:e]
                pad = W - seg.shape[0]
                if pad: seg = torch.cat([seg, seg[-1:].expand(pad, -1, -1, -1)])
                v = seg.permute(1, 0, 2, 3)[None].to(dev)  # [1,3,F,H,W]
                lat = wan.encode(v).latent_dist.mode()
                rec = wan.decode(lat).sample[0].permute(1, 0, 2, 3).clamp(-1, 1).cpu()
                Y[s:e] = rec[: e - s]; s = e
        src = Y
    else:
        src = X
    lats = []
    with torch.no_grad():
        for i in range(0, len(ticks), 16):
            lats.append((sd.encode(src[i:i + 16].to(dev)).latent_dist.mode() * 0.18215).half().cpu())
    out = dict(R); out["lat"] = torch.cat(lats); out["method"] = f"s0_{a.arm}_roundtrip"
    out["s0"] = {"arm": a.arm, "wan": a.wan if a.arm != "sd" else None, "window": a.window, "keys_found": len(have), "ticks": len(ticks)}
    if a.arm != "sd":
        err = (Y[have] - X[have]).abs().mean().item(); out["s0"]["wan_l1_vs_key"] = err
    torch.save(out, f"{a.out}/{sc}_s0.pt"); stats[sc] = out["s0"]; print(sc, out["s0"], flush=True)
json.dump(stats, open(f"{a.out}/s0_stats.json", "w"), indent=1)
print("S0_DONE", a.arm, a.out)
