"""Score RGB frames DIRECTLY with the UNCHANGED signed scorer (scripts/temporal_metrics.py, 764dd3ec95db) — no SD-VAE re-encode
(Astra 06:30: the S0a SD re-encode adds a second codec). The scorer decodes rollout latents with AutoencoderKL.decode; here
AutoencoderKL.from_pretrained is replaced (for this process only) by a stand-in whose decode() returns the stored RGB frame whose
index is written into the pseudo-rollout latent (lat[t] = t + 1 at [0, 0, 0], zeros elsewhere). Everything else (segmentation,
matching, ID / W / F1 / road / ghosts) runs from the signed file as is. Input: <rgb_dir>/<scene>_s0.pt with "rgb" uint8 [T,3,256,512]
plus the template rollout fields (ticks / ok / hard ...), as written by scripts/s0_vae_roundtrip.py --save_rgb.
  python scripts/score_rgb.py --rgb runs/s0/wanfull_rgb --out runs/s0/m_wanfull_rgb.json
"""
import argparse, glob, os, runpy, sys
import torch, diffusers

p = argparse.ArgumentParser(); p.add_argument("--rgb", required=True); p.add_argument("--out", required=True)
a = p.parse_args()
roll = a.rgb.rstrip("/") + "_roll"; os.makedirs(roll, exist_ok=True)
TABLE = {}
for f in sorted(glob.glob(f"{a.rgb}/*_s0.pt")):
    R = torch.load(f, map_location="cpu", weights_only=False); sc = os.path.basename(f)[:-3]
    rgb = R.pop("rgb"); T = rgb.shape[0]; base = len(TABLE)
    lat = torch.zeros(T, 4, 32, 64)
    for t in range(T):
        TABLE[base + t + 1] = rgb[t]; lat[t, 0, 0, 0] = base + t + 1
    R["lat"] = lat; R["method"] = R.get("method", "") + "+rgb_direct"
    torch.save(R, f"{roll}/{sc}.pt")


class _Out:
    def __init__(self, s): self.sample = s


class FakeVAE(torch.nn.Module):
    """decode(z) -> the stored RGB frame (in [-1, 1]) whose index is coded in z (scorer passes lat / 0.18215)."""
    def decode(self, z):
        idx = int(round(float(z[0, 0, 0, 0]) * 0.18215))
        x = TABLE[idx].to(z.device).float() / 127.5 - 1
        return _Out(x[None])


diffusers.AutoencoderKL.from_pretrained = classmethod(lambda cls, *args, **kw: FakeVAE())
sys.argv = ["scripts/temporal_metrics.py", "--roll", roll, "--out", a.out]
runpy.run_path("scripts/temporal_metrics.py", run_name="__main__")
