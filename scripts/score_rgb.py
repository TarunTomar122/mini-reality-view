"""Score RGB frames DIRECTLY with the UNCHANGED signed scorer (scripts/temporal_metrics.py, 764dd3ec95db) — no SD-VAE re-encode
(Astra 06:30: the S0a SD re-encode adds a second codec). The scorer decodes rollout latents with AutoencoderKL.decode; here
AutoencoderKL.from_pretrained is replaced (for this process only, AFTER every real decode below) by a stand-in whose decode()
returns the stored RGB frame whose index is written into the pseudo-rollout latent (lat[t] = index at [0, 0, 0], zeros
elsewhere; float32, exact). Everything else (segmentation, matching, ID / W / F1 / road / ghosts / lag) runs from the signed file.
Input: <rgb_dir>/<scene>_s<seed>.pt with "rgb" uint8 [T,3,256,512] plus the template rollout fields (ticks / ok / hard ...), as
written by scripts/s0_vae_roundtrip.py --save_rgb or scripts/wan_rollout.py; every seed.

Extensions (Astra item 9):
  --lag_ref DIR / --cold DIR : SD-latent reference rollouts (e.g. the signed runs/troll/A20k_indep_dev) are decoded HERE with the
      real SD VAE, exactly as the scorer's to_img does (fp32 decode, clamp, ((x + 1) * 127.5).round().byte()), registered in the
      same index table and written as pseudo-rollouts; the scorer then gets those folders, so swaps (lag vs the donor's cold
      reference) and drift can be scored without the stand-in misreading a real latent.
  --from_lat DIR : score SD-latent rollouts through this RGB path (A@20k replay: with --parity_ref <signed metrics json> every
      per-run field must equal the signed result; exit 1 otherwise).
  --as_receiver  : strip the swap (swap_from = -1, src_ticks = ticks): a swapped render scored against the RECEIVER's keys.
  --post_swap_only : keep only ticks from swap_from on; with --as_receiver vs the receiver keys, else vs the DONOR keys (written
      under the donor's scene name with the donor's ticks, no lag test), for the donor-following check of scripts/wan_panel.py;
      only positions whose receiver AND donor ticks are placement ticks (% 5 == 0) with both keys present stay accepted, so the
      two scores use identical frames ("common_frames" recorded).
Pseudo-rollouts go to a FRESH directory (<rgb>_roll_<tag>_<pid>_<time>, refused if it exists), never reused.
  python scripts/score_rgb.py --rgb runs/wroll/s1_12k_hybrid_dev --out runs/wroll/m_s1_12k_hybrid_dev.json
  python scripts/score_rgb.py --rgb runs/wroll/s1_12k_swap_dev --lag_ref runs/troll/A20k_indep_dev --out runs/wroll/m_s1_12k_swap_dev.json
  python scripts/score_rgb.py --from_lat runs/troll/A20k_indep_dev --parity_ref runs/troll/m_A20k_indep_dev.json --out runs/wroll/m_A20k_rgbpath.json
"""
import argparse, glob, json, math, os, runpy, sys, time
import torch, diffusers

p = argparse.ArgumentParser()
p.add_argument("--rgb", default=""); p.add_argument("--from_lat", default=""); p.add_argument("--out", required=True)
p.add_argument("--lag_ref", default=""); p.add_argument("--cold", default="")
p.add_argument("--as_receiver", action="store_true"); p.add_argument("--post_swap_only", action="store_true")
p.add_argument("--parity_ref", default=""); p.add_argument("--tag", default="")
p.add_argument("--keys", default="data/sim/keys_tpv10", help="key folder (post-swap common-frame selection only)")
a = p.parse_args()
if bool(a.rgb) == bool(a.from_lat):
    raise SystemExit("give exactly one of --rgb / --from_lat")
SRC = (a.rgb or a.from_lat).rstrip("/")
tag = a.tag or ("recv" if a.as_receiver else "") + ("post" if a.post_swap_only else "") or "main"
ROOT = f"{SRC}_roll_{tag}_{os.getpid()}_{int(time.time())}"
os.makedirs(ROOT, exist_ok=False)
TABLE = {}
_real = None


def real_vae():
    global _real
    if _real is None:
        _real = diffusers.AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True).to("cuda").eval()
    return _real


@torch.no_grad()
def decode_like_scorer(lat):
    """[T,4,32,64] SD latents -> uint8 [T,3,256,512], the arithmetic of temporal_metrics.to_img (one tick at a time, fp32)"""
    v = real_vae(); out = []
    for t in range(lat.shape[0]):
        x = v.decode(lat[t][None].to("cuda").float() / 0.18215).sample[0].clamp(-1, 1)
        out.append(((x + 1) * 127.5).round().byte().cpu())
    return torch.stack(out)


def register(R, rgb, dst):
    T = rgb.shape[0]; base = len(TABLE)
    lat = torch.zeros(T, 4, 32, 64)
    for t in range(T):
        TABLE[base + t + 1] = rgb[t]; lat[t, 0, 0, 0] = base + t + 1
    R["lat"] = lat; R["method"] = R.get("method", "") + "+rgb_direct"
    torch.save(R, dst)


PER_TICK = ("ticks", "src_ticks", "ok", "hard", "restart")
roll = f"{ROOT}/roll"; os.makedirs(roll)
for f in sorted(glob.glob(f"{SRC}/*_s[0-9]*.pt")):
    R = torch.load(f, map_location="cpu", weights_only=False); b = os.path.basename(f)[:-3]; sc, seed = b.rsplit("_s", 1)
    rgb = R.pop("rgb") if "rgb" in R else decode_like_scorer(R["lat"])
    sf = int(R.get("swap_from", -1))
    if a.post_swap_only:
        if sf < 0:
            raise SystemExit(f"{b}: --post_swap_only needs a swapped rollout")
        # S1b item 8: the donor-keyed and receiver-keyed scores must use the SAME frames: keep as accepted only positions whose
        # receiver tick AND donor tick are placement ticks (% 5 == 0) with both key files present; everything else ok = False
        rt_, dt_ = R["ticks"].tolist(), R["src_ticks"].tolist()
        common = [i >= sf and int(rt_[i]) % 5 == 0 and int(dt_[i]) % 5 == 0 and bool(R["ok"][i]) and not bool(R["hard"][i])
                  and os.path.exists(f"{a.keys}/{sc}/t{int(rt_[i]):04d}.jpg") and os.path.exists(f"{a.keys}/{R['other']}/t{int(dt_[i]):04d}.jpg")
                  for i in range(len(rt_))]
        R["ok"] = torch.tensor(common, dtype=torch.bool)
        R["common_frames"] = int(sum(common))
        for k in PER_TICK:
            if k in R and torch.is_tensor(R[k]):
                R[k] = R[k][sf:].clone()
        rgb = rgb[sf:]
        if "restart" in R:
            R["restart"][0] = True
        if not a.as_receiver:  # score vs the DONOR keys: the donor's scene name and ticks, no lag test
            R["ticks"] = R["src_ticks"].clone(); sc = R["other"]
        R["swap_from"] = -1; R["other"] = ""; R.pop("src_ticks", None)
    elif a.as_receiver and sf >= 0:
        R["swap_from"] = -1; R["other"] = ""; R.pop("src_ticks", None)
    register(R, rgb, f"{roll}/{sc}_s{seed}.pt")


def ref_dir(src_dir, name):
    """decode an SD-latent reference folder with the real VAE and register it as pseudo-rollouts"""
    d = f"{ROOT}/{name}"; os.makedirs(d)
    for f in sorted(glob.glob(f"{src_dir.rstrip('/')}/*_s[0-9]*.pt")):
        R = torch.load(f, map_location="cpu", weights_only=False)
        register(R, decode_like_scorer(R["lat"]), f"{d}/{os.path.basename(f)}")
    return d


argv = ["scripts/temporal_metrics.py", "--roll", roll, "--out", a.out]
if a.lag_ref:
    argv += ["--lag_ref", ref_dir(a.lag_ref, "lag_ref")]
if a.cold:
    argv += ["--cold", ref_dir(a.cold, "cold")]


class _Out:
    def __init__(self, s): self.sample = s


class FakeVAE(torch.nn.Module):
    """decode(z) -> the stored RGB frame (in [-1, 1]) whose index is coded in z (scorer passes lat / 0.18215)."""
    def decode(self, z):
        idx = int(round(float(z[0, 0, 0, 0]) * 0.18215))
        if idx not in TABLE:
            raise SystemExit(f"stand-in decoder got index {idx}: a latent that was not registered here (real SD latent?)")
        return _Out((TABLE[idx].to(z.device).float() / 127.5 - 1)[None])


_real = None; torch.cuda.empty_cache()
diffusers.AutoencoderKL.from_pretrained = classmethod(lambda cls, *args, **kw: FakeVAE())
sys.argv = argv
runpy.run_path("scripts/temporal_metrics.py", run_name="__main__")

if a.parity_ref:  # A@20k replay: every per-run field equal to the signed metrics
    mine, ref = json.load(open(a.out))["drives"], json.load(open(a.parity_ref))["drives"]
    bad = []
    if set(mine) != set(ref):
        bad.append(("runs", sorted(set(mine) ^ set(ref))[:4]))
    for r in sorted(set(mine) & set(ref)):
        for k in sorted(set(mine[r]) | set(ref[r])):
            x, y = mine[r].get(k), ref[r].get(k)
            same = (x == y) or (isinstance(x, float) and isinstance(y, float) and (math.isclose(x, y, rel_tol=0, abs_tol=1e-9) or (x != x and y != y)))
            if not same:
                bad.append((r, k))
    print("RGB_PATH_PARITY", "PASS" if not bad else f"FAIL {len(bad)} fields, e.g. {bad[:6]}")
    raise SystemExit(0 if not bad else 1)
