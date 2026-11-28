"""Copy warning "long minus short memory" (AGREED r1 gate reports; Neon 01:30: the signed scorer skips the late gap on
--reset_at rollouts because their history chain is broken at the reset). For each dev drive / seed present in both folders:
mean LPIPS-to-key (alex, same decode / key crop as scripts/temporal_metrics.py) over ticks 150-169 of the normal (long memory)
rollout minus the same for the --reset_at 134 (short memory) rollout. Warning if the mean difference > +0.01 (long memory
worse late in the drive). Positions must hold the same absolute ticks in both rollouts. Read-only.
  python scripts/reset_gap.py runs/troll/V16k_dev runs/troll/V16k_reset_dev --out runs/s_lock/resetgap_V16k.json
"""
import argparse, glob, json, os
import numpy as np, torch, lpips
from PIL import Image
from diffusers import AutoencoderKL
p = argparse.ArgumentParser(); p.add_argument("long"); p.add_argument("short"); p.add_argument("--out", required=True)
p.add_argument("--keys", default="data/sim/keys_tpv10"); a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
lp = lpips.LPIPS(net="alex").to(dev).eval()
tens = lambda im: torch.from_numpy(np.asarray(im)).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1


def to_img(lat):
    with torch.no_grad():
        x = vae.decode(lat[None].to(dev).float() / 0.18215).sample[0].clamp(-1, 1)
    return Image.fromarray(((x + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())


rows = {}
for fl in sorted(glob.glob(f"{a.long}/*_s*.pt")):
    b = os.path.basename(fl)[:-3]; fs = f"{a.short}/{b}.pt"
    if not os.path.exists(fs):
        continue
    L, S = torch.load(fl, weights_only=False), torch.load(fs, weights_only=False)
    assert S.get("reset_at") == 134 and L.get("reset_at") == -1, f"{b}: unexpected reset settings"
    assert L["ticks"][:170].tolist() == S["ticks"][:170].tolist(), f"{b}: tick positions differ"
    sc = b.rsplit("_s", 1)[0]; dl, ds = [], []
    for i in range(150, min(170, len(L["ticks"]))):
        if not (bool(L["ok"][i]) and not bool(L["hard"][i]) and bool(S["ok"][i]) and not bool(S["hard"][i])):
            continue
        kp = f"{a.keys}/{sc}/t{int(L['ticks'][i]):04d}.jpg"
        if not os.path.exists(kp):
            continue
        K = tens(Image.open(kp).convert("RGB").resize((512, 288), Image.BILINEAR).crop((0, 32, 512, 288)))
        with torch.no_grad():
            dl.append(float(lp(tens(to_img(L["lat"][i])), K))); ds.append(float(lp(tens(to_img(S["lat"][i])), K)))
    if dl:
        rows[b] = {"long": float(np.mean(dl)), "short": float(np.mean(ds)), "diff": float(np.mean(dl) - np.mean(ds)), "n": len(dl)}
drv = {}
for b, v in rows.items():
    drv.setdefault(b.rsplit("_s", 1)[0], []).append(v["diff"])
per_drive = {d: float(np.mean(v)) for d, v in drv.items()}
mean = float(np.mean(list(per_drive.values()))) if per_drive else None
out = {"long": a.long, "short": a.short, "ticks": "150-169", "runs": len(rows), "drives": len(per_drive), "per_drive": per_drive,
       "mean_long_minus_short": mean, "warning": (mean is not None and mean > 0.01), "rows": rows}
json.dump(out, open(a.out, "w"), indent=1)
print(f"RESETGAP runs {len(rows)} drives {len(per_drive)} long-minus-short LPIPS {mean:+.4f} (warn > +0.01: {out['warning']})")
