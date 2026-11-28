"""A5 cache check for vmix (council/AGREED-causal-r1.md Amendment A5): must pass before the trainer reads the cache (update 2001).
  1. every train drive has <cache>/<scene>_s0.pt from scripts/build_cache.py with the signed fields: ckpt_sha of the A@20k file,
     roll_code of the current build_cache.py + mrv/sd.py, method cache_off, rho 1.0, seed 0, same amp / batch / merge in every
     file, enc_sha of the current encoding, ticks aligned with the encoding, finite non-zero latents on every non-hard tick;
  2. parity: ONE train drive re-rendered by the UNCHANGED signed scripts/causal_rollout.py --mode off --rho 1.0 --seeds 0 (the
     matched memory-off rollout, batch 1) equals the cache: ticks / source ticks / ok / hard / restart / amp equal, max |latent
     diff| < 0.01 (this is what validates batching; run on the GPU type the cache was built on);
  3. report: cache-to-key latent MSE on usable ticks (ok, not hard), train drives (the cache) vs dev drives (the signed A@20k
     independent dev render runs/troll/A20k_indep_dev = memory off, one z0 per drive and seed; T2 ties it to causal_rollout off
     at rho 1.0), averaged within drive then across drives.
Writes a JSON record (kind cache_check, pass, per-file sha256 digests) that scripts/train_clean.py re-verifies before use.

  python scripts/cache_check.py --cache runs/cache/A20k_off_rho1_train --out runs/cache/check.json
"""
import argparse, hashlib, json, math, os, subprocess, sys, time
import torch

p = argparse.ArgumentParser()
p.add_argument("--cache", default="runs/cache/A20k_off_rho1_train")
p.add_argument("--ckpt", default="runs/longA/ckpt_20000.pt")
p.add_argument("--drive", default="", help="parity drive (default: the first train drive)")
p.add_argument("--parity_out", default="", help="causal_rollout output dir (default <cache>_parity)")
p.add_argument("--dev_ref", default="runs/troll/A20k_indep_dev")
p.add_argument("--enc", default="data/enc_tpv10"); p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--out", required=True)
a = p.parse_args()
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
L = json.load(open(a.lists)); TRAIN, DEVL = L["train"], L["dev"]
FAIL = []


def need(c, m):
    if not c:
        FAIL.append(m); print("FAIL:", m, flush=True)
    return bool(c)


CKPT_SHA = sha(a.ckpt)[:16]
ROLL_CODE = hashlib.sha256(b"".join(open(f, "rb").read() for f in ["scripts/build_cache.py", "mrv/sd.py"])).hexdigest()[:12]
CODE = {f: sha(f) for f in ["scripts/cache_check.py", "scripts/build_cache.py", "scripts/causal_rollout.py", "mrv/sd.py"]}


def usable_mse(lat, E, ref_ticks=None):
    """mean over usable ticks (ok and not hard) of the per-tick latent MSE vs the key latents; positions aligned by ticks"""
    n = len(lat)
    if ref_ticks is not None and not torch.equal(ref_ticks, E["ticks"][:n]):
        return None
    m = [i for i in range(n) if bool(E["ok"][i]) and not bool(E["hard"][i])]
    if not m:
        return None
    return ((lat[m].float() - E["lat"][m].float()) ** 2).mean().item()


# ---------- 1. every train drive: file, signed fields, alignment ----------
files, train_mse, common = {}, {}, {}
for sc in sorted(TRAIN):
    f = f"{a.cache}/{sc}_s0.pt"
    if not need(os.path.isfile(f), f"missing {f}"):
        continue
    files[sc] = sha(f)
    c = torch.load(f, map_location="cpu"); E = torch.load(f"{a.enc}/{sc}.pt")
    ok = need(c.get("ckpt_sha") == CKPT_SHA and c.get("roll_code") == ROLL_CODE and c.get("method") == "cache_off" and c.get("rho") == 1.0
              and c.get("seed") == 0 and c.get("salt_from") == -1 and c.get("start_tick") == 0, f"{sc}: signed fields "
              f"{ {k: c.get(k) for k in ('ckpt_sha', 'roll_code', 'method', 'rho', 'seed')} } != ckpt {CKPT_SHA} / roll {ROLL_CODE} / cache_off / 1.0 / 0")
    need(c.get("enc_sha") == sha(f"{a.enc}/{sc}.pt")[:16], f"{sc}: encoding changed since the cache was built")
    need(len(c["lat"]) == len(E["ticks"]) and torch.equal(c["ticks"], E["ticks"]) and torch.equal(c["hard"], E["hard"].bool()), f"{sc}: ticks / flags not aligned")
    for k in ("amp", "batch", "merge", "ddim_timesteps"):
        common.setdefault(k, c.get(k)); need(c.get(k) == common[k], f"{sc}: {k} {c.get(k)} differs from other files' {common[k]}")
    nh = [i for i in range(len(c["lat"])) if not bool(c["hard"][i])]
    lat_nh = c["lat"][nh].float()
    need(bool(torch.isfinite(lat_nh).all()) and bool((lat_nh.abs().flatten(1).amax(1) > 0).all()), f"{sc}: non-finite or empty latents")
    train_mse[sc] = usable_mse(c["lat"], E)
need(common.get("merge") is False, f"cache built with --merge {common.get('merge')} (parity first: unmerged only)")

# ---------- 2. parity: one train drive by the unchanged causal_rollout.py (memory off, rho 1.0, seed 0, batch 1) ----------
drv = a.drive or TRAIN[0]
pdir = a.parity_out or a.cache.rstrip("/") + "_parity"
os.makedirs(pdir, exist_ok=True); dl = f"{pdir}/drive.txt"; open(dl, "w").write(drv + "\n")
t0 = time.time()
r = subprocess.run([sys.executable, "scripts/causal_rollout.py", "--ckpt", a.ckpt, "--mode", "off", "--rho", "1.0", "--split", "train",
                    "--drive_list", dl, "--seeds", "0", "--out", pdir], capture_output=True, text=True)
parity = {"drive": drv, "dir": pdir, "exit": r.returncode, "minutes": round((time.time() - t0) / 60, 1)}
if need(r.returncode == 0, f"causal_rollout parity render failed: {r.stdout[-400:]} {r.stderr[-400:]}"):
    x, c = torch.load(f"{pdir}/{drv}_s0.pt"), torch.load(f"{a.cache}/{drv}_s0.pt")
    eq = len(x["lat"]) == len(c["lat"]) and all(torch.equal(x[k], c[k]) for k in ("ticks", "src_ticks", "ok", "hard", "restart")) and x["amp"] == c["amp"]
    d = (x["lat"].float() - c["lat"].float()).abs().max().item() if len(x["lat"]) == len(c["lat"]) else float("inf")
    parity.update({"fields_equal": eq, "amp": [x["amp"], c["amp"]], "n_ticks": len(x["lat"]), "max_diff": d, "ref_ckpt_sha": x.get("ckpt_sha"),
                   "ref_roll_code": x.get("roll_code"), "ref_method": x.get("method"), "ref_rho": x.get("rho"), "ref_file_sha256": sha(f"{pdir}/{drv}_s0.pt")})
    need(x.get("ckpt_sha") == CKPT_SHA and x.get("rho") == 1.0 and x.get("method") == "causal_off", "parity render is not A@20k off at rho 1.0")
    need(eq, f"parity: ticks / flags / amp differ (amp {x['amp']} vs {c['amp']})")
    need(math.isfinite(d) and d < 0.01, f"parity: max |latent diff| {d:.4g} >= 0.01")

# ---------- 3. report: train (cache) vs dev (signed A@20k independent render) cache-to-key MSE ----------
dev_mse = {}
for sc in DEVL:
    f = f"{a.dev_ref}/{sc}_s0.pt"
    if os.path.isfile(f):
        c = torch.load(f, map_location="cpu"); dev_mse[sc] = usable_mse(c["lat"], torch.load(f"{a.enc}/{sc}.pt"), c["ticks"])
mean = lambda d: (sum(v for v in d.values() if v is not None) / max(1, sum(v is not None for v in d.values()))) if d else None
need(len(dev_mse) == len(DEVL) and all(v is not None for v in dev_mse.values()), f"dev reference incomplete / misaligned ({len(dev_mse)}/{len(DEVL)} drives)")
need(len(train_mse) == len(TRAIN) and all(v is not None for v in train_mse.values()), "train cache-to-key MSE incomplete")
rec = {"kind": "cache_check", "pass": not FAIL, "failures": FAIL, "cache": a.cache, "ckpt": a.ckpt, "ckpt_sha": CKPT_SHA, "roll_code": ROLL_CODE,
       "build": common, "files": files, "n_files": len(files), "parity": parity, "dev_ref": a.dev_ref,
       "mse": {"train_mean": mean(train_mse), "dev_mean": mean(dev_mse), "train_per_drive": train_mse, "dev_per_drive": dev_mse},
       "code_sha256": CODE, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out + ".tmp", "w") as fh:
    json.dump(rec, fh, indent=1)
os.replace(a.out + ".tmp", a.out)
print("cache-to-key MSE train %.5f vs dev %.5f" % (rec["mse"]["train_mean"] or float("nan"), rec["mse"]["dev_mean"] or float("nan")))
print("CACHE_CHECK", "PASS" if not FAIL else f"FAIL ({len(FAIL)})", a.out)
sys.exit(0 if not FAIL else 1)
