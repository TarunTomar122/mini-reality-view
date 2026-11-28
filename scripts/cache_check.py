"""A5 cache check for vmix (council/AGREED-causal-r1.md Amendment A5; council/codereview-vmix-astra.md item 3,
codereview-vmix2-astra.md): must pass before the trainer reads the cache (update 2001). Every file is read once; its digest is
computed on the bytes that are parsed.
  1. EXACTLY the 144 train drives each have <cache>/<scene>_s0.pt from scripts/build_cache.py with the signed fields: ckpt_sha of
     --ckpt (A@20k), roll_code of the current build_cache.py + mrv/sd.py, method cache_off, rho 1.0, seed 0, unmerged, the same
     amp / batch / schedule in every file; enc_sha AND ray_sha of the current inputs (each drive's encoding digest is recorded so
     the trainer can compare it with the encoding bytes it loads); fp16 (n, 4, 32, 64) latents with ticks / hard flags aligned
     with the encoding; finite non-zero latents on every non-hard tick;
  2. whole-drive parity: ONE train drive re-rendered by the UNCHANGED signed scripts/causal_rollout.py --mode off --rho 1.0
     --seeds 0 with THIS checker's --enc / --rays / --lists (batch 1) equals the cache on every tick (ticks / source ticks / ok /
     hard / restart / amp equal, max |latent diff| < 0.01). The reference artifact's path and digest are recorded; the shared
     verifier re-loads it and RE-MEASURES parity before the first cache read;
  3. dev: the six signed A@20k seed-0 independent dev renders (runs/troll/A20k_indep_dev: memory off, one z0 per drive) are
     required, complete and signed; their digests are recorded (the gate binds them). They carry no seed field, so seed 0 is
     MEASURED: causal_rollout --split dev --seeds 0 --max_ticks 5 must match each on its non-hard first ticks (< 0.01); those
     artifacts are recorded too. Report: cache-to-key latent MSE on usable ticks, train (cache) vs dev; NaN fails.
The record is then re-read by mrv/cache_verify.py (the same verifier the gate and the trainer use); pass requires that too.

  python scripts/cache_check.py --cache runs/cache/A20k_off_rho1_train --out runs/cache/check.json
"""
import argparse, json, math, os, subprocess, sys, time
import torch
from mrv.cache_verify import A_PATH, DEV_REF, read, hb, tload, sha, file_code, builder_roll_code, rollout_roll_code, ray_sha, make_bind, verify

p = argparse.ArgumentParser()
p.add_argument("--cache", default="runs/cache/A20k_off_rho1_train")
p.add_argument("--ckpt", default=A_PATH)
p.add_argument("--drive", default="", help="parity drive (default: the first train drive)")
p.add_argument("--parity_out", default="", help="causal_rollout output dir (default <cache>_parity)")
p.add_argument("--dev_ref", default=DEV_REF)
p.add_argument("--enc", default="data/enc_tpv10"); p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--out", required=True)
a = p.parse_args()
L = json.loads(read(a.lists)); TRAIN, DEVL = sorted(L["train"]), list(L["dev"])
RAYS = tload(read(a.rays))
FAIL = []


def need(c, m):
    if not c:
        FAIL.append(m); print("FAIL:", m, flush=True)
    return bool(c)


fin = lambda x: isinstance(x, float) and math.isfinite(x)
CKPT_SHA = sha(a.ckpt)[:16]
ROLL_CODE = builder_roll_code()
CODE = file_code()
need(len(TRAIN) == 144 and len(DEVL) == 6, f"{len(TRAIN)} train / {len(DEVL)} dev drives (A5: 144 / 6)")
ROLL = [sys.executable, "scripts/causal_rollout.py", "--ckpt", a.ckpt, "--mode", "off", "--rho", "1.0", "--enc", a.enc, "--rays", a.rays, "--lists", a.lists, "--seeds", "0"]


def usable_mse(lat, E):
    """mean over usable ticks (ok and not hard) of the latent MSE vs the key latents (positions == encoding positions)"""
    m = [i for i in range(len(lat)) if bool(E["ok"][i]) and not bool(E["hard"][i])]
    return ((lat[m].float() - E["lat"][m].float()) ** 2).mean().item() if m else float("nan")


def load_enc(sc):
    buf = read(f"{a.enc}/{sc}.pt")
    return tload(buf), hb(buf)[:16]


# ---------- 1. exactly the train drives: file, signed fields, inputs, shape / dtype, alignment ----------
files, enc_sha, train_mse, common, ENC_T = {}, {}, {}, {}, {}
extra = sorted(f[:-6] for f in os.listdir(a.cache) if f.endswith("_s0.pt") and f[:-6] not in TRAIN) if os.path.isdir(a.cache) else []
need(not extra, f"cache holds drives outside the train list: {extra[:5]}")
for sc in TRAIN:
    f = f"{a.cache}/{sc}_s0.pt"
    if not need(os.path.isfile(f), f"missing {f}"):
        continue
    buf = read(f); files[sc] = hb(buf); c = tload(buf); del buf
    E, enc_sha[sc] = load_enc(sc); ENC_T[sc] = E["ticks"]
    need(c.get("ckpt_sha") == CKPT_SHA and c.get("roll_code") == ROLL_CODE and c.get("method") == "cache_off" and c.get("rho") == 1.0
         and c.get("seed") == 0 and c.get("merge") is False and c.get("salt_from") == -1 and c.get("start_tick") == 0,
         f"{sc}: signed fields { {k: c.get(k) for k in ('ckpt_sha', 'roll_code', 'method', 'rho', 'seed', 'merge')} }")
    need(c.get("enc_sha") == enc_sha[sc], f"{sc}: encoding changed since the cache was built")
    need(c.get("ray_sha") == ray_sha(RAYS[sc]), f"{sc}: rays changed since the cache was built")
    x = c["lat"]
    need(x.dtype == torch.float16 and tuple(x.shape) == (len(E["ticks"]), 4, 32, 64), f"{sc}: latents {x.dtype} {tuple(x.shape)}")
    need(torch.equal(c["ticks"], E["ticks"]) and torch.equal(c["hard"].bool(), E["hard"].bool()), f"{sc}: ticks / flags not aligned")
    for k in ("amp", "batch", "merge", "ddim_timesteps"):
        common.setdefault(k, c.get(k)); need(c.get(k) == common[k], f"{sc}: {k} {c.get(k)} differs from other files' {common[k]}")
    nh = [i for i in range(len(x)) if not bool(c["hard"][i])]
    xn = x[nh].float()
    need(bool(torch.isfinite(xn).all()) and bool((xn.abs().flatten(1).amax(1) > 0).all()), f"{sc}: non-finite or empty latents")
    train_mse[sc] = usable_mse(x, E)
    need(fin(train_mse[sc]), f"{sc}: train MSE {train_mse[sc]}")

# ---------- 2. whole-drive parity: the unchanged causal_rollout.py on the same inputs (memory off, rho 1.0, seed 0, batch 1) ----------
drv = a.drive or TRAIN[0]
pdir = a.parity_out or a.cache.rstrip("/") + "_parity"
os.makedirs(pdir, exist_ok=True); dl = f"{pdir}/drive.txt"; open(dl, "w").write(drv + "\n")
t0 = time.time()
r = subprocess.run(ROLL + ["--split", "train", "--drive_list", dl, "--out", pdir], capture_output=True, text=True)
ref_file = f"{pdir}/{drv}_s0.pt"
parity = {"drive": drv, "dir": pdir, "ref_file": ref_file, "exit": r.returncode, "minutes": round((time.time() - t0) / 60, 1)}
if need(r.returncode == 0 and os.path.isfile(ref_file), f"causal_rollout parity render failed: {r.stdout[-400:]} {r.stderr[-400:]}") and drv in files:
    pbuf = read(ref_file); parity["ref_file_sha256"] = hb(pbuf); x = tload(pbuf); del pbuf
    cbuf = read(f"{a.cache}/{drv}_s0.pt"); need(hb(cbuf) == files[drv], f"cache file {drv} changed during the check"); c = tload(cbuf); del cbuf
    n_enc = len(ENC_T[drv])
    whole = len(x["lat"]) == len(c["lat"]) == n_enc
    eq = whole and all(torch.equal(x[k], c[k]) for k in ("ticks", "src_ticks", "ok", "hard", "restart")) and x["amp"] == c["amp"]
    d = (x["lat"].float() - c["lat"].float()).abs().max().item() if whole else float("inf")
    parity.update({"fields_equal": eq, "whole": whole, "enc_ticks": n_enc, "amp": [x["amp"], c["amp"]], "n_ticks": len(x["lat"]), "max_diff": d,
                   "ref_ckpt_sha": x.get("ckpt_sha"), "ref_roll_code": x.get("roll_code"), "ref_method": x.get("method"), "ref_rho": x.get("rho"),
                   "ref_enc_sha": x.get("enc_sha"), "ref_ray_sha": x.get("ray_sha")})
    need(x.get("ckpt_sha") == CKPT_SHA and x.get("rho") == 1.0 and x.get("method") == "causal_off" and x.get("roll_code") == rollout_roll_code(),
         "parity render is not the unchanged causal_rollout of A@20k off at rho 1.0")
    need(x.get("enc_sha") == c.get("enc_sha") and x.get("ray_sha") == c.get("ray_sha"), "parity render and cache saw different inputs")
    need(eq, f"parity: whole drive {whole} ({len(x['lat'])}/{len(c['lat'])}/{n_enc} ticks); ticks / flags / amp equal {eq} (amp {x['amp']} vs {c['amp']})")
    need(math.isfinite(d) and d < 0.01, f"parity: max |latent diff| {d:.4g} >= 0.01")

# ---------- 3. dev: six bound signed seed-0 references (digests), seed 0 measured, MSE report ----------
sdir = a.cache.rstrip("/") + "_devseed"
r2 = subprocess.run(ROLL + ["--split", "dev", "--max_ticks", "5", "--out", sdir], capture_output=True, text=True)
need(r2.returncode == 0, f"dev seed-check render failed: {r2.stdout[-400:]} {r2.stderr[-400:]}")
dev_mse, dev_files, dev_seed = {}, {}, {}
for sc in DEVL:
    f = f"{a.dev_ref}/{sc}_s0.pt"
    if not need(os.path.isfile(f), f"dev reference {f} missing"):
        continue
    dbuf = read(f); dev_files[sc] = hb(dbuf); c = tload(dbuf); del dbuf
    E, _ = load_enc(sc)
    ok = need(str(c.get("snapshot", "")).endswith(A_PATH) and c.get("snapshot_step") == 20000 and c.get("method") == "independent"
              and c.get("swap_from", -1) == -1 and c.get("seed", 0) == 0, f"{f}: not the signed A@20k independent render ({c.get('snapshot')}, {c.get('method')})")
    ok = need(len(c["lat"]) == len(E["ticks"]) and torch.equal(c["ticks"], E["ticks"]), f"{f}: incomplete ({len(c['lat'])}/{len(E['ticks'])} ticks)") and ok
    sf = f"{sdir}/{sc}_s0.pt"
    if need(r2.returncode == 0 and os.path.isfile(sf), f"{sc}: seed-check artifact missing"):
        sbuf = read(sf); dev_seed[sc] = {"file": sf, "sha256": hb(sbuf)}; s_ = tload(sbuf); del sbuf
        k = len(s_["lat"]); nh = [i for i in range(k) if not bool(s_["hard"][i])]
        ds = (s_["lat"][nh].float() - c["lat"][nh].float()).abs().max().item() if nh and torch.equal(s_["ticks"], c["ticks"][:k]) else float("inf")
        dev_seed[sc]["max_diff"] = ds
        ok = need(all(q.startswith(f"{sc}|0|") for q in s_["noise_key"]) and math.isfinite(ds) and ds < 0.01,
                  f"{sc}: dev reference is not seed 0 (first {len(nh)} non-hard ticks differ by {ds:.4g})") and ok
    dev_mse[sc] = usable_mse(c["lat"], E) if ok else float("nan")
    need(fin(dev_mse[sc]), f"{sc}: dev MSE {dev_mse[sc]}")
mean = lambda d: sum(d.values()) / len(d) if d and all(fin(v) for v in d.values()) else float("nan")
rec = {"kind": "cache_check", "pass": not FAIL, "failures": FAIL, "cache": a.cache, "ckpt": a.ckpt, "ckpt_sha": CKPT_SHA, "roll_code": ROLL_CODE,
       "lists": a.lists, "enc": a.enc, "rays": a.rays, "build": common, "files": files, "n_files": len(files), "enc_sha": enc_sha, "parity": parity,
       "dev_ref": a.dev_ref, "dev_files": dev_files, "dev_seed": dev_seed,
       "mse": {"train_mean": mean(train_mse), "dev_mean": mean(dev_mse), "train_per_drive": train_mse, "dev_per_drive": dev_mse},
       "code_sha256": CODE, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}


def write(rec):
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(a.out + ".tmp", a.out)


write(rec)
if rec["pass"]:  # self-check with the shared verifier (the gate / trainer view); a mismatch turns the record into a fail
    data = {"ticks": ENC_T, "rays": RAYS, "enc_sha": enc_sha}
    ok, why, _, _ = verify(a.out, make_bind(a.ckpt, a.cache, a.lists, a.enc, a.rays, CODE, a.dev_ref), data)
    if not ok:
        rec["pass"] = False; rec["failures"] = [f"shared verifier: {why}"]; print("FAIL: shared verifier:", why); write(rec)
print("cache-to-key MSE train %.5f vs dev %.5f" % (rec["mse"]["train_mean"], rec["mse"]["dev_mean"]))
print("CACHE_CHECK", "PASS" if rec["pass"] else f"FAIL ({len(rec['failures'])})", a.out)
sys.exit(0 if rec["pass"] else 1)
