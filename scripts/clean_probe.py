"""A4 pre-launch probe for vclean (council/AGREED-causal-r1.md, Amendment A4; Astra's probe, Fable's split reporting; admission
per council/codereview-vclean-astra.md). Does the proposed clean ring carry scene-specific information under deployment noise?
Read-only on checkpoints.

Weights: vpriv's same-step causal blocks (runs/vpriv/ckpt_8000.pt, ckpt_12000.pt) loaded strictly into mrv/clean_video blocks
(same parameter names / shapes). Clips: the 8 fixed dev probe clips of scripts/noise_probe.py / train_causal.py (K+1 = 5
ticks, recipient = last tick, donor = a clip of a DIFFERENT dev drive; identities recorded). Seed 0.
Prefixes (ticks 0..K-1) are genuinely GENERATED with the proposed rollout at rho 1.0: x_T = the signed per-(drive, seed) CRC32
tensor z0 of the RECIPIENT drive, DDIM 25, eta 0, CFG 2, style 2, rows [cond, uncond]; calls 1..24 read the ring, call 25 (t*)
runs with deltas off and its block inputs are published to the ring after the tick (clean_rollout.py's loop).
  true     ring from the recipient clip's own generated prefix (its cameras)
  donor    ring from the donor clip's prefix generated with IDENTICAL noise tensors (same z0), donor cameras
  off      memory off (temporal blocks off == A)
  teacher  ring from the recipient's key latents of ticks 0..K-1 noised to t* (private draws), cond context (teacher-forced)
Score: recipient eps-MSE at t = 300 and 700 (recipient key latent, cameras and eps held fixed across arms; 4 eps draws per
clip, the same at both checkpoints), cond row of the ring only. Paired differences per clip, averaged within drive (recipient
drive) then across drives: donor - true and off - true (positive = true better), teacher_gain = off - teacher.

Admission (anything failing -> "NO DECISION", never LAUNCH): checkpoints exactly runs/vpriv/ckpt_8000.pt and ckpt_12000.pt
(full sha256 recorded), step 8000 / 12000, K 4, train_noise.private_clips 2, base_sha == vpriv's recorded base_sha (and equal
across both), seed 0, 4 draws, the eight fixed windows (recorded), every loss / difference finite from all 4 draws, a 2080 Ti,
and <= 60 wall minutes per checkpoint INCLUDING the shared setup (charged in full to each checkpoint).
Reading: at BOTH checkpoints donor - true > 0 and off - true > 0 at t 300 and off - true >= 0 at t 700 -> "LAUNCH"; a valid miss
-> "R2" (council round r2, not an automatic kill). The teacher gain only sets the optional jitter flag.

  python scripts/clean_probe.py --out runs/probe/clean_probe.json
"""
import argparse, hashlib, io, json, math, os, time, zlib
T_START = time.time()
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.clean_video import add_clean_temporal, set_mode, set_stage, publish, reset, ring_state, load_ring

WANT = {8000: "runs/vpriv/ckpt_8000.pt", 12000: "runs/vpriv/ckpt_12000.pt"}
p = argparse.ArgumentParser()
p.add_argument("--ckpts", default=",".join(WANT.values()))
p.add_argument("--out", required=True)
p.add_argument("--vpriv_args", default="runs/vpriv/args.json", help="vpriv's recorded args (base_sha)")
p.add_argument("--enc", default="data/enc_tpv10"); p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--seed", type=int, default=0, help="rollout seed of the prefixes (A4: seed 0)")
p.add_argument("--reps", type=int, default=4, help="recipient eps draws per clip (A4 decision: 4)")
a = p.parse_args()
dev = "cuda"; SD = "sd-legacy/stable-diffusion-v1-5"; RHO = 1.0; TV = (300, 700); BUDGET_MIN = 60.0
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
CODE_FILES = [os.path.relpath(__file__), "mrv/clean_video.py", "mrv/sd.py"]  # keys match the suite manifest (run from the repo root)
CODE_SHA256 = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in CODE_FILES}
INVALID = []  # global admission failures -> NO DECISION


def invalid(m):
    INVALID.append(m); print("INVALID:", m, flush=True)


CKPTS = a.ckpts.split(",")
if CKPTS != list(WANT.values()):
    invalid(f"checkpoints {CKPTS} != {list(WANT.values())}")
if a.seed != 0:
    invalid(f"seed {a.seed} != 0")
if a.reps != 4:
    invalid(f"reps {a.reps} != 4")
GPU = torch.cuda.get_device_name()
if "2080 Ti" not in GPU:
    invalid(f"GPU {GPU} is not a 2080 Ti (A4 budget is 60 2080 Ti-minutes per checkpoint)")
VPRIV = json.load(open(a.vpriv_args)) if os.path.exists(a.vpriv_args) else {}
VBASE = VPRIV.get("base_sha")
if not VBASE:
    invalid(f"no recorded base_sha in {a.vpriv_args}")
ck = torch.load(CKPTS[0], map_location="cpu"); r = ck["args"]["rank"]; K = ck["K"]; L = K + 1
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); res_ = unet.load_state_dict(ck["lora"], strict=False)
assert not res_.unexpected_keys and {n for n, _ in unet.named_parameters() if "lora" in n} == set(ck["lora"]), "LoRA load mismatch"
unet.to(dev, AMP).eval()
TW = add_clean_temporal(unet, K)
NAMES = [n for n, m in unet.named_modules() if m.__class__.__name__ == "WithCleanTemporal"]
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
LORA_KEYS = set(ck["lora"]); del ck
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); ddim.set_timesteps(25); TS = ddim.timesteps
T_STAR = int(TS[-1]); assert T_STAR <= 81, T_STAR
acp = ddim.alphas_cumprod.to(dev)
RAYS = torch.load(a.rays); DEVL = json.load(open(a.lists))["dev"]

# the 8 fixed probe clips (same generator recipe as scripts/noise_probe.py / train_causal.py); identities recorded
gp = torch.Generator().manual_seed(4242); pc = {}
for sc in DEVL:
    e = torch.load(f"{a.enc}/{sc}.pt")
    ok, seq, ticks, hard = e["ok"].tolist(), e["seq"].tolist(), e["ticks"].tolist(), e["hard"].tolist()
    st_ = [i for i in range(len(ticks) - L + 1) if all(ok[i + k] and not hard[i + k] and seq[i + k] == seq[i] and ticks[i + k] == ticks[i] + k for k in range(L))]
    pc[sc] = [(i, ticks[i:i + L], e["tokens"][i:i + L].float(), e["lat"][i:i + L].float()) for i in (st_[k_] for k_ in torch.randperm(len(st_), generator=gp)[:2].tolist())]
PAIRS, WINDOWS = [], []
wid = lambda sc, w: {"drive": sc, "start": w[0], "ticks": w[1], "lat_sha": hashlib.sha256(w[3].numpy().tobytes()).hexdigest()[:16],
                     "tok_sha": hashlib.sha256(w[2].numpy().tobytes()).hexdigest()[:16]}
for pi in range(8):
    rsc, dsc = DEVL[pi % len(DEVL)], DEVL[(pi + 1) % len(DEVL)]
    rw, dw = pc[rsc][(pi // len(DEVL)) % len(pc[rsc])], pc[dsc][0]
    if rsc == dsc:
        invalid(f"pair {pi}: recipient and donor from the same drive")
    PAIRS.append(((rsc, rw[2], rw[3]), (dsc, dw[2], dw[3]))); WINDOWS.append({"recipient": wid(rsc, rw), "donor": wid(dsc, dw)})
if len(PAIRS) != 8 or len({(w["recipient"]["drive"], w["recipient"]["start"]) for w in WINDOWS}) != 8:
    invalid("the eight probe windows are not eight distinct recipient windows")
st1 = torch.full((1,), 2, dtype=torch.long, device=dev)
ray_of = lambda sc: RAYS[sc][None].to(dev).float()
with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
    U_NULL = adapter(PAIRS[0][0][1][0:1].to(dev), ray_of(PAIRS[0][0][0]), st1, torch.ones(1, dtype=torch.bool, device=dev)).to(AMP)
SETUP_MIN = (time.time() - T_START) / 60


def cond(tk, sc, j):
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        return adapter(tk[j:j + 1].to(dev), ray_of(sc), st1).to(AMP)


def gen_prefix(tk, sc, z0):
    """generate ticks 0..K-1 with the proposed rollout (rho 1.0, ring of final-call features); returns (ring snapshot, latents)"""
    reset(TW); set_mode(TW, "stream"); lats = []
    for j in range(K):
        c, z = cond(tk, sc, j), z0.clone()  # rho 1.0: x_T = z0 every tick
        for si, t in enumerate(TS):
            set_stage(TW, si == len(TS) - 1)
            with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
                e = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([c, U_NULL])).sample.float()
            ec, eu = e.chunk(2); z = ddim.step(eu + 2.0 * (ec - eu), t, z).prev_sample
        publish(TW); lats.append(z[0].float().cpu())
    set_stage(TW, False)
    return ring_state(TW), torch.stack(lats)


def teacher_ring(tk, sc, lt, eps_m):
    """ring from the key latents of ticks 0..K-1 noised to t* (teacher-forced), cond row only"""
    reset(TW); set_mode(TW, "stream"); set_stage(TW, True); ab = acp[T_STAR]
    for j in range(K):
        xm = ab.sqrt() * lt[j:j + 1].to(dev) + (1 - ab).sqrt() * eps_m[j:j + 1].to(dev)  # fp32 latents under autocast, as the rollout
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
            unet(xm, torch.full((1,), T_STAR, device=dev), encoder_hidden_states=cond(tk, sc, j))
        publish(TW)
    set_stage(TW, False)
    return ring_state(TW)


def mean(v):
    return sum(v) / len(v)


fin = lambda x: isinstance(x, float) and math.isfinite(x)
ARMS = ("true", "donor", "off", "teacher")
DIFFS = ("true_vs_donor", "true_vs_off", "teacher_gain")
OUT = {"code_sha256": CODE_SHA256, "rho": RHO, "t_star": T_STAR, "ddim_timesteps": [int(x) for x in TS], "K": K, "seed": a.seed, "reps": a.reps,
       "amp": str(AMP), "gpu": GPU, "t": list(TV), "windows": WINDOWS, "vpriv_args": a.vpriv_args, "vpriv_base_sha": VBASE,
       "setup_minutes": round(SETUP_MIN, 2), "ckpts": {}}
for path in CKPTS:
    t0 = time.time(); why = []  # this checkpoint's admission failures
    raw = open(path, "rb").read(); sha = hashlib.sha256(raw).hexdigest()
    ck = torch.load(io.BytesIO(raw), map_location="cpu"); del raw
    step = ck.get("step")
    if WANT.get(step) != path:
        why.append(f"step {step} at {path} is not one of {WANT}")
    if ck.get("K") != 4 or K != 4:
        why.append(f"K {ck.get('K')} != 4")
    if (ck.get("train_noise") or {}).get("private_clips") != 2:
        why.append(f"train_noise {ck.get('train_noise')} is not private_clips 2")
    if not VBASE or ck.get("base_sha") != VBASE:
        why.append(f"base_sha {ck.get('base_sha')} != vpriv's recorded {VBASE}")
    if "memory" in ck:
        why.append(f"not a same-step vpriv checkpoint (memory {ck.get('memory')})")
    if set(ck["lora"]) != LORA_KEYS:
        why.append("LoRA tensor set differs from the first checkpoint's")
    if ck.get("block_names") != NAMES or len(ck["temporal"]) != 16 or len(TW) != 16:
        raise SystemExit(f"{path}: temporal block names / count differ from the clean wrappers'")
    for w, sd_ in zip(TW, ck["temporal"]):
        w.temporal.load_state_dict(sd_, strict=True)  # same names / shapes as CausalTemporalBlock
    for w in TW:
        w.temporal.to(dev, AMP).eval()
    g_eps = torch.Generator().manual_seed(777); g_mem = torch.Generator().manual_seed(778)  # same draws at every checkpoint
    clips = []
    for pi, ((rsc, tk, lt), (dsc, otk, olt)) in enumerate(PAIRS):
        z0 = torch.randn(1, 4, 32, 64, generator=torch.Generator().manual_seed(zlib.crc32(f"{rsc}|{a.seed}".encode()))).to(dev)
        ring_true, lat_true = gen_prefix(tk, rsc, z0)
        ring_donor, _ = gen_prefix(otk, dsc, z0)  # different drive's cameras, identical noise tensors
        ring_teach = teacher_ring(tk, rsc, lt, torch.randn(K, 4, 32, 64, generator=g_mem))
        c_rec = cond(tk, rsc, K)  # recipient cameras (fixed across arms)
        x0 = lt[K:K + 1].to(dev)
        draws = {tv: {k: [] for k in ARMS} for tv in TV}  # per-draw losses (all kept: finiteness checked on every one)
        for _ in range(a.reps):
            eps = torch.randn(1, 4, 32, 64, generator=g_eps).to(dev)
            for tv in TV:
                ab = acp[tv]; xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps; tt = torch.full((1,), tv, device=dev)
                for arm, ring, rows in (("true", ring_true, [0]), ("donor", ring_donor, [0]), ("teacher", ring_teach, None), ("off", None, None)):
                    if ring is None:
                        set_mode(TW, "off")
                    else:
                        set_mode(TW, "stream"); set_stage(TW, False); load_ring(TW, ring, rows)
                    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
                        pr = unet(xt, tt, encoder_hidden_states=c_rec).sample.float()
                    draws[tv][arm].append(((pr - eps) ** 2).mean().item())
        rec = {"pair": pi, "recipient": rsc, "donor": dsc, "prefix_mse_to_key": round(((lat_true - lt[:K]) ** 2).mean().item(), 5)}
        for tv in TV:
            dd = draws[tv]
            n_ok = all(len(dd[k]) == a.reps and all(fin(x) for x in dd[k]) for k in ARMS)
            m = {k: mean(dd[k]) for k in ARMS}
            rec[str(tv)] = {"draws": dd, **m, "true_vs_donor": m["donor"] - m["true"], "true_vs_off": m["off"] - m["true"],
                            "teacher_gain": m["off"] - m["teacher"]}
            if not n_ok or not all(fin(rec[str(tv)][k]) for k in DIFFS):
                why.append(f"pair {pi} t {tv}: missing or non-finite losses / differences")
        clips.append(rec); print(path, pi, {k: v for k, v in rec.items() if k not in ("300", "700")},
                                 {tv: {k: round(rec[str(tv)][k], 6) for k in DIFFS} for tv in TV}, flush=True)
    reset(TW)
    if len(clips) != 8:
        why.append(f"{len(clips)} of 8 clips")
    drives = sorted({c["recipient"] for c in clips}, key=DEVL.index)
    summ = {}
    for tv in TV:
        per_drive = {d: {k: mean([c[str(tv)][k] for c in clips if c["recipient"] == d]) for k in DIFFS} for d in drives}
        summ[str(tv)] = {"per_drive": per_drive, **{k: mean([x[k] for x in per_drive.values()]) for k in DIFFS},
                         **{k + "_drives_pos": sum(x[k] > 0 for x in per_drive.values()) for k in ("true_vs_donor", "true_vs_off")}}
        if not all(fin(summ[str(tv)][k]) for k in DIFFS):
            why.append(f"t {tv}: non-finite drive means")
    s3, s7 = summ["300"], summ["700"]
    reading = {"donor_minus_true_t300_gt0": s3["true_vs_donor"] > 0, "off_minus_true_t300_gt0": s3["true_vs_off"] > 0,
               "off_minus_true_t700_ge0": s7["true_vs_off"] >= 0}
    reading["pass"] = all(reading.values())
    own = (time.time() - t0) / 60; total = SETUP_MIN + own
    if total > BUDGET_MIN:
        why.append(f"{total:.1f} min incl. setup > {BUDGET_MIN:.0f}")
    OUT["ckpts"][path] = {"ckpt_sha256": sha, "step": step, "K": ck.get("K"), "train_noise": ck.get("train_noise"), "base_sha": ck.get("base_sha"),
                          "rollout_rho_of_ckpt": ck.get("rho"), "clips": clips, "summary": summ, "reading": reading,
                          "gen_gain_t300": s3["true_vs_off"], "teacher_gain_t300": s3["teacher_gain"], "minutes_own": round(own, 2),
                          "minutes_incl_setup": round(total, 2), "admission_failures": why, "valid": not why}
    print(path, "reading", reading, f"{total:.1f} min incl. setup", "VALID" if not why else f"INVALID {why}", flush=True)
    del ck
# ---------- A4 launch reading over both checkpoints ----------
by_step = {v["step"]: v for v in OUT["ckpts"].values()}
reasons = list(INVALID)
for s_, path in WANT.items():
    v = OUT["ckpts"].get(path)
    if v is None or v["step"] != s_:
        reasons.append(f"vpriv {s_ // 1000}k not probed")
    elif not v["valid"]:
        reasons += [f"{s_ // 1000}k: {x}" for x in v["admission_failures"]]
if not reasons and by_step[8000]["base_sha"] != by_step[12000]["base_sha"]:
    reasons.append("8k / 12k frozen bases differ")
if reasons:
    OUT["a4_reading"] = {"decision": "NO DECISION", "launch": None, "valid": False, "reasons": reasons}
else:
    both = [by_step[8000], by_step[12000]]
    launch = all(v["reading"]["pass"] for v in both)
    OUT["a4_reading"] = {"decision": "LAUNCH" if launch else "R2", "launch": launch, "valid": True, "reasons": [],
                         "per_ckpt": {str(s_): by_step[s_]["reading"] for s_ in (8000, 12000)},
                         "jitter_allowed": all(v["gen_gain_t300"] < 0.5 * v["teacher_gain_t300"] for v in both),
                         "note": "R2 = council round r2, not an automatic kill. donor-true and off-true are reported separately per "
                                 "checkpoint and t; the teacher gain only sets the optional {t*, t*+40, t*+80} jitter flag"}
print("A4 reading:", json.dumps({k: v for k, v in OUT["a4_reading"].items() if k != "per_ckpt"}), flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out + ".tmp", "w") as fh:
    json.dump(OUT, fh, indent=1)
os.replace(a.out + ".tmp", a.out)
print("CLEAN_PROBE_DONE", OUT["a4_reading"]["decision"], a.out)
