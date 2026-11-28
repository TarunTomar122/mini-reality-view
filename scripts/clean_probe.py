"""A4 pre-launch probe for vclean (council/AGREED-causal-r1.md, Amendment A4; Astra's probe, Fable's split reporting).
Does the proposed clean ring carry scene-specific information under deployment noise? Read-only on checkpoints.

Weights: vpriv's same-step causal blocks (runs/vpriv/ckpt_8000.pt, ckpt_12000.pt) loaded strictly into mrv/clean_video blocks
(same parameter names / shapes). Clips: the 8 fixed dev probe clips of scripts/noise_probe.py / train_causal.py (K+1 = 5
ticks, recipient = last tick, donor = a clip of a DIFFERENT dev drive). Seed 0.
Prefixes (ticks 0..K-1) are genuinely GENERATED with the proposed rollout at rho 1.0: x_T = the signed per-(drive, seed) CRC32
tensor z0 of the RECIPIENT drive, DDIM 25, eta 0, CFG 2, style 2, rows [cond, uncond]; calls 1..24 read the ring, call 25 (t*)
runs with deltas off and its block inputs are published to the ring after the tick (clean_rollout.py's loop).
  true     ring from the recipient clip's own generated prefix (its cameras)
  donor    ring from the donor clip's prefix generated with IDENTICAL noise tensors (same z0), donor cameras
  off      memory off (temporal blocks off == A)
  teacher  ring from the recipient's key latents of ticks 0..K-1 noised to t* (private draws), cond context (teacher-forced)
Score: recipient eps-MSE at t = 300 and 700 (recipient key latent, cameras and eps held fixed across arms; --reps eps draws),
cond row of the ring only (a single conditional forward, as in training). Paired differences per clip, averaged within drive
(recipient drive) then across drives: true_vs_donor = donor - true, true_vs_off = off - true (positive = true better),
teacher_gain = off - teacher.
A4 launch reading: at BOTH vpriv 8k and 12k, true beats donor AND off at t 300 (drive means > 0) and is no worse than off at
t 700 (drive mean >= 0); else round r2 (not an automatic kill). Incomplete probe = no decision. Also reported: the jitter
indicator (generated-history gain under half the teacher-forced gain at t 300 at both checkpoints -> key-noise jitter allowed).

  python scripts/clean_probe.py --out runs/probe/clean_probe.json [--ckpts runs/vpriv/ckpt_8000.pt,runs/vpriv/ckpt_12000.pt]
"""
import argparse, hashlib, io, json, os, time, zlib
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.clean_video import add_clean_temporal, set_mode, set_stage, publish, reset, ring_state, load_ring

p = argparse.ArgumentParser()
p.add_argument("--ckpts", default="runs/vpriv/ckpt_8000.pt,runs/vpriv/ckpt_12000.pt")
p.add_argument("--out", required=True)
p.add_argument("--enc", default="data/enc_tpv10"); p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--seed", type=int, default=0, help="rollout seed of the prefixes (A4: seed 0)")
p.add_argument("--reps", type=int, default=4, help="recipient eps draws per clip (identical across arms and checkpoints)")
a = p.parse_args()
dev = "cuda"; SD = "sd-legacy/stable-diffusion-v1-5"; RHO = 1.0; TV = (300, 700)
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
CODE = hashlib.sha256(b"".join(open(f, "rb").read() for f in [__file__, "mrv/clean_video.py", "mrv/sd.py"])).hexdigest()[:12]
CKPTS = a.ckpts.split(",")
ck = torch.load(CKPTS[0], map_location="cpu"); r = ck["args"]["rank"]; K = ck["K"]; L = K + 1
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); res_ = unet.load_state_dict(ck["lora"], strict=False)
assert not res_.unexpected_keys and {n for n, _ in unet.named_parameters() if "lora" in n} == set(ck["lora"]), "LoRA load mismatch"
unet.to(dev, AMP).eval()
TW = add_clean_temporal(unet, K)
NAMES = [n for n, m in unet.named_modules() if m.__class__.__name__ == "WithCleanTemporal"]
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
BASE = ck.get("base_sha"); del ck
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); ddim.set_timesteps(25); TS = ddim.timesteps
T_STAR = int(TS[-1]); assert T_STAR <= 81, T_STAR
acp = ddim.alphas_cumprod.to(dev)
RAYS = torch.load(a.rays); DEVL = json.load(open(a.lists))["dev"]

# the 8 fixed probe clips (same generator recipe as scripts/noise_probe.py / train_causal.py)
gp = torch.Generator().manual_seed(4242); pc = {}
for sc in DEVL:
    e = torch.load(f"{a.enc}/{sc}.pt")
    ok, seq, ticks, hard = e["ok"].tolist(), e["seq"].tolist(), e["ticks"].tolist(), e["hard"].tolist()
    st_ = [i for i in range(len(ticks) - L + 1) if all(ok[i + k] and not hard[i + k] and seq[i + k] == seq[i] and ticks[i + k] == ticks[i] + k for k in range(L))]
    pc[sc] = [(e["tokens"][i:i + L].float(), e["lat"][i:i + L].float()) for i in (st_[k_] for k_ in torch.randperm(len(st_), generator=gp)[:2].tolist())]
PAIRS = []
for pi in range(8):
    rsc, dsc = DEVL[pi % len(DEVL)], DEVL[(pi + 1) % len(DEVL)]
    assert rsc != dsc
    PAIRS.append(((rsc,) + pc[rsc][(pi // len(DEVL)) % len(pc[rsc])], (dsc,) + pc[dsc][0]))
st1 = torch.full((1,), 2, dtype=torch.long, device=dev)
ray_of = lambda sc: RAYS[sc][None].to(dev).float()
with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
    U_NULL = adapter(PAIRS[0][0][1][0:1].to(dev), ray_of(PAIRS[0][0][0]), st1, torch.ones(1, dtype=torch.bool, device=dev)).to(AMP)


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


OUT = {"code": CODE, "rho": RHO, "t_star": T_STAR, "K": K, "seed": a.seed, "reps": a.reps, "amp": str(AMP), "t": list(TV),
       "pairs": [[x[0], y[0]] for x, y in PAIRS], "ckpts": {}}
for path in CKPTS:
    t0 = time.time()
    raw = open(path, "rb").read(); sha = hashlib.sha256(raw).hexdigest()[:16]
    ck = torch.load(io.BytesIO(raw), map_location="cpu"); del raw
    assert ck.get("K") == K and len(ck["temporal"]) == 16 and len(TW) == 16, "need 16 blocks with the same K"
    if ck.get("block_names") != NAMES:
        raise SystemExit(f"{path}: temporal block names differ from the clean wrappers'")
    if ck.get("base_sha") != BASE:
        raise SystemExit(f"{path}: frozen base differs from {CKPTS[0]}'s")
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
        acc = {tv: {k: 0.0 for k in ("true", "donor", "off", "teacher")} for tv in TV}
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
                    acc[tv][arm] += ((pr - eps) ** 2).mean().item() / a.reps
        rec = {"recipient": rsc, "donor": dsc, "prefix_mse_to_key": round(((lat_true - lt[:K]) ** 2).mean().item(), 5)}
        for tv in TV:
            m = acc[tv]
            rec[str(tv)] = {**{k: round(v, 6) for k, v in m.items()}, "true_vs_donor": m["donor"] - m["true"], "true_vs_off": m["off"] - m["true"],
                            "teacher_gain": m["off"] - m["teacher"]}
        clips.append(rec); print(path, pi, rec, flush=True)
    reset(TW)
    drives = sorted({c["recipient"] for c in clips}, key=DEVL.index)
    summ = {}
    for tv in TV:
        per_drive = {d: {k: mean([c[str(tv)][k] for c in clips if c["recipient"] == d]) for k in ("true_vs_donor", "true_vs_off", "teacher_gain")} for d in drives}
        summ[str(tv)] = {"per_drive": {d: {k: round(v, 6) for k, v in x.items()} for d, x in per_drive.items()},
                         **{k: mean([x[k] for x in per_drive.values()]) for k in ("true_vs_donor", "true_vs_off", "teacher_gain")},
                         **{k + "_drives_pos": sum(x[k] > 0 for x in per_drive.values()) for k in ("true_vs_donor", "true_vs_off")}}
    s3, s7 = summ["300"], summ["700"]
    reading = {"true_beats_donor_t300": s3["true_vs_donor"] > 0, "true_beats_off_t300": s3["true_vs_off"] > 0,
               "true_not_worse_than_off_t700": s7["true_vs_off"] >= 0}
    reading["pass"] = all(reading.values())
    mins = (time.time() - t0) / 60
    OUT["ckpts"][path] = {"ckpt_sha": sha, "step": ck.get("step"), "train_noise": ck.get("train_noise"), "rollout_rho_of_ckpt": ck.get("rho"),
                          "memory_of_ckpt": ck.get("memory", "same_step"), "clips": clips, "summary": summ, "reading": reading,
                          "gen_gain_t300": s3["true_vs_off"], "teacher_gain_t300": s3["teacher_gain"], "minutes": round(mins, 1),
                          "gpu": torch.cuda.get_device_name(), "over_60min_budget": mins > 60, "complete": len(clips) == len(PAIRS)}
    print(path, "reading", reading, f"{mins:.1f} min", flush=True)
    del ck
# ---------- A4 launch reading over both checkpoints ----------
steps = {v["step"]: v for v in OUT["ckpts"].values()}
complete = all(s_ in steps and steps[s_]["complete"] for s_ in (8000, 12000))
if not complete:
    OUT["a4_reading"] = {"decision": "NO DECISION (incomplete probe: need complete vpriv 8k and 12k)", "launch": None}
else:
    both = [steps[8000], steps[12000]]
    launch = all(v["reading"]["pass"] for v in both)
    OUT["a4_reading"] = {"launch": launch, "decision": "LAUNCH vclean" if launch else "ROUND r2 (not an automatic kill)",
                         "per_ckpt": {s_: steps[s_]["reading"] for s_ in (8000, 12000)},
                         "jitter_allowed": all(v["gen_gain_t300"] < 0.5 * v["teacher_gain_t300"] for v in both),
                         "note": "true-vs-donor and true-vs-off are reported separately per checkpoint and t (Fable); the jitter rule "
                                 "{t*, t*+40, t*+80} applies only if generated gain < half the teacher-forced gain at t 300 at both"}
print("A4 reading:", OUT["a4_reading"], flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(OUT, open(a.out, "w"), indent=1, default=float)
print("CLEAN_PROBE_DONE", a.out)
