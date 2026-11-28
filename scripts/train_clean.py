"""Clean-memory live video adapter on the FROZEN A@20k renderer: arm "vmix" (council/AGREED-causal-r1.md, Amendment A5; default)
or arm "vclean" (Amendment A4).

A5 "vmix" = A4 below with two changes inside the memory pass (council/r2-swap-fable.md items 3-5):
  * jitter: per clip, the memory timestep is drawn uniformly from {1, 41, 81} on a SEPARATE resumable RNG (state saved in the
    checkpoint); the source latents are noised to that timestep with the private memory draws and frozen-A features are
    collected there (query's camera-dropout context);
  * source mix: updates 1..mix_from (2000) read the real key latents (bit-identical to the pure key + jitter arm); from update
    mix_from+1, exactly one of the two clips per update, chosen by a coin on a THIRD resumable RNG, reads the CACHED frozen-A@20k
    memory-off rho-1.0 latents of the same ticks (--cache, built by scripts/causal_rollout.py; slurm/build_cache.sh). The target
    is always the real key. Before update mix_from+1 a passing scripts/cache_check.py record (--cache_record; every cache file
    digest re-verified) is required; without one the job saves and exits cleanly ("holds"); a later --resume continues.
A4 "vclean" (as signed): memory timestep fixed at t*, real key source only; "--arm vclean --jitter 1,41,81" = the A5 signed
pure key + jitter attribution control (no cached source).
Cache admission (council/codereview-vmix-astra.md): ONE verifier (mrv/cache_verify.py) for the gate's passed and pending states,
bound to the trainer's actual A@20k file, the reviewed builder / checker / rollout / dependency hashes, exactly the 144 train
drives and the whole-drive parity result. A passed gate admits only its approved record digest; a pending run re-checks
--cache_record when it reaches update mix_from (a record that arrived during training is picked up without a resume), and the
first record it uses is PINNED in every later checkpoint and re-verified on every resume. Resume identity (codereview-vmix2):
the checkpoint's gate record digest and approved cache record must equal the current ones (a passed-gate checkpoint cannot
resume under a pending gate or another record), and a checkpoint past mix_from must carry a pin. Every record / encoding /
cache file is hashed on the bytes that are parsed; each cache drive's encoding digest must equal the digest of the encoding
bytes this trainer loaded.

Recipe = vpriv (scripts/train_causal.py with --private_clips 2) except where the past comes from. Only the 16 new temporal
blocks (mrv/clean_video.py) train; the SD 1.5 UNet, A's LoRA and A's CameraAdapter are frozen (asserted). Each update:
  * the SAME draws as vpriv (one generator per update, seed * 100003 + step): 2 clips x 16 consecutive usable ticks of the 144
    train drives, ONE timestep per clip, per-tick noise (both clips fully private: mrv/noise.build_eps(private_clips=2), rho 0),
    whole-clip camera dropout 10%, style 2. A4 excludes t* from the noisy pass: a clip whose draw equals t* is redrawn
    uniformly from the other 999 values with a SEPARATE generator (all vpriv draws stay bit-identical);
  * memory pass: one extra no-grad frozen-A forward (temporal deltas OFF) at t* = DDIM-25's last timestep on the clips' key
    latents noised to t* with private draws (separate generator) and the same camera-dropout context; each block's input
    features are collected (detached) by hooks;
  * the noisy pass in clip mode: tick i reads its own token + the memory of ticks i-4 .. i-1; epsilon MSE (fp32) over ticks
    1..15; AdamW 1e-4, wd 0, 500-update warm-up, grad clip 1.0, bf16 autocast, fp32 temporal parameters.
Init: torch.manual_seed(seed) right before the blocks are built (same construction as vpriv -> same init; checked against
vpriv's recorded init_sha with --init_ref). Rollout noise rho is A4's gate rho (1.0; 0.95 only if T2 fails), stored in the
checkpoint as "rho", separately from the training noise.
Production admission (council/codereview-vclean-astra.md): vpriv's recorded init_sha, seed, block names, base_sha and training
inputs (lists, enc, rays, frames, clips) must match, and --gate_record (scripts/vclean_gate.py) must re-verify NOW: signed vpriv
no-joint-pass outcome, valid probe reading LAUNCH, clean suite FAILS=0 with code hashes == these files, timing record, and the T2
result that fixes the gate rho (1.0 on a matched T2 pass, 0.95 only on a documented numeric T2 failure); --rho_gate must equal
the record's. --timing runs skip the gate (throwaway weights, *timing* / *ctest* dirs only).
Logged every 100 updates (every update with --timing): loss, grad norm, out-norm, age-bias size, step_s, data_s, mem_s, peak GB,
batch_sha (vpriv's formula), mem_sha (t, memory-noise, jitter and coin draws), mem_t / mem_src per clip, code_sha.

  python scripts/train_clean.py --arm vmix --init runs/longA/ckpt_20000.pt --out runs/vmix --steps 16000 --rho_gate 1.0 \
      --gate_record runs/vclean_gate/vmix.json --cache runs/cache/A20k_off_rho1_train --cache_record runs/cache/check.json [--resume]
"""
import argparse, hashlib, json, os, subprocess, sys, time
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.clean_video import add_clean_temporal, set_mode, set_memory, collect_inputs
from mrv.noise import build_eps
import io
from mrv.cache_verify import make_bind, verify, file_code, resume_identity, read, hb, PINS

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--out", required=True)
p.add_argument("--K", type=int, default=4)
p.add_argument("--rho_gate", type=float, default=1.0, choices=[1.0, 0.95], help="A4 rollout / gate noise share: 1.0, or the signed 0.95 if T2 fails; stored as the checkpoint's rho")
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--frames", type=int, default=16)
p.add_argument("--clips", type=int, default=2)
p.add_argument("--steps", type=int, default=16000)
p.add_argument("--save_at", default="1000,2000,4000,6000,8000,10000,12000,14000,16000")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--init_ref", default="runs/vpriv/args.json", help="vpriv's args.json: recorded init_sha / seed / block names / base_sha / training inputs must equal ours (A4: vpriv's recorded init, seeds and draws)")
p.add_argument("--gate_record", default="", help="record written by scripts/vclean_gate.py; required unless --timing")
p.add_argument("--arm", default="vmix", choices=["vmix", "vclean"], help="A5 vmix (jitter + cached-source mix) or A4 vclean (t* only, key source)")
p.add_argument("--jitter", default="", help="memory timesteps (comma list); default vmix 1,41,81 / vclean t*")
p.add_argument("--mix_from", type=int, default=-1, help="vmix: cached source for one clip per update from update mix_from+1 (default 2000); vclean: 0 (never)")
p.add_argument("--cache", default="runs/cache/A20k_off_rho1_train", help="cached A@20k memory-off rho-1 train rollouts (<cache>/<scene>_s0.pt)")
p.add_argument("--pins", default=PINS, help="independent vmix pins (parity reference + dev digests); production must use council/vmix_pins.json")
p.add_argument("--cache_record", default="", help="passing scripts/cache_check.py record; required before update mix_from+1 (else the job holds)")
p.add_argument("--resume", action="store_true")
p.add_argument("--max_steps_this_job", type=int, default=0, help="stop after this many updates (timing / staged runs)")
p.add_argument("--timing", action="store_true", help="timing / test run: output dir must contain 'timing' or 'ctest'; weights are throwaway")
a = p.parse_args()
# ---------- AGREED r1 + A4 configuration (reject anything else) ----------
if a.K != 4 or a.frames != 16 or a.clips != 2:
    raise SystemExit(f"AGREED r1 / A4 fix K=4, 16-tick clips, 2 clips/update (got K={a.K}, frames={a.frames}, clips={a.clips})")
if a.timing and not any(s in a.out for s in ("timing", "ctest")):
    raise SystemExit("--timing runs must write to a *timing* / *ctest* directory")
PRIVATE_CLIPS, TRAIN_RHO = 2, 0.0  # A4: both clips private (rho 0 training noise), as vpriv
CODE_FILES = ["scripts/train_clean.py", "mrv/clean_video.py", "mrv/sd.py", "mrv/noise.py", "mrv/cache_verify.py"]
assert os.path.samefile(__file__, CODE_FILES[0]), "run from the repo root as scripts/train_clean.py"
CODE_SHA = hashlib.sha1(b"".join(open(f, "rb").read() for f in CODE_FILES)).hexdigest()[:12]
GATE, GATE_SHA = None, None
if not a.timing:  # production: the gate record must re-verify now and bind this exact code and gate rho
    if not a.gate_record:
        raise SystemExit("production training needs --gate_record (scripts/vclean_gate.py)")
    _gb = read(a.gate_record); GATE_SHA = hb(_gb)  # read ONCE: the gate verifies exactly these bytes (stdin) and we parse them
    _r = subprocess.run([sys.executable, "scripts/vclean_gate.py", "--verify", "-"], input=_gb, capture_output=True)
    if _r.returncode != 0 or f"sha256 {GATE_SHA}" not in _r.stdout.decode():
        raise SystemExit(f"gate record does not verify now: {_r.stdout.decode()[-800:]} {_r.stderr.decode()[-800:]}")
    GATE = json.loads(_gb)
    if GATE.get("arm") != a.arm:
        raise SystemExit(f"gate record is for arm {GATE.get('arm')}, not {a.arm}")
    if not GATE.get("valid") or GATE.get("rho_gate") not in (1.0, 0.95) or GATE["rho_gate"] != a.rho_gate:
        raise SystemExit(f"--rho_gate {a.rho_gate} != the gate record's {GATE.get('rho_gate')} (valid={GATE.get('valid')})")
    for f_ in CODE_FILES:
        if GATE.get("code_sha256", {}).get(f_) != hashlib.sha256(open(f_, "rb").read()).hexdigest():
            raise SystemExit(f"{f_} is not the code the gate's suite tested")
os.makedirs(a.out, exist_ok=True)
dev, AMP = "cuda", torch.bfloat16
SD = "sd-legacy/stable-diffusion-v1-5"
_dd = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); _dd.set_timesteps(25)
DDIM_TS = [int(x) for x in _dd.timesteps]  # the rollout's actual DDIM-25 schedule (recorded)
T_STAR = DDIM_TS[-1]  # its last timestep (expected 1)
if T_STAR > 81:
    raise SystemExit(f"A4: t* = {T_STAR} > 81 -> back to council")
print(f"t* = {T_STAR} (DDIM-25 last timestep; schedule {DDIM_TS})", flush=True)
JITTER = [int(x) for x in a.jitter.split(",")] if a.jitter else ([1, 41, 81] if a.arm == "vmix" else [T_STAR])
MIX_FROM = a.mix_from if a.mix_from >= 0 else (2000 if a.arm == "vmix" else 0)
if not all(j_ in DDIM_TS and j_ <= 81 for j_ in JITTER) or len(set(JITTER)) != len(JITTER):
    raise SystemExit(f"jitter {JITTER} must be distinct DDIM-25 timesteps <= 81")
if a.arm == "vclean" and (JITTER not in ([T_STAR], [1, 41, 81]) or MIX_FROM):
    raise SystemExit("arm vclean: A4 (memory at t*) or the A5 key + jitter control (1,41,81); key source only")
RECIPE = "vmix" if a.arm == "vmix" else ("vclean_A4" if JITTER == [T_STAR] else "vclean_keyjit_control")
if GATE is not None and GATE.get("recipe", "vclean_A4" if GATE.get("arm") == "vclean" else "vmix") != RECIPE:
    raise SystemExit(f"gate record is for recipe {GATE.get('recipe')}, not {RECIPE}")
if not a.timing and MIX_FROM and not a.cache_record:
    raise SystemExit("vmix production needs --cache_record <path> (it may not exist yet: the run picks it up at update mix_from)")
if not a.timing and a.arm == "vmix" and (JITTER != [1, 41, 81] or MIX_FROM != 2000):
    raise SystemExit(f"A5 vmix fixes jitter {{1, 41, 81}} and mix from update 2001 (got {JITTER}, {MIX_FROM})")
print(f"arm {a.arm}: memory timesteps {JITTER}, cached source from update {MIX_FROM + 1 if MIX_FROM else 'never'}", flush=True)
TRAIN = json.load(open(a.lists))["train"]
assert not set(TRAIN) & set(json.load(open(a.lists))["dev"]), "train / dev overlap"
RAYS = torch.load(io.BytesIO(read(a.rays)))
Fn = a.frames

# ---------- clips: Fn consecutive usable ticks, one sequence, no hard break (as vpriv) ----------
E, starts, ENC_SHA = {}, [], {}
for sc in sorted(TRAIN):
    _eb = read(f"{a.enc}/{sc}.pt"); ENC_SHA[sc] = hb(_eb)[:16]  # the digest of the encoding bytes actually used
    e = torch.load(io.BytesIO(_eb)); E[sc] = e; del _eb
    ok, seq, ticks, hard = e["ok"].tolist(), e["seq"].tolist(), e["ticks"].tolist(), e["hard"].tolist()
    for i in range(len(ticks) - Fn + 1):
        if all(ok[i + k] and not hard[i + k] and seq[i + k] == seq[i] and ticks[i + k] == ticks[i] + k for k in range(Fn)):
            starts.append((sc, i))
COVER = {sc: sum(1 for s_, _ in starts if s_ == sc) for sc in sorted(TRAIN)}
print(f"clips: {len(starts)} start positions of {Fn} ticks from {len({s for s, _ in starts})}/{len(E)} drives", flush=True)


_ab = read(a.init); A_SHA = hb(_ab)[:16]  # the trainer's ACTUAL A@20k bytes: hashed and parsed from the same buffer
ck0 = torch.load(io.BytesIO(_ab), map_location="cpu"); del _ab
if not a.timing and a.pins != PINS:
    raise SystemExit(f"production uses the reviewed pins {PINS}, not {a.pins}")
BIND, APPROVED = None, None
if MIX_FROM:
    if GATE is not None:  # production: the gate's bind (reviewed code, A@20k digest, drives, inputs) and, if passed, its approved record
        BIND = GATE.get("cache_bind") or {}
        if BIND.get("pins") != PINS:
            raise SystemExit(f"gate cache bind uses pins {BIND.get('pins')}, not the reviewed {PINS}")
        if BIND.get("ckpt_sha") != A_SHA or [BIND.get(k) for k in ("cache", "lists", "enc", "rays")] != [a.cache, a.lists, a.enc, a.rays]:
            raise SystemExit(f"gate cache bind {[BIND.get(k) for k in ('ckpt_sha', 'cache', 'lists', 'enc', 'rays')]} != this run ({A_SHA}, {a.cache}, {a.lists}, {a.enc}, {a.rays})")
        st_ = GATE.get("cache", {}).get("status", "")
        APPROVED = GATE["cache"]["sha256"] if st_ == "passed" else None
        if not (st_ == "passed" or st_.startswith("pending")):
            raise SystemExit(f"gate cache status {st_!r}")
    else:  # timing / tests: the same verifier against the files on disk
        BIND = make_bind(a.init, a.cache, a.lists, a.enc, a.rays, file_code(), a.pins, ckpt_sha=A_SHA)


def try_cache(pin):
    """shared verifier + approval / pin identity -> ({scene: fp16 latents}, record sha256, "") or (None, None, reason)"""
    ok, why, rsha, lat = verify(a.cache_record, BIND, {"ticks": {sc: E[sc]["ticks"] for sc in TRAIN}, "rays": RAYS, "enc_sha": ENC_SHA})
    if not ok:
        return None, None, why
    if APPROVED and rsha != APPROVED:
        return None, None, f"record {rsha[:12]} is not the gate-approved record {APPROVED[:12]}"
    if pin and rsha != pin:
        return None, None, f"record {rsha[:12]} differs from the record pinned in the checkpoint {pin[:12]}"
    return lat, rsha, ""


CACHE, CACHE_SHA, PIN, CACHE_WHY = None, None, None, "no cached source (mix_from 0)"
G_JIT = torch.Generator().manual_seed(a.seed * 1000003 + 1)  # resumable stream 2: memory timestep per clip
G_SRC = torch.Generator().manual_seed(a.seed * 1000003 + 2)  # resumable stream 3: which clip reads the cache

# ---------- model: frozen A + clean temporal blocks ----------
r = ck0["args"]["rank"]  # ck0: parsed above from the hashed A bytes
assert "hist" not in ck0 and "temporal" not in ck0, "init must be a plain no-memory renderer (A)"
if not a.init.endswith("runs/longA/ckpt_20000.pt") or ck0.get("step") != 20000 or ck0["args"].get("arm") != "A":
    raise SystemExit(f"A4: init must be A@20k (runs/longA/ckpt_20000.pt), got {a.init} step {ck0.get('step')} arm {ck0['args'].get('arm')}")
_h = hashlib.sha256()
for k_ in sorted(ck0["lora"]):
    _h.update(k_.encode()); _h.update(ck0["lora"][k_].float().numpy().tobytes())
for k_ in sorted(ck0["adapter"]):
    _h.update(k_.encode()); _h.update(ck0["adapter"][k_].float().numpy().tobytes())
BASE_SHA = _h.hexdigest()[:16]
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS))
res = unet.load_state_dict(ck0["lora"], strict=False)
assert not res.unexpected_keys, f"unexpected LoRA keys: {res.unexpected_keys[:5]}"
lora_names = {n for n, _ in unet.named_parameters() if "lora" in n}
assert lora_names == set(ck0["lora"]) and len(lora_names) > 0, "every LoRA tensor must come from the init checkpoint"
unet.requires_grad_(False); unet.to(dev, AMP)
LORA_REF = {n: q for n, q in unet.named_parameters() if "lora" in n}  # references survive wrapping (names gain ".inner.")
LORA_PRE = {n: q.detach().clone() for n, q in LORA_REF.items()}
torch.manual_seed(a.seed)  # explicit, reproducible temporal-block init: same construction as vpriv -> vpriv's recorded init
TW = add_clean_temporal(unet, a.K)
INIT_SHA = hashlib.sha256(b"".join(q.detach().float().cpu().numpy().tobytes() for w in TW for q in w.temporal.parameters())).hexdigest()[:16]
BLOCK_NAMES = [n for n, m in unet.named_modules() if m.__class__.__name__ == "WithCleanTemporal"]
assert len(BLOCK_NAMES) == 16 and len(TW) == 16, len(TW)
INIT_MATCH, REF_DIFF = None, ["init_ref missing"]
if os.path.exists(a.init_ref):  # vpriv's recorded init, seed, blocks, frozen base and training inputs
    ref_ = json.load(open(a.init_ref))
    REF_DIFF = [k_ for k_, v_ in (("init_sha", INIT_SHA), ("seed", a.seed), ("block_names", BLOCK_NAMES), ("base_sha", BASE_SHA), ("lists", a.lists),
                                  ("enc", a.enc), ("rays", a.rays), ("frames", a.frames), ("clips", a.clips)) if ref_.get(k_) != v_]
    INIT_MATCH = not REF_DIFF
    print(f"init_sha {INIT_SHA} vs {a.init_ref} {ref_.get('init_sha')}: {'MATCH (init, seed, blocks, base, inputs)' if INIT_MATCH else 'DIFFERENT ' + str(REF_DIFF)}", flush=True)
if not a.timing and not INIT_MATCH:
    raise SystemExit(f"A4: vclean must start from vpriv's recorded init, seeds, base and inputs (init_ref {a.init_ref}: differs in {REF_DIFF})")
for w in TW:
    w.temporal.to(dev).float().requires_grad_(True)
adapter = CameraAdapter(n_styles=ck0["args"].get("n_styles", 3)); adapter.load_state_dict(ck0["adapter"], strict=True)
adapter.requires_grad_(False); adapter.to(dev).eval()
AD_PRE = {n: q.detach().clone() for n, q in adapter.state_dict().items()}
unet.eval()  # frozen renderer in eval mode (temporal blocks have no dropout)
params = [q for w in TW for q in w.temporal.parameters()]
trainable = [q for q in list(unet.parameters()) + list(adapter.parameters()) if q.requires_grad]
assert {id(q) for q in trainable} == {id(q) for q in params}, "only the temporal blocks may train"
print(f"temporal params: {sum(q.numel() for q in params)}; frozen base + A LoRA ({len(lora_names)} tensors) + A adapter", flush=True)
opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
acp = DDPMScheduler.from_pretrained(SD, subfolder="scheduler").alphas_cumprod.to(dev)
assert torch.equal(acp.cpu(), _dd.alphas_cumprod.cpu()), "DDPM / DDIM noise schedules differ"
start, ck = 0, f"{a.out}/ckpt.pt"
if os.path.exists(ck):
    if not a.resume:
        raise SystemExit(f"{ck} exists: pass --resume or use a fresh --out")
    s_ = torch.load(ck, map_location="cpu")
    if s_.get("timing") and not a.timing:
        raise SystemExit("refusing to resume production training from a timing / test checkpoint")
    rec_ = s_.get("args", {}).get("clean", {})
    for key_, want_ in (("seed", a.seed), ("lists", a.lists), ("enc", a.enc), ("rays", a.rays), ("frames", a.frames), ("clips", a.clips), ("rho_gate", a.rho_gate),
                        ("arm", a.arm), ("cache", a.cache)):
        if rec_.get(key_) != want_:
            raise SystemExit(f"incompatible resume: recipe {key_} {rec_.get(key_)} != {want_}")
    for key_, want_ in (("K", a.K), ("init", a.init), ("rho", a.rho_gate), ("t_star", T_STAR), ("base_sha", BASE_SHA), ("init_sha", INIT_SHA),
                        ("block_names", BLOCK_NAMES), ("code_sha", CODE_SHA), ("memory", "tstar_final_call"), ("ddim_timesteps", DDIM_TS),
                        ("arm", a.arm), ("jitter", JITTER), ("mix_from", MIX_FROM), ("cache", a.cache)):
        if s_.get(key_) != want_:
            raise SystemExit(f"incompatible resume: {key_} {s_.get(key_)} != {want_}")
    if len(s_["temporal"]) != len(TW):
        raise SystemExit("incompatible resume: block count")
    for w, sd_ in zip(TW, s_["temporal"]):
        w.temporal.load_state_dict(sd_, strict=True)
    opt.load_state_dict(s_["opt"]); start = s_["step"]
    G_JIT.set_state(s_["rng"]["jitter"]); G_SRC.set_state(s_["rng"]["source"])  # resumable jitter / source streams
    PIN = s_.get("cache_record_sha256")  # the record this run first used (pinned)
    why_ = resume_identity({"gate": s_.get("gate_record_sha256"), "approved": s_.get("cache_approved_sha256"), "pin": PIN},
                           {"gate": GATE_SHA, "approved": APPROVED}, MIX_FROM, start)
    if why_:
        raise SystemExit(f"incompatible resume: {why_}")
    print("resumed at", start, "pinned cache record", PIN, flush=True)
if MIX_FROM:
    CACHE, CACHE_SHA, CACHE_WHY = try_cache(PIN)
    print("cached source:", f"record {a.cache_record} verified ({len(CACHE)} drives)" if CACHE is not None else f"NOT available yet: {CACHE_WHY}", flush=True)
    if CACHE is None and (PIN or APPROVED or start > MIX_FROM):
        raise SystemExit(f"the {'pinned' if PIN else 'gate-approved' if APPROVED else 'required'} cache record does not verify: {CACHE_WHY}")
json.dump({**vars(a), "recipe": RECIPE, "jitter": JITTER, "mix_from": MIX_FROM, "cache_record_sha256": PIN, "cache_bind": BIND, "a_sha": A_SHA, "code_sha": CODE_SHA, "init_sha": INIT_SHA, "init_match": INIT_MATCH, "init_ref_diff": REF_DIFF, "base_sha": BASE_SHA,
           "t_star": T_STAR, "ddim_timesteps": DDIM_TS, "gate_record_sha256": GATE_SHA, "gate": GATE,
           "block_names": BLOCK_NAMES, "clip_starts": len(starts), "drives_with_clips": sum(1 for v in COVER.values() if v),
           "drives_without_clips": [k for k, v in COVER.items() if not v], "clips_per_drive": COVER}, open(f"{a.out}/args.json", "w"), indent=1)
TRAIN_NOISE = {"private_clips": PRIVATE_CLIPS, "rho": TRAIN_RHO, "t_rule": f"U{{0..999}} minus t*={T_STAR} (t* draws redrawn, separate generator)",
               "memory": f"frozen A, deltas off, source latents noised to a per-clip timestep from {JITTER} (resumable stream) with private per-tick "
                         f"draws, same dropout; source = real key" + (f", from update {MIX_FROM + 1} one clip per update cached A@20k latents (coin, resumable stream)" if MIX_FROM else "")}


def save(step, path):
    d = {"temporal": [w.temporal.state_dict() for w in TW], "opt": opt.state_dict(), "step": step, "init": a.init, "K": a.K, "frames": Fn,
         "arm": a.arm, "recipe": RECIPE, "jitter": JITTER, "mix_from": MIX_FROM, "cache": a.cache, "cache_record_sha256": PIN,
         "rng": {"jitter": G_JIT.get_state(), "source": G_SRC.get_state()}, "memory": "tstar_final_call", "t_star": T_STAR, "ddim_timesteps": DDIM_TS, "rho": a.rho_gate, "train_noise": TRAIN_NOISE,
         "gate_record_sha256": GATE_SHA, "cache_approved_sha256": APPROVED,
         "noise_policy": "one t per clip (t* excluded); fully private eps per tick; memory = features at the clip's memory timestep of the source latents",
         "code_sha": CODE_SHA, "base_sha": BASE_SHA, "init_sha": INIT_SHA, "block_names": BLOCK_NAMES, "timing": a.timing,
         "args": {**ck0["args"], "clean": vars(a)}, "adapter": ck0["adapter"], "lora": ck0["lora"]}
    torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)


def memory_of(x0, eps_m, ctx, tm=None):
    """no-grad frozen-A forward at timestep tm per row (default t*; temporal deltas off): per-block input features, detached"""
    tm = torch.full((x0.shape[0],), T_STAR, device=dev) if tm is None else tm
    ab = acp[tm].view(-1, 1, 1, 1); xm = ab.sqrt() * x0 + (1 - ab).sqrt() * eps_m
    set_mode(TW, "off")
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        return collect_inputs(TW, lambda: unet(xm, tm, encoder_hidden_states=ctx))


# ---------- copy probe (AGREED r1, adapted): 8 fixed dev clips of K+1 ticks, t = 900 / 700 / 300 ----------
DEVL = json.load(open(a.lists))["dev"]
gp = torch.Generator().manual_seed(4242)
_pc = {}  # dev drive -> candidate (tokens, lat) windows of K+1 legal ticks (same selection as train_causal / noise_probe)
for sc in DEVL:
    e = torch.load(f"{a.enc}/{sc}.pt")
    ok, seq, ticks, hard = e["ok"].tolist(), e["seq"].tolist(), e["ticks"].tolist(), e["hard"].tolist()
    L = a.K + 1
    st_ = [i for i in range(len(ticks) - L + 1) if all(ok[i + k] and not hard[i + k] and seq[i + k] == seq[i] and ticks[i + k] == ticks[i] + k for k in range(L))]
    _pc[sc] = [(e["tokens"][i:i + L].float(), e["lat"][i:i + L].float()) for i in (st_[k_] for k_ in torch.randperm(len(st_), generator=gp)[:2].tolist())]
PROBE, DONOR = [], []  # 8 fixed (recipient, donor) pairs from DIFFERENT dev drives
for pi in range(8):
    rsc, dsc = DEVL[pi % len(DEVL)], DEVL[(pi + 1) % len(DEVL)]
    assert rsc != dsc
    PROBE.append((rsc,) + _pc[rsc][(pi // len(DEVL)) % len(_pc[rsc])]); DONOR.append((dsc,) + _pc[dsc][0])
P_EPS = [torch.randn(1, 4, 32, 64, generator=gp) for _ in PROBE]  # drawn as in train_causal (keeps P_OWN identical); unused: rho 0
P_OWN = [torch.randn(a.K + 1, 4, 32, 64, generator=gp) for _ in PROBE]  # private per-tick noise of the noisy pass
P_MEM = [torch.randn(a.K + 1, 4, 32, 64, generator=gp) for _ in PROBE]  # private per-tick noise of the t* memory pass


def probe(step):
    """loss on the last tick of each probe clip: blocks on / off / memory from another drive's past / current cameras swapped"""
    L = a.K + 1; res = {}
    st5 = torch.full((L,), 2, dtype=torch.long, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        for tv in (900, 700, 300):
            acc = {"on": 0.0, "off": 0.0, "past_other": 0.0, "cams_other": 0.0}
            for pi, (sc, tk, lt) in enumerate(PROBE):
                osc, otk, olt = DONOR[pi]
                eps = P_OWN[pi].to(dev); tt = torch.full((L,), tv, device=dev); ab = acp[tt].view(-1, 1, 1, 1)
                xt = ab.sqrt() * lt.to(dev) + (1 - ab).sqrt() * eps
                rays = RAYS[sc][None].expand(L, *RAYS[sc].shape).to(dev).float()
                ctx = adapter(tk.to(dev), rays, st5)
                mem_true = memory_of(lt.to(dev), P_MEM[pi].to(dev), ctx)
                x0o = lt.clone(); tko = tk.clone(); ro = rays.clone()
                x0o[:-1] = olt[:-1]; tko[:-1] = otk[:-1]; ro[:-1] = RAYS[osc][None].to(dev).float()
                mem_other = memory_of(x0o.to(dev), P_MEM[pi].to(dev), adapter(tko.to(dev), ro, st5))  # only past rows are read
                tkc = tk.clone(); rc = rays.clone(); tkc[-1] = otk[-1]; rc[-1] = RAYS[osc].to(dev).float()
                ctx_c = adapter(tkc.to(dev), rc, st5)
                for kind in acc:
                    c_ = ctx_c if kind == "cams_other" else ctx
                    if kind == "off":
                        set_mode(TW, "off")
                    else:
                        set_mode(TW, "clip", L); set_memory(TW, mem_other if kind == "past_other" else mem_true)
                    pr = unet(xt, tt, encoder_hidden_states=c_).sample.float()
                    acc[kind] += ((pr[-1] - eps[-1]) ** 2).mean().item() / len(PROBE)
            res[tv] = acc
    set_memory(TW, None)
    flag = any(res[tv]["cams_other"] - res[tv]["on"] < res[tv]["past_other"] - res[tv]["on"] for tv in (900, 700))  # unrounded
    rec = {"step": step, "probe": {tv: {k: round(v, 5) for k, v in d.items()} for tv, d in res.items()}, "copy_flag": flag}
    print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()


SAVE = set(int(x) for x in a.save_at.split(","))
log = open(f"{a.out}/log.jsonl", "a")
style = torch.full((a.clips * Fn,), 2, dtype=torch.long, device=dev)
w_loss = torch.zeros(a.clips, Fn, 1, 1, 1, device=dev); w_loss[:, 1:] = 1.0  # loss on ticks 1..Fn-1
end = a.steps if not a.max_steps_this_job else min(a.steps, start + a.max_steps_this_job)
PICK_H = hashlib.sha1()
T_WIN, N_WIN = time.time(), 0  # end-to-end window timing (data + compute + logging + saving + probes)
t_data = t_gpu = t_mem = 0.0; n_acc = 0
HOLD = None
for step in range(start, end):
    if MIX_FROM and step >= MIX_FROM and CACHE is None:  # update step+1 reads the cache: re-check the record now (it may have arrived)
        CACHE, CACHE_SHA, CACHE_WHY = try_cache(PIN)
        if CACHE is not None:
            rec = {"cache_acquired": step, "record": a.cache_record, "record_sha256": CACHE_SHA}
            print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    if MIX_FROM and step >= MIX_FROM and PIN is None and CACHE is not None:  # first use: pin this record for the rest of the run
        PIN = CACHE_SHA
        rec = {"cache_pinned": step, "record_sha256": PIN}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    if MIX_FROM and step >= MIX_FROM and CACHE is None:  # still no verified cache: hold here (A5)
        save(step, ck); HOLD = step
        rec = {"hold": step, "reason": CACHE_WHY, "resume": "rerun with --resume --cache_record <passing cache_check record>"}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        break
    t0 = time.time()
    g = torch.Generator().manual_seed(a.seed * 100003 + step)  # vpriv's draws, in vpriv's order
    pick = torch.randint(0, len(starts), (a.clips,), generator=g).tolist()
    t = torch.randint(0, 1000, (a.clips,), generator=g)
    eps_clip = torch.randn(a.clips, 1, 4, 32, 64, generator=g); eps_own = torch.randn(a.clips, Fn, 4, 32, 64, generator=g)
    eps = build_eps(eps_clip, eps_own, TRAIN_RHO, PRIVATE_CLIPS, step)  # mrv/noise.py: both clips private == eps_own
    drop = (torch.rand(a.clips, generator=g) < 0.1).repeat_interleave(Fn)
    gm = torch.Generator().manual_seed(10 ** 9 + a.seed * 100003 + step)  # vclean-only draws (t* redraw, memory noise)
    t_re = torch.randint(0, 999, (a.clips,), generator=gm); t_re = t_re + (t_re >= T_STAR).long()  # uniform over {0..999} \ {t*}
    t = torch.where(t == T_STAR, t_re, t)
    assert not bool((t == T_STAR).any())
    eps_mem = torch.randn(a.clips * Fn, 4, 32, 64, generator=gm)
    batch_sha = hashlib.sha1(str(pick).encode() + t.numpy().tobytes() + eps[:, 0, 0, 0].numpy().tobytes() + drop.numpy().tobytes()).hexdigest()[:10]
    PICK_H.update(str(pick).encode() + t.numpy().tobytes() + drop.numpy().tobytes())
    mem_t = [JITTER[k_] for k_ in torch.randint(0, len(JITTER), (a.clips,), generator=G_JIT).tolist()]  # stream 2 (every update)
    src = ["key"] * a.clips
    if MIX_FROM and step >= MIX_FROM:  # update > mix_from: one clip reads the cache, coin on stream 3
        src[torch.randint(0, a.clips, (1,), generator=G_SRC).item()] = "cache"
    mem_sha = hashlib.sha1(t.numpy().tobytes() + eps_mem[:, 0, 0, 0].numpy().tobytes() + str((mem_t, src)).encode()).hexdigest()[:10]  # all vclean-only draws
    tok, lat, ray, msrc = [], [], [], []
    for ci, (sc, i) in enumerate(starts[k] for k in pick):
        e = E[sc]
        tok.append(e["tokens"][i:i + Fn].float()); lat.append(e["lat"][i:i + Fn].float()); ray.append(RAYS[sc][None].expand(Fn, *RAYS[sc].shape))
        msrc.append(CACHE[sc][i:i + Fn].float() if src[ci] == "cache" else lat[-1])  # memory source: cached A latents or real key
    tok = torch.cat(tok).to(dev); x0 = torch.cat(lat).to(dev); ray = torch.cat(ray).to(dev).float(); xsrc = torch.cat(msrc).to(dev)
    tt = t.repeat_interleave(Fn).to(dev); eps = eps.to(dev); eps_mem = eps_mem.to(dev)
    tmem = torch.tensor(mem_t).repeat_interleave(Fn).to(dev)
    torch.cuda.synchronize(); t1 = time.time()
    ab = acp[tt].view(-1, 1, 1, 1); xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    for g_ in opt.param_groups:
        g_["lr"] = 1e-4 * min(1, (step + 1) / 500)
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        ctx = adapter(tok, ray, style, drop.to(dev))  # the same dropout context feeds the memory pass and the noisy pass
    mem = memory_of(xsrc, eps_mem, ctx, tmem)  # target stays the real key x0
    torch.cuda.synchronize(); t1m = time.time()
    set_mode(TW, "clip", Fn); set_memory(TW, mem)
    with torch.autocast("cuda", dtype=AMP):
        pred = unet(xt, tt, encoder_hidden_states=ctx).sample
    set_memory(TW, None); del mem
    err = ((pred.float() - eps) ** 2).view(a.clips, Fn, 4, 32, 64)
    loss = (err * w_loss).sum() / (w_loss.sum() * 4 * 32 * 64)
    if step == start:
        OUT_PRE = [w.temporal.out.weight.detach().clone() for w in TW]
    opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if step == start:  # guard: the frozen renderer must be bit-identical after a real optimizer step; temporal params must move
        assert all(torch.equal(LORA_REF[n], LORA_PRE[n]) for n in LORA_REF), "LoRA changed!"
        assert all(torch.equal(q, AD_PRE[n]) for n, q in adapter.state_dict().items()), "adapter changed!"
        moved = sum(int(not torch.equal(w.temporal.out.weight, o_)) for w, o_ in zip(TW, OUT_PRE))
        assert moved == 16, f"temporal output weights moved in {moved}/16 blocks"
        print(f"frozen check after first update: {len(LORA_REF)} LoRA + {len(AD_PRE)} adapter tensors bit-identical; 16/16 temporal out weights moved", flush=True)
    torch.cuda.synchronize(); t2 = time.time()
    t_data += t1 - t0; t_mem += t1m - t1; t_gpu += t2 - t1; n_acc += 1
    if (step + 1) % 100 == 0 or step == end - 1 or step == start or a.timing:  # timing / test runs log every update
        rec = {"update": step + 1, "e2e_window_s": round((time.time() - T_WIN) / max(1, step + 1 - start - N_WIN), 3), "loss": round(loss.item(), 4), "gn": round(gn.item(), 4),
               "out_norm": round(sum(w.temporal.out.weight.norm().item() ** 2 for w in TW) ** 0.5, 4),
               "age_bias_abs": round(sum(w.temporal.age_bias.abs().sum().item() for w in TW), 4),
               "step_s": round(t_gpu / n_acc, 3), "mem_s": round(t_mem / n_acc, 3), "data_s": round(t_data / n_acc, 3),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "batch_sha": batch_sha, "mem_sha": mem_sha, "mem_t": mem_t, "mem_src": src, "pick_sha_cum": PICK_H.hexdigest()[:10],
               "private_clips": PRIVATE_CLIPS, "t_star": T_STAR, "arm": a.arm, "code_sha": CODE_SHA}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush(); t_data = t_gpu = t_mem = 0.0; n_acc = 0
        T_WIN, N_WIN = time.time(), step + 1 - start
    if (step + 1) % 500 == 0 or step + 1 == end:
        save(step + 1, ck); probe(step + 1)
    if step + 1 in SAVE:
        save(step + 1, f"{a.out}/ckpt_{step + 1}.pt")
if HOLD is not None:
    print(f"CLEAN_TRAIN_HOLD {HOLD}: saved at update {HOLD}; update {HOLD + 1} needs the cached source ({CACHE_WHY}); resume with --resume --cache_record")
else:
    print("CLEAN_TRAIN_DONE", end)
