"""Causal (live) video adapter on the FROZEN A@20k renderer (council/AGREED-causal-r1.md).

Only the 16 new causal temporal blocks (mrv/causal_video.py) train; the SD 1.5 UNet, A's LoRA and A's CameraAdapter are frozen
(asserted). Each update: 2 clips of 16 consecutive usable ticks (same sequence, no hard break) from the 144 train drives
(data/enc_tpv10); ONE diffusion timestep per clip (uniform 0..999) shared by its ticks; noise eps = sqrt(rho) eps_clip +
sqrt(1-rho) eps_tick (rho fixed before training from the off sweep; the rollout uses the same rho); epsilon MSE in fp32
over ticks 1..15 (tick 0 is context only; its temporal delta is exactly zero by construction); whole-clip camera dropout 10%;
style 2; AdamW 1e-4, no weight decay, 500-update warm-up, grad clip 1.0, bf16 autocast, fp32 temporal parameters.
Logged every 100 updates: loss, grad norm, temporal out-norm, step_s (GPU), data_s (batch assembly), peak memory, batch_sha;
code_sha + args in args.json and every log row.

  python scripts/train_causal.py --init runs/longA/ckpt_20000.pt --out runs/vlive --steps 8000 [--K 4] [--resume]
"""
import argparse, hashlib, json, os, time
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.causal_video import add_causal_temporal, set_mode
from mrv.noise import build_eps

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--out", required=True)
p.add_argument("--K", type=int, default=4)
p.add_argument("--rho", type=float, required=True, help="noise share across ticks: eps = sqrt(rho) eps_clip + sqrt(1-rho) eps_tick (council r3); the rollout uses the same rho (stored in the checkpoint)")
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--frames", type=int, default=16)
p.add_argument("--clips", type=int, default=2)
p.add_argument("--steps", type=int, default=8000)
p.add_argument("--save_at", default="1000,2000,4000,6000,8000,10000,12000,14000,16000,20000")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--resume", action="store_true")
p.add_argument("--max_steps_this_job", type=int, default=0, help="stop after this many updates (timing / staged runs)")
p.add_argument("--private_clips", type=int, default=0, choices=[0, 1, 2],
               help="A3 draft (council/causal-video-a3r2*): number of the 2 clips per update trained with FULLY private noise per tick (rho 0 for those clips); rollout rho unchanged")
p.add_argument("--rho_decision", default="", help="JSON written by scripts/rho_decision.py; required unless --timing (AGREED: rho fixed before update 1)")
p.add_argument("--timing", action="store_true", help="timing / test run: output dir must contain 'timing' or 'ctest'; weights are throwaway")
a = p.parse_args()
# ---------- AGREED-causal-r1 configuration (reject anything else) ----------
if a.K != 4 or a.frames != 16 or a.clips != 2:
    raise SystemExit(f"AGREED r1 fixes K=4, 16-tick clips, 2 clips/update (got K={a.K}, frames={a.frames}, clips={a.clips})")
if a.rho not in (0.5, 0.8, 0.9, 0.95):
    raise SystemExit(f"AGREED r1 + A2: rho must be 0.5 / 0.8 / 0.9 / 0.95 (got {a.rho}); never 1.0")
if a.timing:
    if not any(s in a.out for s in ("timing", "ctest")):
        raise SystemExit("--timing runs must write to a *timing* / *ctest* directory")
else:
    if not a.rho_decision:
        raise SystemExit("training needs --rho_decision (the recorded off-sweep decision, AGREED r1)")
    dec = json.load(open(a.rho_decision))
    import subprocess as _sp, tempfile as _tf
    # rebuild the decision NOW from the canonical evidence files and the bound clock (council/rho_fallback_state.json), with
    # the same CPU-tested rule; it must give the recorded mode / rho / evidence digests (no cached flags trusted)
    _tmp = _tf.mktemp(suffix=".json", dir=os.environ.get("TMPDIR", "/tmp"))
    _r = _sp.run([".venv/bin/python", "scripts/rho_decision.py", "--out", _tmp], capture_output=True, text=True)
    if not os.path.exists(_tmp):
        raise SystemExit(f"could not rebuild the rho decision: {_r.stdout[-400:]} {_r.stderr[-400:]}")
    now_ = json.load(open(_tmp))
    rec_keys_ = lambda d: {k: (v.get("admissible"), v.get("metrics_sha"), v.get("meets")) for k, v in d["rows"].items()}
    if not dec.get("valid") or (now_.get("mode"), now_.get("chosen")) != (dec.get("mode"), dec.get("chosen")) or dec.get("chosen") != a.rho \
            or rec_keys_(now_) != rec_keys_(dec) or now_.get("rho_rule_sha") != dec.get("rho_rule_sha"):
        raise SystemExit(f"rho decision does not hold now: recorded {dec.get('mode')} {dec.get('chosen')}, rebuilt {now_.get('mode')} "
                         f"{now_.get('chosen')} ({now_.get('note')}); --rho {a.rho}")
    _sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
    for rho_, row2_ in dec["rows"].items():  # the evidence must still be the files the decision was made on
        if row2_.get("metrics_sha") is None:
            continue  # row never scored (not reached by the ordered rule)
        if _sha(row2_["metrics_file"]) != row2_["metrics_sha"]:
            raise SystemExit(f"rho decision evidence changed: {row2_['metrics_file']}")
    if _sha(dec["comparator"]) != dec["comparator_sha"]:
        raise SystemExit("rho decision comparator changed")
os.makedirs(a.out, exist_ok=True)
dev, AMP = "cuda", torch.bfloat16
SD = "sd-legacy/stable-diffusion-v1-5"
CODE_SHA = hashlib.sha1(b"".join(open(f, "rb").read() for f in [__file__, "mrv/causal_video.py", "mrv/sd.py", "mrv/noise.py"])).hexdigest()[:12]
TRAIN = json.load(open(a.lists))["train"]
assert not set(TRAIN) & set(json.load(open(a.lists))["dev"]), "train / dev overlap"
RAYS = torch.load(a.rays)
Fn = a.frames

# ---------- clips: Fn consecutive usable ticks, one sequence, no hard break ----------
E, starts = {}, []
for sc in sorted(TRAIN):
    e = torch.load(f"{a.enc}/{sc}.pt"); E[sc] = e
    ok, seq, ticks, hard = e["ok"].tolist(), e["seq"].tolist(), e["ticks"].tolist(), e["hard"].tolist()
    for i in range(len(ticks) - Fn + 1):
        if all(ok[i + k] and not hard[i + k] and seq[i + k] == seq[i] and ticks[i + k] == ticks[i] + k for k in range(Fn)):
            starts.append((sc, i))
COVER = {sc: sum(1 for s_, _ in starts if s_ == sc) for sc in sorted(TRAIN)}
print(f"clips: {len(starts)} start positions of {Fn} ticks from {len({s for s, _ in starts})}/{len(E)} drives", flush=True)

# ---------- model: frozen A + causal temporal blocks ----------
ck0 = torch.load(a.init, map_location="cpu"); r = ck0["args"]["rank"]
assert "hist" not in ck0 and "temporal" not in ck0, "init must be a plain no-memory renderer (A)"
if not a.init.endswith("runs/longA/ckpt_20000.pt") or ck0.get("step") != 20000 or ck0["args"].get("arm") != "A":
    raise SystemExit(f"AGREED r1: init must be A@20k (runs/longA/ckpt_20000.pt), got {a.init} step {ck0.get('step')} arm {ck0['args'].get('arm')}")
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
torch.manual_seed(a.seed)  # explicit, reproducible temporal-block init (vlive's own init was unseeded and cannot be recovered)
TW = add_causal_temporal(unet, a.K)
INIT_SHA = hashlib.sha256(b"".join(q.detach().float().cpu().numpy().tobytes() for w in TW for q in w.temporal.parameters())).hexdigest()[:16]  # seeded init (recorded)
BLOCK_NAMES = [n for n, m in unet.named_modules() if m.__class__.__name__ == "WithCausalTemporal"]
assert len(BLOCK_NAMES) == 16
assert len(TW) == 16, len(TW)
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
start, ck = 0, f"{a.out}/ckpt.pt"
if os.path.exists(ck):
    if not a.resume:
        raise SystemExit(f"{ck} exists: pass --resume or use a fresh --out")
    s_ = torch.load(ck, map_location="cpu")
    if s_.get("timing") and not a.timing:
        raise SystemExit("refusing to resume production training from a timing / test checkpoint")
    rec_ = s_.get("args", {}).get("causal", {})
    for key_, want_ in (("seed", a.seed), ("lists", a.lists), ("enc", a.enc), ("frames", a.frames), ("clips", a.clips), ("rho_decision", a.rho_decision), ("private_clips", a.private_clips)):
        if rec_.get(key_) != want_:
            raise SystemExit(f"incompatible resume: recipe {key_} {rec_.get(key_)} != {want_}")
    for key_, want_ in (("K", a.K), ("init", a.init), ("rho", a.rho), ("base_sha", BASE_SHA), ("block_names", BLOCK_NAMES), ("code_sha", CODE_SHA)):
        if s_.get(key_) != want_:
            raise SystemExit(f"incompatible resume: {key_} {s_.get(key_)} != {want_}")
    if len(s_["temporal"]) != len(TW):
        raise SystemExit("incompatible resume: block count")
    for w, sd_ in zip(TW, s_["temporal"]):
        w.temporal.load_state_dict(sd_, strict=True)
    opt.load_state_dict(s_["opt"]); start = s_["step"]; print("resumed at", start, flush=True)
json.dump({**vars(a), "code_sha": CODE_SHA, "init_sha": INIT_SHA, "base_sha": BASE_SHA, "block_names": BLOCK_NAMES, "clip_starts": len(starts),
           "drives_with_clips": sum(1 for v in COVER.values() if v), "drives_without_clips": [k for k, v in COVER.items() if not v],
           "clips_per_drive": COVER}, open(f"{a.out}/args.json", "w"), indent=1)


def save(step, path):
    d = {"temporal": [w.temporal.state_dict() for w in TW], "opt": opt.state_dict(), "step": step, "init": a.init, "K": a.K,
         "frames": Fn, "noise_policy": "one t per clip; eps = sqrt(rho) eps_clip + sqrt(1-rho) eps_tick", "rho": a.rho, "train_noise": {"private_clips": a.private_clips, "rho_shared_clips": a.rho}, "code_sha": CODE_SHA, "base_sha": BASE_SHA, "block_names": BLOCK_NAMES, "timing": a.timing,
         "args": {**ck0["args"], "causal": vars(a)}, "adapter": ck0["adapter"], "lora": ck0["lora"]}
    torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)


# ---------- copy probe (AGREED-causal-r1): 8 fixed dev clips of K+1 ticks, t = 900 / 700 / 300 ----------
DEVL = json.load(open(a.lists))["dev"]
gp = torch.Generator().manual_seed(4242)
_pc = {}  # dev drive -> candidate (tokens, lat) windows of K+1 legal ticks
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
P_EPS = [torch.randn(1, 4, 32, 64, generator=gp) for _ in PROBE]  # one draw per probe clip (shared over its ticks, rho-mixed below)
P_OWN = [torch.randn(a.K + 1, 4, 32, 64, generator=gp) for _ in PROBE]


def probe(step):
    """loss on the last tick of each probe clip: blocks on / off / past from another drive / current cameras from another drive"""
    L = a.K + 1; res = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        for tv in (900, 700, 300):
            acc = {"on": 0.0, "off": 0.0, "past_other": 0.0, "cams_other": 0.0}
            for pi, (sc, tk, lt) in enumerate(PROBE):
                osc, otk, olt = DONOR[pi]
                eps = (a.rho ** 0.5 * P_EPS[pi] + (1 - a.rho) ** 0.5 * P_OWN[pi]).to(dev)
                tt = torch.full((L,), tv, device=dev); ab = acp[tt].view(-1, 1, 1, 1)
                for kind in acc:
                    x0 = lt.clone(); tkk = tk.clone(); rays = RAYS[sc][None].expand(L, *RAYS[sc].shape).clone()
                    if kind == "past_other":
                        x0[:-1] = olt[:-1]; tkk[:-1] = otk[:-1]; rays[:-1] = RAYS[osc][None]
                    if kind == "cams_other":
                        tkk[-1] = otk[-1]; rays[-1] = RAYS[osc]
                    x0 = x0.to(dev); xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
                    ctx = adapter(tkk.to(dev), rays.to(dev).float(), torch.full((L,), 2, dtype=torch.long, device=dev))
                    set_mode(TW, "off" if kind == "off" else "clip", L)
                    pr = unet(xt, tt, encoder_hidden_states=ctx).sample.float()
                    acc[kind] += ((pr[-1] - eps[-1]) ** 2).mean().item() / len(PROBE)
            res[tv] = acc
    flag = any(res[tv]["cams_other"] - res[tv]["on"] < res[tv]["past_other"] - res[tv]["on"] for tv in (900, 700))  # unrounded
    rec = {"step": step, "probe": {tv: {k: round(v, 5) for k, v in d.items()} for tv, d in res.items()}, "copy_flag": flag}
    print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    set_mode(TW, "clip", Fn)


SAVE = set(int(x) for x in a.save_at.split(","))
log = open(f"{a.out}/log.jsonl", "a")
style = torch.full((a.clips * Fn,), 2, dtype=torch.long, device=dev)
w_loss = torch.zeros(a.clips, Fn, 1, 1, 1, device=dev); w_loss[:, 1:] = 1.0  # loss on ticks 1..Fn-1
end = a.steps if not a.max_steps_this_job else min(a.steps, start + a.max_steps_this_job)
PICK_H = hashlib.sha1()
T_WIN, N_WIN = time.time(), 0  # end-to-end window timing (data + compute + logging + saving + probes)
t_data = t_gpu = 0.0; n_acc = 0
for step in range(start, end):
    t0 = time.time()
    g = torch.Generator().manual_seed(a.seed * 100003 + step)
    pick = torch.randint(0, len(starts), (a.clips,), generator=g).tolist()
    t = torch.randint(0, 1000, (a.clips,), generator=g)
    eps_clip = torch.randn(a.clips, 1, 4, 32, 64, generator=g); eps_own = torch.randn(a.clips, Fn, 4, 32, 64, generator=g)
    eps = build_eps(eps_clip, eps_own, a.rho, a.private_clips, step)  # mrv/noise.py (CPU-tested; private 0 == AGREED r1)
    drop = (torch.rand(a.clips, generator=g) < 0.1).repeat_interleave(Fn)
    batch_sha = hashlib.sha1(str(pick).encode() + t.numpy().tobytes() + eps[:, 0, 0, 0].numpy().tobytes() + drop.numpy().tobytes()).hexdigest()[:10]
    PICK_H.update(str(pick).encode() + t.numpy().tobytes() + drop.numpy().tobytes())  # cumulative: data / t / dropout draws only (mode-independent)
    tok, lat, ray = [], [], []
    for sc, i in (starts[k] for k in pick):
        e = E[sc]
        tok.append(e["tokens"][i:i + Fn].float()); lat.append(e["lat"][i:i + Fn].float()); ray.append(RAYS[sc][None].expand(Fn, *RAYS[sc].shape))
    tok = torch.cat(tok).to(dev); x0 = torch.cat(lat).to(dev); ray = torch.cat(ray).to(dev).float()
    tt = t.repeat_interleave(Fn).to(dev); eps = eps.to(dev)
    torch.cuda.synchronize(); t1 = time.time()
    ab = acp[tt].view(-1, 1, 1, 1); xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    for g_ in opt.param_groups:
        g_["lr"] = 1e-4 * min(1, (step + 1) / 500)
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        ctx = adapter(tok, ray, style, drop.to(dev))
    set_mode(TW, "clip", Fn)
    with torch.autocast("cuda", dtype=AMP):
        pred = unet(xt, tt, encoder_hidden_states=ctx).sample
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
    t_data += t1 - t0; t_gpu += t2 - t1; n_acc += 1
    if (step + 1) % 100 == 0 or step == end - 1 or step == start or a.timing:  # timing / test runs log every update
        rec = {"update": step + 1, "e2e_window_s": round((time.time() - T_WIN) / max(1, step + 1 - start - N_WIN), 3), "loss": round(loss.item(), 4), "gn": round(gn.item(), 4),
               "out_norm": round(sum(w.temporal.out.weight.norm().item() ** 2 for w in TW) ** 0.5, 4),
               "age_bias_abs": round(sum(w.temporal.age_bias.abs().sum().item() for w in TW), 4),
               "step_s": round(t_gpu / n_acc, 3), "data_s": round(t_data / n_acc, 3),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "batch_sha": batch_sha, "pick_sha_cum": PICK_H.hexdigest()[:10], "private_clips": a.private_clips, "code_sha": CODE_SHA}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush(); t_data = t_gpu = 0.0; n_acc = 0
        T_WIN, N_WIN = time.time(), step + 1 - start
    if (step + 1) % 500 == 0 or step + 1 == end:
        save(step + 1, ck); probe(step + 1)
    if step + 1 in SAVE:
        save(step + 1, f"{a.out}/ckpt_{step + 1}.pt")
print("CAUSAL_TRAIN_DONE", end)
