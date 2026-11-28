"""Clean-memory live video adapter "vclean" on the FROZEN A@20k renderer (council/AGREED-causal-r1.md, Amendment A4).

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
Logged every 100 updates: loss, grad norm, out-norm, age-bias size, step_s, data_s, mem_s, peak GB, batch_sha, code_sha.

  python scripts/train_clean.py --init runs/longA/ckpt_20000.pt --out runs/vclean --steps 16000 [--rho_gate 1.0] [--resume]
"""
import argparse, hashlib, json, os, time
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.clean_video import add_clean_temporal, set_mode, set_memory, collect_inputs
from mrv.noise import build_eps

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
p.add_argument("--init_ref", default="runs/vpriv/args.json", help="vpriv's args.json: its recorded init_sha / seed must equal ours (A4: vpriv's recorded init)")
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
os.makedirs(a.out, exist_ok=True)
dev, AMP = "cuda", torch.bfloat16
SD = "sd-legacy/stable-diffusion-v1-5"
CODE_SHA = hashlib.sha1(b"".join(open(f, "rb").read() for f in [__file__, "mrv/clean_video.py", "mrv/sd.py", "mrv/noise.py"])).hexdigest()[:12]
_dd = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); _dd.set_timesteps(25)
T_STAR = int(_dd.timesteps[-1])  # the actual last timestep of the rollout's DDIM-25 schedule (expected 1)
if T_STAR > 81:
    raise SystemExit(f"A4: t* = {T_STAR} > 81 -> back to council")
print(f"t* = {T_STAR} (DDIM-25 last timestep)", flush=True)
TRAIN = json.load(open(a.lists))["train"]
assert not set(TRAIN) & set(json.load(open(a.lists))["dev"]), "train / dev overlap"
RAYS = torch.load(a.rays)
Fn = a.frames

# ---------- clips: Fn consecutive usable ticks, one sequence, no hard break (as vpriv) ----------
E, starts = {}, []
for sc in sorted(TRAIN):
    e = torch.load(f"{a.enc}/{sc}.pt"); E[sc] = e
    ok, seq, ticks, hard = e["ok"].tolist(), e["seq"].tolist(), e["ticks"].tolist(), e["hard"].tolist()
    for i in range(len(ticks) - Fn + 1):
        if all(ok[i + k] and not hard[i + k] and seq[i + k] == seq[i] and ticks[i + k] == ticks[i] + k for k in range(Fn)):
            starts.append((sc, i))
COVER = {sc: sum(1 for s_, _ in starts if s_ == sc) for sc in sorted(TRAIN)}
print(f"clips: {len(starts)} start positions of {Fn} ticks from {len({s for s, _ in starts})}/{len(E)} drives", flush=True)

# ---------- model: frozen A + clean temporal blocks ----------
ck0 = torch.load(a.init, map_location="cpu"); r = ck0["args"]["rank"]
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
INIT_MATCH = None
if os.path.exists(a.init_ref):
    ref_ = json.load(open(a.init_ref))
    INIT_MATCH = ref_.get("init_sha") == INIT_SHA and ref_.get("seed") == a.seed and ref_.get("block_names") == BLOCK_NAMES
    print(f"init_sha {INIT_SHA} vs {a.init_ref} {ref_.get('init_sha')} (seed {ref_.get('seed')}): {'MATCH' if INIT_MATCH else 'DIFFERENT'}", flush=True)
if not a.timing and not INIT_MATCH:
    raise SystemExit(f"A4: vclean must start from vpriv's recorded init (init_ref {a.init_ref}: match={INIT_MATCH})")
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
    for key_, want_ in (("seed", a.seed), ("lists", a.lists), ("enc", a.enc), ("frames", a.frames), ("clips", a.clips), ("rho_gate", a.rho_gate)):
        if rec_.get(key_) != want_:
            raise SystemExit(f"incompatible resume: recipe {key_} {rec_.get(key_)} != {want_}")
    for key_, want_ in (("K", a.K), ("init", a.init), ("rho", a.rho_gate), ("t_star", T_STAR), ("base_sha", BASE_SHA), ("init_sha", INIT_SHA),
                        ("block_names", BLOCK_NAMES), ("code_sha", CODE_SHA), ("memory", "tstar_final_call")):
        if s_.get(key_) != want_:
            raise SystemExit(f"incompatible resume: {key_} {s_.get(key_)} != {want_}")
    if len(s_["temporal"]) != len(TW):
        raise SystemExit("incompatible resume: block count")
    for w, sd_ in zip(TW, s_["temporal"]):
        w.temporal.load_state_dict(sd_, strict=True)
    opt.load_state_dict(s_["opt"]); start = s_["step"]; print("resumed at", start, flush=True)
json.dump({**vars(a), "arm": "vclean", "code_sha": CODE_SHA, "init_sha": INIT_SHA, "init_match": INIT_MATCH, "base_sha": BASE_SHA, "t_star": T_STAR,
           "block_names": BLOCK_NAMES, "clip_starts": len(starts), "drives_with_clips": sum(1 for v in COVER.values() if v),
           "drives_without_clips": [k for k, v in COVER.items() if not v], "clips_per_drive": COVER}, open(f"{a.out}/args.json", "w"), indent=1)
TRAIN_NOISE = {"private_clips": PRIVATE_CLIPS, "rho": TRAIN_RHO, "t_rule": f"U{{0..999}} minus t*={T_STAR} (t* draws redrawn, separate generator)",
               "memory": f"frozen A, deltas off, key latents noised to t*={T_STAR} with private per-tick draws, same dropout"}


def save(step, path):
    d = {"temporal": [w.temporal.state_dict() for w in TW], "opt": opt.state_dict(), "step": step, "init": a.init, "K": a.K, "frames": Fn,
         "arm": "vclean", "memory": "tstar_final_call", "t_star": T_STAR, "rho": a.rho_gate, "train_noise": TRAIN_NOISE,
         "noise_policy": "one t per clip (t* excluded); fully private eps per tick; memory = t* features of the key latents",
         "code_sha": CODE_SHA, "base_sha": BASE_SHA, "init_sha": INIT_SHA, "block_names": BLOCK_NAMES, "timing": a.timing,
         "args": {**ck0["args"], "clean": vars(a)}, "adapter": ck0["adapter"], "lora": ck0["lora"]}
    torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)


def memory_of(x0, eps_m, ctx):
    """no-grad frozen-A forward at t* (temporal deltas off): per-block input features of every row, detached"""
    ab = acp[T_STAR]; xm = ab.sqrt() * x0 + (1 - ab).sqrt() * eps_m
    tm = torch.full((x0.shape[0],), T_STAR, device=dev)
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
for step in range(start, end):
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
    tok, lat, ray = [], [], []
    for sc, i in (starts[k] for k in pick):
        e = E[sc]
        tok.append(e["tokens"][i:i + Fn].float()); lat.append(e["lat"][i:i + Fn].float()); ray.append(RAYS[sc][None].expand(Fn, *RAYS[sc].shape))
    tok = torch.cat(tok).to(dev); x0 = torch.cat(lat).to(dev); ray = torch.cat(ray).to(dev).float()
    tt = t.repeat_interleave(Fn).to(dev); eps = eps.to(dev); eps_mem = eps_mem.to(dev)
    torch.cuda.synchronize(); t1 = time.time()
    ab = acp[tt].view(-1, 1, 1, 1); xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    for g_ in opt.param_groups:
        g_["lr"] = 1e-4 * min(1, (step + 1) / 500)
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        ctx = adapter(tok, ray, style, drop.to(dev))  # the same dropout context feeds the memory pass and the noisy pass
    mem = memory_of(x0, eps_mem, ctx)
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
               "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "batch_sha": batch_sha, "pick_sha_cum": PICK_H.hexdigest()[:10],
               "private_clips": PRIVATE_CLIPS, "t_star": T_STAR, "code_sha": CODE_SHA}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush(); t_data = t_gpu = t_mem = 0.0; n_acc = 0
        T_WIN, N_WIN = time.time(), step + 1 - start
    if (step + 1) % 500 == 0 or step + 1 == end:
        save(step + 1, ck); probe(step + 1)
    if step + 1 in SAVE:
        save(step + 1, f"{a.out}/ckpt_{step + 1}.pt")
print("CLEAN_TRAIN_DONE", end)
