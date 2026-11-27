"""Causal (live) video adapter on the FROZEN A@20k renderer (council/AGREED-causal-r1.md).

Only the 16 new causal temporal blocks (mrv/causal_video.py) train; the SD 1.5 UNet, A's LoRA and A's CameraAdapter are frozen
(asserted). Each update: 2 clips of 16 consecutive usable ticks (same sequence, no hard break) from the 144 train drives
(data/enc_tpv10); ONE diffusion timestep per clip (uniform 0..999) shared by its ticks; ONE Gaussian noise tensor per clip
broadcast over its ticks (matches the signed A rollouts, which reuse one noise tensor per drive and seed); epsilon MSE in fp32
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
p.add_argument("--save_at", default="1000,2000,4000,6000,8000,10000,12000")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--resume", action="store_true")
p.add_argument("--max_steps_this_job", type=int, default=0, help="stop after this many updates (timing / staged runs)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev, AMP = "cuda", torch.bfloat16
SD = "sd-legacy/stable-diffusion-v1-5"
CODE_SHA = hashlib.sha1(b"".join(open(f, "rb").read() for f in [__file__, "mrv/causal_video.py", "mrv/sd.py"])).hexdigest()[:12]
TRAIN = json.load(open(a.lists))["train"]
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
print(f"clips: {len(starts)} start positions of {Fn} ticks from {len({s for s, _ in starts})}/{len(E)} drives", flush=True)

# ---------- model: frozen A + causal temporal blocks ----------
ck0 = torch.load(a.init, map_location="cpu"); r = ck0["args"]["rank"]
assert "hist" not in ck0, "init must be a no-memory renderer (A)"
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS))
res = unet.load_state_dict(ck0["lora"], strict=False)
assert not res.unexpected_keys, f"unexpected LoRA keys: {res.unexpected_keys[:5]}"
lora_names = {n for n, _ in unet.named_parameters() if "lora" in n}
assert lora_names == set(ck0["lora"]) and len(lora_names) > 0, "every LoRA tensor must come from the init checkpoint"
unet.requires_grad_(False); unet.to(dev, AMP)
TW = add_causal_temporal(unet, a.K)
assert len(TW) == 16, len(TW)
for w in TW:
    w.temporal.to(dev).float().requires_grad_(True)
adapter = CameraAdapter(n_styles=ck0["args"].get("n_styles", 3)); adapter.load_state_dict(ck0["adapter"], strict=True)
adapter.requires_grad_(False); adapter.to(dev).eval()
unet.eval()  # frozen renderer in eval mode (temporal blocks have no dropout)
params = [q for w in TW for q in w.temporal.parameters()]
trainable = [q for q in list(unet.parameters()) + list(adapter.parameters()) if q.requires_grad]
assert {id(q) for q in trainable} == {id(q) for q in params}, "only the temporal blocks may train"
print(f"temporal params: {sum(q.numel() for q in params)}; frozen base + A LoRA ({len(lora_names)} tensors) + A adapter", flush=True)
frozen_sha = hashlib.sha1(b"".join(v.detach().float().cpu().numpy().tobytes() for k, v in sorted(ck0["lora"].items())[:8])).hexdigest()[:12]
opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
acp = DDPMScheduler.from_pretrained(SD, subfolder="scheduler").alphas_cumprod.to(dev)
start, ck = 0, f"{a.out}/ckpt.pt"
if os.path.exists(ck):
    if not a.resume:
        raise SystemExit(f"{ck} exists: pass --resume or use a fresh --out")
    s_ = torch.load(ck, map_location="cpu")
    if s_.get("K") != a.K or s_.get("init") != a.init or s_.get("rho") != a.rho:
        raise SystemExit(f"incompatible resume: K {s_.get('K')} init {s_.get('init')}")
    for w, sd_ in zip(TW, s_["temporal"]):
        w.temporal.load_state_dict(sd_)
    opt.load_state_dict(s_["opt"]); start = s_["step"]; print("resumed at", start, flush=True)
json.dump({**vars(a), "code_sha": CODE_SHA, "frozen_lora_sha": frozen_sha, "clip_starts": len(starts)}, open(f"{a.out}/args.json", "w"), indent=1)


def save(step, path):
    d = {"temporal": [w.temporal.state_dict() for w in TW], "opt": opt.state_dict(), "step": step, "init": a.init, "K": a.K,
         "frames": Fn, "noise_policy": "one t per clip; eps = sqrt(rho) eps_clip + sqrt(1-rho) eps_tick", "rho": a.rho, "code_sha": CODE_SHA,
         "args": {**ck0["args"], "causal": vars(a)}, "adapter": ck0["adapter"], "lora": ck0["lora"]}
    torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)


# ---------- copy probe (AGREED-causal-r1): 8 fixed dev clips of K+1 ticks, t = 900 / 700 / 300 ----------
DEVL = json.load(open(a.lists))["dev"]
PROBE = []
gp = torch.Generator().manual_seed(4242)
for sc in DEVL:
    e = torch.load(f"{a.enc}/{sc}.pt")
    ok, seq, ticks, hard = e["ok"].tolist(), e["seq"].tolist(), e["ticks"].tolist(), e["hard"].tolist()
    L = a.K + 1
    st_ = [i for i in range(len(ticks) - L + 1) if all(ok[i + k] and not hard[i + k] and seq[i + k] == seq[i] and ticks[i + k] == ticks[i] + k for k in range(L))]
    for k_ in torch.randperm(len(st_), generator=gp)[:2].tolist():
        i = st_[k_]; PROBE.append((sc, e["tokens"][i:i + L].float(), e["lat"][i:i + L].float()))
PROBE = PROBE[:8]
P_EPS = [torch.randn(1, 4, 32, 64, generator=gp) for _ in PROBE]  # one draw per probe clip (shared over its ticks, rho-mixed below)
P_OWN = [torch.randn(a.K + 1, 4, 32, 64, generator=gp) for _ in PROBE]


def probe(step):
    """loss on the last tick of each probe clip: blocks on / off / past from another drive / current cameras from another drive"""
    L = a.K + 1; res = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        for tv in (900, 700, 300):
            acc = {"on": 0.0, "off": 0.0, "past_other": 0.0, "cams_other": 0.0}
            for pi, (sc, tk, lt) in enumerate(PROBE):
                osc, otk, olt = PROBE[(pi + 1) % len(PROBE)]
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
            res[tv] = {k: round(v, 5) for k, v in acc.items()}
    flag = any(res[tv]["cams_other"] - res[tv]["on"] < res[tv]["past_other"] - res[tv]["on"] for tv in (900, 700))
    rec = {"step": step, "probe": res, "copy_flag": flag}
    print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    set_mode(TW, "clip", Fn)


SAVE = set(int(x) for x in a.save_at.split(","))
log = open(f"{a.out}/log.jsonl", "a")
style = torch.full((a.clips * Fn,), 2, dtype=torch.long, device=dev)
w_loss = torch.zeros(a.clips, Fn, 1, 1, 1, device=dev); w_loss[:, 1:] = 1.0  # loss on ticks 1..Fn-1
end = a.steps if not a.max_steps_this_job else min(a.steps, start + a.max_steps_this_job)
t_data = t_gpu = 0.0; n_acc = 0
for step in range(start, end):
    t0 = time.time()
    g = torch.Generator().manual_seed(a.seed * 100003 + step)
    pick = torch.randint(0, len(starts), (a.clips,), generator=g).tolist()
    t = torch.randint(0, 1000, (a.clips,), generator=g)
    eps_clip = torch.randn(a.clips, 1, 4, 32, 64, generator=g); eps_own = torch.randn(a.clips, Fn, 4, 32, 64, generator=g)
    eps = (a.rho ** 0.5 * eps_clip + (1 - a.rho) ** 0.5 * eps_own).reshape(a.clips * Fn, 4, 32, 64)
    drop = (torch.rand(a.clips, generator=g) < 0.1).repeat_interleave(Fn)
    tok, lat, ray = [], [], []
    for sc, i in (starts[k] for k in pick):
        e = E[sc]
        tok.append(e["tokens"][i:i + Fn].float()); lat.append(e["lat"][i:i + Fn].float()); ray.append(RAYS[sc][None].expand(Fn, *RAYS[sc].shape))
    tok = torch.cat(tok).to(dev); x0 = torch.cat(lat).to(dev); ray = torch.cat(ray).to(dev).float()
    tt = t.repeat_interleave(Fn).to(dev); eps = eps.to(dev)
    batch_sha = hashlib.sha1(str(pick).encode() + t.numpy().tobytes() + eps[0, 0, 0].cpu().numpy().tobytes()).hexdigest()[:10]
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
    opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if step == start:  # guard: the frozen renderer must be bit-identical after an optimizer step
        cur = {n: q for n, q in unet.named_parameters() if "lora" in n}
        assert all(torch.equal(cur[n].detach().cpu(), ck0["lora"][n].to(cur[n].dtype)) for n in ck0["lora"]), "LoRA changed!"
        assert all(torch.equal(q.detach().cpu(), ck0["adapter"][n]) for n, q in adapter.state_dict().items()), "adapter changed!"
        print("frozen check after first update: LoRA + adapter bit-identical to", a.init, flush=True)
    torch.cuda.synchronize(); t2 = time.time()
    t_data += t1 - t0; t_gpu += t2 - t1; n_acc += 1
    if step % 100 == 0 or step == end - 1:
        rec = {"step": step, "loss": round(loss.item(), 4), "gn": round(gn.item(), 4),
               "out_norm": round(sum(w.temporal.out.weight.norm().item() ** 2 for w in TW) ** 0.5, 4),
               "age_bias_abs": round(sum(w.temporal.age_bias.abs().sum().item() for w in TW), 4),
               "step_s": round(t_gpu / n_acc, 3), "data_s": round(t_data / n_acc, 3),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "batch_sha": batch_sha, "code_sha": CODE_SHA}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush(); t_data = t_gpu = 0.0; n_acc = 0
    if (step + 1) % 500 == 0 or step + 1 == end:
        save(step + 1, ck); probe(step + 1)
    if step + 1 in SAVE:
        save(step + 1, f"{a.out}/ckpt_{step + 1}.pt")
print("CAUSAL_TRAIN_DONE", end)
