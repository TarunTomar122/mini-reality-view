"""Acceptance tests for the causal video adapter (council/AGREED-causal-r1.md), run BEFORE training. fp32, one GPU.

  T1 parity: a fresh (zero-init) adapter in stream and clip mode, and mode off, equal the unwrapped A UNet EXACTLY.
  T2 stream == clip: with random NONZERO temporal weights, tick-by-tick stream mode equals the causal clip forward
     (max |diff| < 1e-4) over F ticks (> K, so eviction is exercised); after reset(), a stream equals a fresh clip.
  T3 prefix invariance: a 12-tick causal rollout (DDIM 25, CFG 2) with the cameras of ticks >= 8 swapped for another drive's
     gives IDENTICAL latents for ticks 0..7, and different latents later (the camera path is connected).
  T4 frozen: one optimizer step on the temporal params leaves every LoRA / adapter tensor bit-identical and changes the
     temporal output weight.
Prints PASS / FAIL per test and exits 1 on any FAIL.

  python scripts/causal_tests.py --init runs/longA/ckpt_20000.pt [--K 4] [--F 16]
"""
import argparse, copy, json, sys, zlib
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.causal_video import add_causal_temporal, set_mode, set_step, reset

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--K", type=int, default=4)
p.add_argument("--F", type=int, default=16)
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--rays", default="data/calib_rays.pt")
a = p.parse_args()
dev = "cuda"; SD = "sd-legacy/stable-diffusion-v1-5"
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
ck = torch.load(a.init, map_location="cpu"); r = ck["args"]["rank"]
FAIL = []


def load_unet():
    u = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
    u.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); u.load_state_dict(ck["lora"], strict=False)
    return u.to(dev).eval()


def report(name, ok, msg):
    print(("PASS " if ok else "FAIL ") + name + ": " + msg, flush=True)
    if not ok:
        FAIL.append(name)


adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
RAYS = torch.load(a.rays)
dev_drives = json.load(open("data/temporal_lists.json"))["dev"]
sc, sc2 = dev_drives[0], dev_drives[1]
E = torch.load(f"{a.enc}/{sc}.pt"); E2 = torch.load(f"{a.enc}/{sc2}.pt")
st = lambda n: torch.full((n,), 2, dtype=torch.long, device=dev)
Fn = a.F
with torch.no_grad():
    tok = E["tokens"][:Fn].to(dev).float(); ray = RAYS[sc][None].expand(Fn, *RAYS[sc].shape).to(dev).float()
    ctx = adapter(tok, ray, st(Fn))
g = torch.Generator().manual_seed(0)
xt = torch.randn(Fn, 4, 32, 64, generator=g).to(dev); t = torch.full((Fn,), 500, device=dev)

# ---------- T1 parity ----------
base = load_unet()
with torch.no_grad():
    ref = base(xt, t, encoder_hidden_states=ctx).sample  # batch F (compared with clip / off)
    ref1 = torch.cat([base(xt[i:i + 1], t[i:i + 1], encoder_hidden_states=ctx[i:i + 1]).sample for i in range(Fn)])  # batch 1 (vs stream)
wrapped = load_unet(); TW = add_causal_temporal(wrapped, a.K); [w.temporal.to(dev) for w in TW]
with torch.no_grad():
    set_mode(TW, "off"); o_off = wrapped(xt, t, encoder_hidden_states=ctx).sample
    set_mode(TW, "clip", Fn); o_clip = wrapped(xt, t, encoder_hidden_states=ctx).sample
    set_mode(TW, "stream"); reset(TW); set_step(TW, 0)
    o_str = torch.cat([wrapped(xt[i:i + 1], t[i:i + 1], encoder_hidden_states=ctx[i:i + 1]).sample for i in range(Fn)])
d1 = max((o_off - ref).abs().max().item(), (o_clip - ref).abs().max().item(), (o_str - ref1).abs().max().item())
report("T1 zero-init / off == A", d1 == 0.0, f"max |diff| {d1:.3g} (must be exactly 0)")

# ---------- T2 stream == clip with nonzero weights ----------
g2 = torch.Generator().manual_seed(1)
with torch.no_grad():
    for w in TW:
        for n_, q in w.temporal.named_parameters():
            if n_.startswith("norm") or n_.startswith("ln"):
                continue
            q.add_(torch.randn(q.shape, generator=g2).to(dev) * 0.05)
    set_mode(TW, "clip", Fn); o_clip = wrapped(xt, t, encoder_hidden_states=ctx).sample
    set_mode(TW, "stream"); reset(TW); set_step(TW, 0)
    o_str = torch.cat([wrapped(xt[i:i + 1], t[i:i + 1], encoder_hidden_states=ctx[i:i + 1]).sample for i in range(Fn)])
    h = Fn // 2  # reset half-way: stream after reset == clip of the second half alone
    set_mode(TW, "clip", Fn - h); o_clip2 = wrapped(xt[h:], t[h:], encoder_hidden_states=ctx[h:]).sample
    set_mode(TW, "stream"); reset(TW); set_step(TW, 0)
    o_str2 = torch.cat([wrapped(xt[i:i + 1], t[i:i + 1], encoder_hidden_states=ctx[i:i + 1]).sample for i in range(h, Fn)])
d2 = (o_clip - o_str).abs().max().item(); d2b = (o_clip2 - o_str2).abs().max().item()
dA = (o_clip[a.K + 1:] - ref[a.K + 1:]).abs().max().item()
report("T2 stream == causal clip", d2 < 1e-4 and d2b < 1e-4 and dA > 1e-3,
       f"max |diff| {d2:.2e} (F={Fn}, K={a.K}), after reset {d2b:.2e}, differs from A by {dA:.2e} (weights are live)")

# ---------- T3 prefix invariance in a real causal rollout ----------
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); ddim.set_timesteps(25); TS = ddim.timesteps
z0 = torch.randn(1, 4, 32, 64, generator=torch.Generator().manual_seed(zlib.crc32(f"{sc}|0".encode()))).to(dev)


def rollout(swap_from):
    reset(TW); set_mode(TW, "stream"); outs = []
    for i in range(12):
        src, sr = (E2, sc2) if i >= swap_from else (E, sc)
        with torch.no_grad():
            tk = src["tokens"][i:i + 1].to(dev).float(); ry = RAYS[sr][None].to(dev).float()
            c = adapter(tk, ry, st(1)); u = adapter(tk, ry, st(1), torch.ones(1, dtype=torch.bool, device=dev))
            z = z0.clone()
            for si, tt in enumerate(TS):
                set_step(TW, si)
                e = wrapped(torch.cat([z, z]), tt, encoder_hidden_states=torch.cat([c, u])).sample
                ec, eu = e.chunk(2); z = ddim.step(eu + 2.0 * (ec - eu), tt, z).prev_sample
        outs.append(z)
    return torch.cat(outs)


ra, rb = rollout(99), rollout(8)
dpre = (ra[:8] - rb[:8]).abs().max().item(); dpost = (ra[8:] - rb[8:]).abs().max().item()
report("T3 future cameras never change the past", dpre == 0.0 and dpost > 1e-3, f"prefix max |diff| {dpre:.3g} (must be 0), later {dpost:.3g} (must be > 0)")

# ---------- T4 frozen parts stay frozen ----------
tr = load_unet(); tr.requires_grad_(False); T4 = add_causal_temporal(tr, a.K)
for w in T4:
    w.temporal.to(dev).requires_grad_(True)
params = [q for w in T4 for q in w.temporal.parameters()]
trainable = [q for q in tr.parameters() if q.requires_grad]
before = {n: q.detach().clone() for n, q in tr.named_parameters() if "lora" in n}
ad_before = {n: q.detach().clone() for n, q in adapter.named_parameters()}
w_before = T4[0].temporal.out.weight.detach().clone()
opt = torch.optim.AdamW(params, lr=1e-3)
set_mode(T4, "clip", Fn); tr.train()
loss = (tr(xt, t, encoder_hidden_states=ctx).sample ** 2).mean(); loss.backward(); opt.step()
same = all(torch.equal(before[n], q) for n, q in tr.named_parameters() if "lora" in n) and \
       all(torch.equal(ad_before[n], q) for n, q in adapter.named_parameters())
moved = not torch.equal(w_before, T4[0].temporal.out.weight)
report("T4 only temporal params train", same and moved and {id(q) for q in trainable} == {id(q) for q in params},
       f"LoRA+adapter unchanged {same}, temporal out moved {moved}, trainable == temporal {len(trainable)} == {len(params)}")
print("CAUSAL_TESTS", "FAIL " + ",".join(FAIL) if FAIL else "ALL PASS", flush=True)
sys.exit(1 if FAIL else 0)
