"""Live (causal) rollout of the causal video adapter (council/AGREED-causal-r1.md): tick by tick, current cameras only, no
look-ahead. Noise: x_T(tick) = sqrt(rho) z0 + sqrt(1-rho) z_tick, z0 = the signed temporal_rollout.py tensor per (drive, seed),
z_tick keyed drive|seed|tick (rho = 1 -> exactly the signed A noise), DDIM 25, eta 0, CFG 2, style 2, rows [cond, uncond]. The temporal blocks read K / V rings per (layer, DDIM step)
holding the last K ticks of this stream; a hard break clears the rings (restart cold); a camera swap (--swap, lag test) keeps
them. Output = the signed temporal_rollout format (<out>/<scene>_s<seed>.pt) so temporal_metrics.py (764dd3ec95db) scores it.

  python scripts/causal_rollout.py --ckpt runs/vlive/ckpt_2000.pt --split dev --out runs/troll/V2k_dev [--mode off]
"""
import argparse, json, os, zlib
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.causal_video import add_causal_temporal, set_mode, set_step, reset

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True, help="causal checkpoint (or a plain A checkpoint with --mode off)")
p.add_argument("--mode", default="stream", choices=["stream", "off"])
p.add_argument("--split", default="dev", choices=["dev", "decision", "final", "train"])
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--seeds", default="0,1")
p.add_argument("--drive_list", default="")
p.add_argument("--swap", action="store_true", help="lag test: from tick --swap_at on, feed another drive's cameras (history kept)")
p.add_argument("--swap_at", type=int, default=50)
p.add_argument("--salt_from", type=int, default=-1, help="test T5: from this tick on, use a different private noise draw")
p.add_argument("--start_tick", type=int, default=0, help="test T6: start the stream at this absolute tick (earlier ticks skipped)")
p.add_argument("--max_ticks", type=int, default=0)
p.add_argument("--rho", type=float, default=None, help="noise share across ticks; default = the checkpoint's rho (required with --mode off on a plain A checkpoint)")
p.add_argument("--reset_at", type=int, default=-1, help="test: clear the temporal rings at this tick (as after a hard break)")
p.add_argument("--shard", default="0/1")
p.add_argument("--fp32", action="store_true", help="tests: run in fp32")
p.add_argument("--out", required=True)
a = p.parse_args()
dev = "cuda"
AMP = torch.float32 if a.fp32 else (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16)
SD = "sd-legacy/stable-diffusion-v1-5"
os.makedirs(a.out, exist_ok=True)
ck = torch.load(a.ckpt, map_location="cpu"); r = ck["args"]["rank"]
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); res_ = unet.load_state_dict(ck["lora"], strict=False)
assert not res_.unexpected_keys and {n for n, _ in unet.named_parameters() if "lora" in n} == set(ck["lora"]), "LoRA load mismatch"
unet.to(dev, AMP).eval()
TW = []
if "temporal" in ck:
    TW = add_causal_temporal(unet, ck["K"])
    names_ = [n for n, m in unet.named_modules() if m.__class__.__name__ == "WithCausalTemporal"]
    assert len(TW) == 16 and len(ck["temporal"]) == 16, "need 16 temporal blocks"
    if "block_names" in ck and ck["block_names"] != names_:
        raise SystemExit("temporal block names differ from the checkpoint's")
    for w, sd_ in zip(TW, ck["temporal"]):
        w.temporal.load_state_dict(sd_, strict=True)
    for w in TW:
        w.temporal.to(dev, AMP).eval()
elif a.mode == "stream":
    raise SystemExit("--mode stream needs a causal checkpoint")
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); ddim.set_timesteps(25); TS = ddim.timesteps
RAYS = torch.load(a.rays)
drives = json.load(open(a.lists))[a.split]
ALL = list(drives)  # swap partners come from the full split, even with --drive_list
if a.drive_list:
    want_ = set(open(a.drive_list).read().split())
    if not want_ <= set(drives):
        raise SystemExit(f"--drive_list drives not in split {a.split}: {sorted(want_ - set(drives))}")
    drives = [d_ for d_ in drives if d_ in want_]
missing_ = [d_ for d_ in drives if not os.path.exists(f"{a.enc}/{d_}.pt")]
if missing_:
    raise SystemExit(f"missing encodings: {missing_}")
SETTINGS = {"snapshot": a.ckpt, "method": f"causal_{a.mode}", "rho": None, "swap": a.swap, "swap_at": a.swap_at,
            "salt_from": a.salt_from, "start_tick": a.start_tick, "reset_at": a.reset_at, "max_ticks": a.max_ticks}
st = torch.full((1,), 2, dtype=torch.long, device=dev)
RHO = a.rho if a.rho is not None else ck.get("rho")
if RHO is None:
    raise SystemExit("no rho: pass --rho (plain A checkpoint) or use a causal checkpoint that stores it")
if a.rho is not None and "rho" in ck and abs(a.rho - ck["rho"]) > 1e-9:
    raise SystemExit(f"--rho {a.rho} differs from the checkpoint's rho {ck['rho']}")
SETTINGS["rho"] = RHO
set_mode(TW, a.mode)


def denoise(z, c, u):
    for si, t in enumerate(TS):
        set_step(TW, si)
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP, enabled=not a.fp32):
            e = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([c, u])).sample.float()
        ec, eu = e.chunk(2); z = ddim.step(eu + 2.0 * (ec - eu), t, z).prev_sample
    return z


si_, sn_ = map(int, a.shard.split("/"))
for di, sc in enumerate(drives):
    if di % sn_ != si_:
        continue
    E = torch.load(f"{a.enc}/{sc}.pt")
    other = ALL[(ALL.index(sc) + len(ALL) // 2) % len(ALL)]
    if a.swap and not os.path.exists(f"{a.enc}/{other}.pt"):
        raise SystemExit(f"--swap: donor {other} missing")
    E2 = torch.load(f"{a.enc}/{other}.pt") if a.swap else None
    n = len(E["ticks"]) if not a.max_ticks else min(a.max_ticks, len(E["ticks"]))
    for seed in map(int, a.seeds.split(",")):
        fo = f"{a.out}/{sc}_s{seed}.pt"
        if os.path.exists(fo):  # reuse only an artifact made with exactly these settings
            old_ = torch.load(fo)
            bad_ = {k: (old_.get(k), v) for k, v in SETTINGS.items() if old_.get(k) != v}
            if bad_:
                raise SystemExit(f"{fo} exists with different settings {bad_}: use a new --out")
            continue
        g = torch.Generator().manual_seed(zlib.crc32(f"{sc}|{seed}".encode()))  # same tensor as the signed A rollouts
        z0 = torch.randn(1, 4, 32, 64, generator=g).to(dev)
        reset(TW)
        out, restart, oks, hards, srct, fresh = [], [], [], [], [], True
        for i in range(a.start_tick, n):
            if E2 is not None and i >= a.swap_at and i >= len(E2["ticks"]):
                break
            src, j, sc_r = (E2, i, other) if (E2 is not None and i >= a.swap_at) else (E, i, sc)
            oks.append(bool(src["ok"][j])); hards.append(bool(src["hard"][j])); srct.append(int(src["ticks"][j]))
            if bool(src["hard"][j]):
                out.append(torch.zeros(4, 32, 64, dtype=torch.float16)); restart.append(True); reset(TW); fresh = True
                continue
            tok = src["tokens"][j:j + 1].to(dev).float(); ry = RAYS[sc_r][None].to(dev).float()
            with torch.no_grad(), torch.autocast("cuda", dtype=AMP, enabled=not a.fp32):
                c = adapter(tok, ry, st); u = adapter(tok, ry, st, torch.ones(1, dtype=torch.bool, device=dev))
            if i == a.reset_at:
                reset(TW); fresh = True  # forced cold restart (test T6)
            restart.append(fresh); fresh = False
            gn = torch.Generator().manual_seed(zlib.crc32((f"{sc}|{seed}|{i}" + ("|salt" if 0 <= a.salt_from <= i else "")).encode()))  # per-tick private noise
            zn = torch.randn(1, 4, 32, 64, generator=gn).to(dev)
            z = denoise(RHO ** 0.5 * z0 + (1 - RHO) ** 0.5 * zn if RHO < 1 else z0.clone(), c.to(AMP), u.to(AMP))
            out.append(z[0].half().cpu())
        n_out = len(out)
        rec = {"ticks": E["ticks"][a.start_tick:a.start_tick + n_out], "src_ticks": torch.tensor(srct), "lat": torch.stack(out), "ok": torch.tensor(oks),
               "hard": torch.tensor(hards), "restart": torch.tensor(restart), "amp": str(AMP), "snapshot": a.ckpt,
               "snapshot_step": int(ck.get("step", -1)), "s": -1.0, "method": f"causal_{a.mode}", "K": ck.get("K", 0), "rho": RHO, "reset_at": a.reset_at,
               "swap_from": a.swap_at if E2 is not None else -1, **SETTINGS, "base_args": {k: ck["args"].get(k) for k in ("arm", "rank")},
               "code_sha": ck.get("code_sha"), "other": other if E2 is not None else "", "batch": 1}
        torch.save(rec, fo + ".tmp"); os.replace(fo + ".tmp", fo)
        print(sc, "seed", seed, n_out, "ticks", flush=True)
for di, sc in enumerate(drives):  # verify the full expected output set of this shard
    if di % sn_ == si_:
        for seed in map(int, a.seeds.split(",")):
            assert os.path.exists(f"{a.out}/{sc}_s{seed}.pt"), f"missing output {sc}_s{seed}"
print("CAUSAL_ROLLOUT_DONE", a.mode, a.split)
