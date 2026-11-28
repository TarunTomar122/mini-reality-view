"""Live rollout of the clean-memory adapter "vclean" (council/AGREED-causal-r1.md, Amendment A4): tick by tick, current
cameras only, no look-ahead, no extra UNet call vs A. Noise exactly as scripts/causal_rollout.py: x_T(tick) = sqrt(rho) z0 +
sqrt(1-rho) z_tick, z0 = the signed temporal_rollout.py tensor per (drive, seed), z_tick keyed drive|seed|tick (rho = 1 ->
exactly the signed A noise); DDIM 25, eta 0, CFG 2, style 2, rows [cond, uncond]. rho defaults to the checkpoint's (A4 gate
rho, 1.0 unless T2 failed). Memory (mrv/clean_video.py): ONE ring per block of the last K ticks' t* features (cond and uncond
rows separate); DDIM calls 1..24 read it; call 25 (t* = the last timestep) runs with temporal deltas OFF and stages its block
inputs, published to the ring after the tick (evicting beyond K). A hard break / --reset_at clears the ring; a camera swap
(--swap, lag test) keeps it. --mode off == frozen A (with --rho 1.0 it reproduces causal_rollout.py --mode off at rho 1.0
exactly: same noise, same calls). Output = the signed temporal_rollout format (<out>/<scene>_s<seed>.pt).

  python scripts/clean_rollout.py --ckpt runs/vclean/ckpt_2000.pt --split dev --out runs/troll/C2k_dev [--mode off]
"""
import argparse, hashlib, io, json, os, zlib
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS
from mrv.clean_video import add_clean_temporal, set_mode, set_stage, publish, reset

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True, help="vclean checkpoint (or a plain A / any checkpoint with --mode off)")
p.add_argument("--mode", default="stream", choices=["stream", "off"])
p.add_argument("--split", default="dev", choices=["dev", "decision", "final", "train"])
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--seeds", default="0,1")
p.add_argument("--drive_list", default="")
p.add_argument("--swap", action="store_true", help="lag test: from tick --swap_at on, feed another drive's cameras (memory kept)")
p.add_argument("--swap_at", type=int, default=50)
p.add_argument("--salt_from", type=int, default=-1, help="test T5: from this tick on, use a different private noise draw")
p.add_argument("--start_tick", type=int, default=0, help="test T6: start the stream at this absolute tick (earlier ticks skipped)")
p.add_argument("--max_ticks", type=int, default=0)
p.add_argument("--rho", type=float, default=None, help="noise share across ticks; default = the checkpoint's rho (required with --mode off on a plain A checkpoint)")
p.add_argument("--reset_at", type=int, default=-1, help="test: clear the memory ring at this tick (as after a hard break)")
p.add_argument("--shard", default="0/1")
p.add_argument("--fp32", action="store_true", help="tests: run in fp32")
p.add_argument("--out", required=True)
a = p.parse_args()
dev = "cuda"
AMP = torch.float32 if a.fp32 else (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16)
SD = "sd-legacy/stable-diffusion-v1-5"
os.makedirs(a.out, exist_ok=True)
_raw = open(a.ckpt, "rb").read()  # one read: the digest describes exactly the bytes that are loaded
CKPT_SHA = hashlib.sha256(_raw).hexdigest()[:16]
ROLL_CODE = hashlib.sha256(b"".join(open(f, "rb").read() for f in [__file__, "mrv/clean_video.py", "mrv/sd.py"])).hexdigest()[:12]
ck = torch.load(io.BytesIO(_raw), map_location="cpu"); r = ck["args"]["rank"]; del _raw
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); res_ = unet.load_state_dict(ck["lora"], strict=False)
assert not res_.unexpected_keys and {n for n, _ in unet.named_parameters() if "lora" in n} == set(ck["lora"]), "LoRA load mismatch"
unet.to(dev, AMP).eval()
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); ddim.set_timesteps(25); TS = ddim.timesteps
TW = []
if "temporal" in ck:
    if a.mode == "stream" and ck.get("memory") != "tstar_final_call":
        raise SystemExit("--mode stream needs a vclean checkpoint (memory = t* final-call features); same-step causal blocks go to causal_rollout.py")
    if "t_star" in ck and int(TS[-1]) != ck["t_star"]:
        raise SystemExit(f"DDIM-25 last timestep {int(TS[-1])} != the checkpoint's t* {ck['t_star']}")
    TW = add_clean_temporal(unet, ck["K"])
    names_ = [n for n, m in unet.named_modules() if m.__class__.__name__ == "WithCleanTemporal"]
    assert len(TW) == 16 and len(ck["temporal"]) == 16, "need 16 temporal blocks"
    if ck.get("block_names") != names_:
        raise SystemExit("temporal block names missing from / different to the checkpoint's")
    for w, sd_ in zip(TW, ck["temporal"]):
        w.temporal.load_state_dict(sd_, strict=True)
    for w in TW:
        w.temporal.to(dev, AMP).eval()
elif a.mode == "stream":
    raise SystemExit("--mode stream needs a vclean checkpoint")
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
if "base_sha" in ck:
    _h = hashlib.sha256()
    for k_ in sorted(ck["lora"]):
        _h.update(k_.encode()); _h.update(ck["lora"][k_].float().numpy().tobytes())
    for k_ in sorted(ck["adapter"]):
        _h.update(k_.encode()); _h.update(ck["adapter"][k_].float().numpy().tobytes())
    if _h.hexdigest()[:16] != ck["base_sha"]:
        raise SystemExit("frozen base in the checkpoint does not match its recorded base_sha")
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
SETTINGS = {"snapshot": a.ckpt, "ckpt_sha": CKPT_SHA, "roll_code": ROLL_CODE, "amp": str(AMP), "method": f"clean_{a.mode}", "rho": None,
            "swap": a.swap, "swap_at": a.swap_at, "salt_from": a.salt_from, "start_tick": a.start_tick, "reset_at": a.reset_at,
            "max_ticks": a.max_ticks}
st = torch.full((1,), 2, dtype=torch.long, device=dev)
RHO = a.rho if a.rho is not None else ck.get("rho")
if RHO is None:
    raise SystemExit("no rho: pass --rho (plain A checkpoint) or use a vclean checkpoint that stores it")
if a.rho is not None and "rho" in ck and abs(a.rho - ck["rho"]) > 1e-9:
    raise SystemExit(f"--rho {a.rho} differs from the checkpoint's rho {ck['rho']}")
SETTINGS["rho"] = RHO
set_mode(TW, a.mode)
STREAM = a.mode == "stream"


def denoise(z, c, u):
    for si, t in enumerate(TS):
        set_stage(TW, STREAM and si == len(TS) - 1)  # calls 1..24 read the ring; call 25 (t*): deltas off, stage memory
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP, enabled=not a.fp32):
            e = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([c, u])).sample.float()
        ec, eu = e.chunk(2); z = ddim.step(eu + 2.0 * (ec - eu), t, z).prev_sample
    if STREAM:
        publish(TW)  # once per tick, after the tick
    return z


si_, sn_ = map(int, a.shard.split("/"))
RY, U_NULL = {}, [None]
for di, sc in enumerate(drives):
    if di % sn_ != si_:
        continue
    _eb = open(f"{a.enc}/{sc}.pt", "rb").read(); ENC_SHA = hashlib.sha256(_eb).hexdigest()[:16]
    E = torch.load(io.BytesIO(_eb)); del _eb
    RAY_SHA = hashlib.sha256(RAYS[sc].float().numpy().tobytes()).hexdigest()[:16]
    other = ALL[(ALL.index(sc) + len(ALL) // 2) % len(ALL)]
    if a.swap and not os.path.exists(f"{a.enc}/{other}.pt"):
        raise SystemExit(f"--swap: donor {other} missing")
    E2 = torch.load(f"{a.enc}/{other}.pt") if a.swap else None
    n = len(E["ticks"]) if not a.max_ticks else min(a.max_ticks, len(E["ticks"]))
    for seed in map(int, a.seeds.split(",")):
        fo = f"{a.out}/{sc}_s{seed}.pt"
        if os.path.exists(fo):  # reuse only an artifact made with exactly these settings
            old_ = torch.load(fo)
            want_ = {**SETTINGS, "enc_sha": ENC_SHA, "ray_sha": RAY_SHA, "other": other if E2 is not None else ""}
            bad_ = {k: (old_.get(k), v) for k, v in want_.items() if old_.get(k) != v}
            if bad_:
                raise SystemExit(f"{fo} exists with different settings {bad_}: use a new --out")
            continue
        g = torch.Generator().manual_seed(zlib.crc32(f"{sc}|{seed}".encode()))  # same tensor as the signed A rollouts
        z0 = torch.randn(1, 4, 32, 64, generator=g).to(dev)
        reset(TW)
        out, restart, oks, hards, srct, fresh = [], [], [], [], [], True
        src_scene, noise_key = [], []  # per-tick provenance: whose cameras, which private draw
        for i in range(a.start_tick, n):
            if E2 is not None and i >= a.swap_at and i >= len(E2["ticks"]):
                break
            src, j, sc_r = (E2, i, other) if (E2 is not None and i >= a.swap_at) else (E, i, sc)
            oks.append(bool(src["ok"][j])); hards.append(bool(src["hard"][j])); srct.append(int(src["ticks"][j])); src_scene.append(sc_r)
            if bool(src["hard"][j]):
                out.append(torch.zeros(4, 32, 64, dtype=torch.float16)); restart.append(True); reset(TW); fresh = True
                continue
            tok = src["tokens"][j:j + 1].to(dev).float()
            if sc_r not in RY:
                RY[sc_r] = RAYS[sc_r][None].to(dev).float()
            ry = RY[sc_r]
            with torch.no_grad(), torch.autocast("cuda", dtype=AMP, enabled=not a.fp32):
                c = adapter(tok, ry, st)
                if U_NULL[0] is None:  # with dropout the adapter returns its learned null for every input: compute once
                    U_NULL[0] = adapter(tok, ry, st, torch.ones(1, dtype=torch.bool, device=dev))
                u = U_NULL[0]
            if i == a.reset_at:
                reset(TW); fresh = True  # forced cold restart (test T6)
            restart.append(fresh); fresh = False
            tid = int(E["ticks"][i])  # receiver tick id (array positions for --start_tick/--swap_at/--reset_at/--salt_from)
            nk_ = f"{sc}|{seed}|{tid}" + ("|salt" if 0 <= a.salt_from <= i else ""); noise_key.append(nk_)
            gn = torch.Generator().manual_seed(zlib.crc32(nk_.encode()))  # per-tick private noise
            zn = torch.randn(1, 4, 32, 64, generator=gn).to(dev)
            z = denoise(RHO ** 0.5 * z0 + (1 - RHO) ** 0.5 * zn if RHO < 1 else z0.clone(), c.to(AMP), u.to(AMP))
            out.append(z[0].half().cpu())
        n_out = len(out)
        rec = {"ticks": E["ticks"][a.start_tick:a.start_tick + n_out], "src_ticks": torch.tensor(srct), "lat": torch.stack(out), "ok": torch.tensor(oks),
               "hard": torch.tensor(hards), "restart": torch.tensor(restart), "amp": str(AMP), "snapshot": a.ckpt,
               "snapshot_step": int(ck.get("step", -1)), "s": -1.0, "method": f"clean_{a.mode}", "K": ck.get("K", 0), "rho": RHO, "reset_at": a.reset_at,
               "t_star": int(TS[-1]), "memory": ck.get("memory", "") if STREAM else "off",
               "swap_from": a.swap_at if E2 is not None else -1, **SETTINGS, "base_args": {k: ck["args"].get(k) for k in ("arm", "rank")},
               "code_sha": ck.get("code_sha"), "enc_sha": ENC_SHA, "ray_sha": RAY_SHA, "src_scene": src_scene, "noise_key": noise_key, "other": other if E2 is not None else "", "batch": 1}
        torch.save(rec, fo + ".tmp"); os.replace(fo + ".tmp", fo)
        print(sc, "seed", seed, n_out, "ticks", flush=True)
for di, sc in enumerate(drives):  # verify the full expected output set of this shard: settings and tick count
    if di % sn_ == si_:
        n_exp = len(torch.load(f"{a.enc}/{sc}.pt")["ticks"])
        n_exp = (min(a.max_ticks, n_exp) if a.max_ticks else n_exp) - a.start_tick
        for seed in map(int, a.seeds.split(",")):
            o_ = torch.load(f"{a.out}/{sc}_s{seed}.pt")
            assert all(o_.get(k) == v for k, v in SETTINGS.items()), f"{sc}_s{seed}: settings differ"
            assert a.swap or len(o_["lat"]) == n_exp, f"{sc}_s{seed}: {len(o_['lat'])} ticks, expected {n_exp}"
print("CLEAN_ROLLOUT_DONE", a.mode, a.split)
