"""A5 "vmix" cache builder (council/AGREED-causal-r1.md Amendment A5): frozen A@20k, memory OFF, rho 1.0, seed 0, DDIM-25, eta 0,
CFG 2, style 2, full cameras, every tick of the train drives; final latents fp16.

Same noise and inputs as scripts/causal_rollout.py --mode off --rho 1.0 (which it must reproduce: scripts/cache_check.py
re-renders one drive with the UNCHANGED causal_rollout.py and requires max |diff| < 0.01): z0 = randn from the drive-keyed CRC32
"<scene>|<seed>" generator, x_T = z0 at every tick (rho 1.0); cond = CameraAdapter(tokens, rays, style 2), uncond = the adapter's
learned null; rows [cond, uncond]; eps = eu + 2 (ec - eu). Memory-off ticks are independent given their noise, so --batch ticks
of one drive are denoised together (rows [cond x B, uncond x B]). Hard-break ticks are written as zeros with restart flags,
exactly as causal_rollout does. Output = causal_rollout's per-drive format (<out>/<scene>_s<seed>.pt) with signed fields
(ckpt_sha, roll_code of THIS script + mrv/sd.py, rho, method "cache_off", seed, batch size, merge flag).
--merge folds the LoRA into the base weights (peft BaseTunerLayer.merge()); OFF by default (parity first).

  python scripts/build_cache.py --ckpt runs/longA/ckpt_20000.pt --out runs/cache/A20k_off_rho1_train --shard 0/10 [--batch 8]
"""
import argparse, hashlib, io, json, os, time, zlib
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/longA/ckpt_20000.pt")
p.add_argument("--split", default="train", choices=["train", "dev", "decision", "final"])
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--batch", type=int, default=8, help="ticks denoised together (2 x batch UNet rows with CFG)")
p.add_argument("--merge", action="store_true", help="fold LoRA into the weights (speed); off by default")
p.add_argument("--drive_list", default="")
p.add_argument("--shard", default="0/1")
p.add_argument("--out", required=True)
a = p.parse_args()
dev = "cuda"; SD = "sd-legacy/stable-diffusion-v1-5"; RHO = 1.0
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16  # == causal_rollout's rule
os.makedirs(a.out, exist_ok=True)
_raw = open(a.ckpt, "rb").read()
CKPT_SHA = hashlib.sha256(_raw).hexdigest()[:16]  # same digest convention as causal_rollout
ROLL_CODE = hashlib.sha256(b"".join(open(f, "rb").read() for f in [os.path.relpath(__file__), "mrv/sd.py"])).hexdigest()[:12]
ck = torch.load(io.BytesIO(_raw), map_location="cpu"); r = ck["args"]["rank"]; del _raw
if "temporal" in ck:
    raise SystemExit("the cache is rendered by the plain frozen renderer (A@20k), not a temporal checkpoint")
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS)); res_ = unet.load_state_dict(ck["lora"], strict=False)
assert not res_.unexpected_keys and {n for n, _ in unet.named_parameters() if "lora" in n} == set(ck["lora"]), "LoRA load mismatch"
unet.to(dev, AMP).eval()
if a.merge:
    from peft.tuners.tuners_utils import BaseTunerLayer
    n_m = 0
    for m in unet.modules():
        if isinstance(m, BaseTunerLayer):
            m.merge(); n_m += 1
    print(f"merged LoRA into {n_m} layers", flush=True)
adapter = CameraAdapter(n_styles=ck["args"].get("n_styles", 3)).to(dev).eval(); adapter.load_state_dict(ck["adapter"])
ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler"); ddim.set_timesteps(25); TS = ddim.timesteps
RAYS = torch.load(a.rays)
drives = json.load(open(a.lists))[a.split]
if a.drive_list:
    want_ = set(open(a.drive_list).read().split())
    if not want_ <= set(drives):
        raise SystemExit(f"--drive_list drives not in split {a.split}: {sorted(want_ - set(drives))}")
    drives = [d_ for d_ in drives if d_ in want_]
missing_ = [d_ for d_ in drives if not os.path.exists(f"{a.enc}/{d_}.pt")]
if missing_:
    raise SystemExit(f"missing encodings: {missing_}")
SETTINGS = {"snapshot": a.ckpt, "ckpt_sha": CKPT_SHA, "roll_code": ROLL_CODE, "amp": str(AMP), "method": "cache_off", "rho": RHO,
            "swap": False, "swap_at": -1, "salt_from": -1, "start_tick": 0, "reset_at": -1, "max_ticks": 0,
            "seed": a.seed, "batch": a.batch, "merge": a.merge, "ddim_timesteps": [int(x) for x in TS]}


def denoise(z, c, u):
    """z: (B, 4, 32, 64) fp32; c / u: (B, 77', 768) -> final latents; rows [cond x B, uncond x B]"""
    for t in TS:
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
            e = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([c, u])).sample.float()
        ec, eu = e.chunk(2); z = ddim.step(eu + 2.0 * (ec - eu), t, z).prev_sample
    return z


si_, sn_ = map(int, a.shard.split("/"))
st1 = torch.full((1,), 2, dtype=torch.long, device=dev)
U_NULL = [None]
t_all, n_all = time.time(), 0
for di, sc in enumerate(drives):
    if di % sn_ != si_:
        continue
    _eb = open(f"{a.enc}/{sc}.pt", "rb").read(); ENC_SHA = hashlib.sha256(_eb).hexdigest()[:16]
    E = torch.load(io.BytesIO(_eb)); del _eb
    RAY_SHA = hashlib.sha256(RAYS[sc].float().numpy().tobytes()).hexdigest()[:16]
    fo = f"{a.out}/{sc}_s{a.seed}.pt"
    if os.path.exists(fo):  # reuse only an artifact made with exactly these settings
        old_ = torch.load(fo)
        bad_ = {k: (old_.get(k), v) for k, v in {**SETTINGS, "enc_sha": ENC_SHA, "ray_sha": RAY_SHA}.items() if old_.get(k) != v}
        if bad_:
            raise SystemExit(f"{fo} exists with different settings {bad_}: use a new --out")
        continue
    t0 = time.time()
    z0 = torch.randn(1, 4, 32, 64, generator=torch.Generator().manual_seed(zlib.crc32(f"{sc}|{a.seed}".encode()))).to(dev)  # signed A tensor
    n = len(E["ticks"]); ry = RAYS[sc][None].to(dev).float()
    hard = [bool(h) for h in E["hard"]]
    restart, fresh = [], True
    for i in range(n):  # restart flags exactly as causal_rollout (first tick, and the tick after every hard break)
        if hard[i]:
            restart.append(True); fresh = True
        else:
            restart.append(fresh); fresh = False
    lat = torch.zeros(n, 4, 32, 64, dtype=torch.float16)  # hard ticks stay zeros
    todo = [i for i in range(n) if not hard[i]]
    for b0 in range(0, len(todo), a.batch):
        idx = todo[b0:b0 + a.batch]; B = len(idx)
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
            c = adapter(E["tokens"][idx].to(dev).float(), ry.expand(B, *ry.shape[1:]), st1.expand(B))
            if U_NULL[0] is None:  # input-independent learned null (causal_rollout computes it once the same way)
                U_NULL[0] = adapter(E["tokens"][idx[:1]].to(dev).float(), ry, st1, torch.ones(1, dtype=torch.bool, device=dev))
        z = denoise(z0.expand(B, -1, -1, -1).clone(), c.to(AMP), U_NULL[0].to(AMP).expand(B, -1, -1))
        lat[idx] = z.half().cpu()
    tids = [int(x) for x in E["ticks"]]
    rec = {"ticks": E["ticks"][:n], "src_ticks": torch.tensor(tids), "lat": lat, "ok": torch.tensor([bool(x) for x in E["ok"]]),
           "hard": torch.tensor(hard), "restart": torch.tensor(restart), "snapshot_step": int(ck.get("step", -1)), "s": -1.0, "K": 0,
           "swap_from": -1, **SETTINGS, "base_args": {k: ck["args"].get(k) for k in ("arm", "rank")}, "code_sha": ck.get("code_sha"),
           "enc_sha": ENC_SHA, "ray_sha": RAY_SHA, "src_scene": [sc] * n, "other": "",
           "noise_key": [f"{sc}|{a.seed}|{tids[i]}" for i in range(n) if not hard[i]]}
    torch.save(rec, fo + ".tmp"); os.replace(fo + ".tmp", fo)
    n_all += len(todo)
    print(sc, n, "ticks", len(todo), "rendered", f"{(time.time() - t0) / max(1, len(todo)):.3f} s/tick", flush=True)
for di, sc in enumerate(drives):  # verify this shard's full output set
    if di % sn_ == si_:
        o_ = torch.load(f"{a.out}/{sc}_s{a.seed}.pt")
        assert all(o_.get(k) == v for k, v in SETTINGS.items()), f"{sc}: settings differ"
        assert len(o_["lat"]) == len(torch.load(f"{a.enc}/{sc}.pt")["ticks"]), f"{sc}: tick count"
print(f"CACHE_BUILD_DONE shard {a.shard}: {n_all} ticks rendered in {(time.time() - t_all) / 3600:.2f} h")
