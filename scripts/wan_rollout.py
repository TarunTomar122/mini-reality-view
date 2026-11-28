"""PLAN-video S1 dev panel: bidirectional sampling of whole drives with the Wan teacher (scripts/train_wan_teacher.py checkpoint).

Streams (Astra item 3): a segment = maximal run of non-hard ticks with consecutive tick ids (mrv.wan_plan.segments; soft rejects
stay inside the stream, a hard tick = black frame + restart). Each segment's guide video is encoded by the frozen Wan VAE in one
call (as the training streams, phase 0), its latents are generated in WINDOWS of --window_latents (default 5 = the training
clip; first window = true start, later ones mid-stream, an overlapping tail window keeps only its new latents:
mrv.wan_plan.plan_windows), and the concatenated stream is decoded once (causal decoder) -> RGB per tick. Windows are sampled
independently (bidirectional inside), so restart = True on each window's first new tick (the drift check does not apply to S1).
Inputs: guides checked with mrv.wan_cond.check_guide against the CHECKPOINT's guide version (train and dev guides must be the
same version); tokens (+ per-tick rays) from the token cache, whose sidecar must carry the same guide version.
Sampling: --sampler unipc (Wan's solver, flow_prediction, flow sigmas, shift; constructed locally) or euler; --steps 30;
--cfg (1.0 = conditional; > 1: rows [cond, uncond], uncond = guide AND tokens dropped); --cond hybrid | guide | tokens | none =
the end-of-S1 ablation at FIXED noise (noise = crc32("<scene>|<seed>|<segment first tick>|<window first latent>")).
--swap: from the first window start at or after --swap_at (mrv.wan_plan.snap_swap: same window borders and noise as the
unswapped run) the guide + tokens of the donor drive (scripts/causal_rollout.py pairing) are fed; swap_from / other / src_ticks
for the scorer; "windows_match_unswapped" / "swap_boundaries_matched" record whether the run reproduces the window borders and
noise keys of the unswapped full-length receiver run (donor hard ticks or a shorter donor change them; the panel then blocks).
Sensitivity (Astra items 4 / S1b 3): once per run, on the first window (sigma 0.6), the output change from removing the guide and
from perturbing the token VALUES at a fixed context length (views shifted), next to an init control (guide conv + gates zeroed);
a checkpoint >= 1000 updates must be finite and change by > max(10 x control, 1e-3 x |v|) on each path, else it is refused.
Sealed splits (decision / final) need --unseal (recipe lock + N1 first). Slice time is debited to --ledger (S1 budget) unless
--ledger '' (scripts/wan_panel.py debits for its rollouts itself).
Output <out>/<scene>_s<seed>.pt: ticks, src_ticks, ok, hard, restart, "rgb" uint8 [T,3,256,512] (scripts/score_rgb.py),
"lat" SD-VAE latents of the same RGB (--sd_lat 1), windows, settings, timing; --mp4 writes <out>/mp4/<scene>_s<seed>.mp4.
  python scripts/wan_rollout.py --ckpt runs/wan_s1/ckpt_12000.pt --split dev --seeds 0,1 --out runs/wroll/s1_12k_hybrid_dev
"""
import argparse, hashlib, io, json, os, subprocess, tempfile, time, zlib
import numpy as np
import torch
from PIL import Image
from mrv import wan_cond as WC
from mrv import wan_plan as WP

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--split", default="dev", choices=["dev", "decision", "final", "train"])
p.add_argument("--unseal", action="store_true")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--guides", default="data/guides_v1")
p.add_argument("--tokens", default="data/wan_tok_v1")
p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--seeds", default="0,1")
p.add_argument("--drive_list", default="")
p.add_argument("--steps", type=int, default=30)
p.add_argument("--sampler", default="unipc", choices=["unipc", "euler"])
p.add_argument("--shift", type=float, default=None)
p.add_argument("--cfg", type=float, default=1.0)
p.add_argument("--cond", default="hybrid", choices=["hybrid", "guide", "tokens", "none"])
p.add_argument("--window_latents", type=int, default=5)
p.add_argument("--max_ticks", type=int, default=0)
p.add_argument("--swap", action="store_true")
p.add_argument("--swap_at", type=int, default=50)
p.add_argument("--donors", default="", help="swap donor plan json (scripts/wan_swap_preflight.py): receiver -> {donor, mode full|shared, n}")
p.add_argument("--sd_lat", type=int, default=1)
p.add_argument("--mp4", action="store_true")
p.add_argument("--mp4_keys", action="store_true")
p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
p.add_argument("--ledger", default="runs/wan_s1_ledger.jsonl")
p.add_argument("--allow_inert", action="store_true")
p.add_argument("--shard", default="0/1")
p.add_argument("--out", required=True)
a = p.parse_args()
if a.split in ("decision", "final") and not a.unseal:
    raise SystemExit(f"{a.split} drives are sealed until recipe lock + N1 (--unseal)")
dev = "cuda"
LEDGER = WP.Ledger(a.ledger, "rollout").start()
DT = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(a.dtype) or (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16)
os.makedirs(a.out, exist_ok=True)
_raw = open(a.ckpt, "rb").read(); CKPT_SHA = hashlib.sha256(_raw).hexdigest()[:16]
ck = torch.load(io.BytesIO(_raw), map_location="cpu", weights_only=False); del _raw
if ck.get("timing"):
    print("WARNING: timing / test checkpoint", flush=True)
CFG = ck["cfg"]; SHIFT = a.shift if a.shift is not None else CFG["shift"]; GV = ck["guide_version"]
if CFG.get("n_extra") != WC.N_EXTRA or tuple(CFG.get("tok_grid", WC.TOK_GRID)) != WC.TOK_GRID:
    raise SystemExit(f"checkpoint layout {CFG.get('n_extra')} / {CFG.get('tok_grid')} != code {WC.N_EXTRA} / {WC.TOK_GRID}")
ROLL_CODE = WC.code_sha(__file__, "mrv/wan_cond.py", "mrv/wan_plan.py")
prompt = torch.load(CFG["prompt_embed"], map_location="cpu") if CFG.get("prompt_embed") else None
model = WC.build(CFG["wan"], rank=CFG["rank"], tok_ticks=CFG["tok_ticks"], use_tokens=CFG["use_tokens"], dtype=DT, device=dev, prompt=prompt,
                 seed=CFG.get("seed", 0))
model.load_trainable(ck["trainable"]); model.eval()
if DT == torch.bfloat16 and WC.base_sha(model) != ck["base_sha"]:
    raise SystemExit("frozen base weights differ from the checkpoint's base_sha")
from diffusers import AutoencoderKLWan, AutoencoderKL
vae = AutoencoderKLWan.from_pretrained(CFG["wan"], subfolder="vae", torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
if WC.vae_sha(vae) != ck["vae_sha"]:
    raise SystemExit("Wan VAE on disk differs from the one the teacher was trained with")
sdv = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True).to(dev).eval() if a.sd_lat else None
USE_TOK = CFG["use_tokens"]; TT = CFG["tok_ticks"]; S_SLOTS = WC.SLOTS[TT]
SETTINGS = {"snapshot": a.ckpt, "ckpt_sha": CKPT_SHA, "roll_code": ROLL_CODE, "amp": str(DT), "method": f"wan_teacher_{a.cond}",
            "steps": a.steps, "sampler": a.sampler, "shift": SHIFT, "cfg": a.cfg, "cond": a.cond, "window_latents": a.window_latents,
            "swap": a.swap, "swap_at": a.swap_at, "max_ticks": a.max_ticks, "sd_lat": a.sd_lat, "guide_version": GV}


def make_unipc():
    from diffusers import UniPCMultistepScheduler
    return UniPCMultistepScheduler(num_train_timesteps=1000, prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=SHIFT,
                                   solver_order=2, predict_x0=True, solver_type="bh2", lower_order_final=True, final_sigmas_type="zero",
                                   timestep_spacing="linspace")


SCHED = make_unipc() if a.sampler == "unipc" else None


@torch.no_grad()
def velocity(z, extra, t, tok, ray):
    gk = a.cond in ("hybrid", "guide"); tk = a.cond in ("hybrid", "tokens") and USE_TOK
    rows_z, rows_x, rows_k = [z], [extra if gk else torch.zeros_like(extra)], [torch.tensor([tk], device=dev)]
    if a.cfg != 1.0:
        rows_z.append(z); rows_x.append(torch.zeros_like(extra)); rows_k.append(torch.tensor([False], device=dev))
    B = len(rows_z)
    tt = tok.expand(B, *tok.shape[1:]) if tok is not None else None
    rr = ray.expand(B, *ray.shape[1:]) if ray is not None else None
    with torch.autocast("cuda", dtype=DT):
        v = model(torch.cat(rows_z).to(DT), torch.cat(rows_x).to(DT), t.expand(B).float(), tt, rr, torch.cat(rows_k)).float()
    if a.cfg != 1.0:
        vc, vu = v.chunk(2); return vu + a.cfg * (vc - vu)
    return v


@torch.no_grad()
def sample(z, extra, tok, ray):
    if a.sampler == "euler":
        sg = WC.sigma_shift(torch.linspace(1.0, 0.0, a.steps + 1), SHIFT).to(dev)
        for i in range(a.steps):
            z = z + (sg[i + 1] - sg[i]) * velocity(z, extra, (sg[i] * 1000).view(1), tok, ray)
        return z
    SCHED.set_timesteps(a.steps, device=dev)
    for t in SCHED.timesteps:
        z = SCHED.step(velocity(z, extra, t.view(1).to(dev), tok, ray), t, z, return_dict=False)[0]
    return z


def load_drive(sc):
    E = torch.load(f"{a.enc}/{sc}.pt", map_location="cpu", weights_only=False)
    G = torch.load(f"{a.guides}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True)
    WC.check_guide(f"{a.guides}/{sc}.pt", G, json.load(open(f"{a.guides}/{sc}.json")), E["ticks"].tolist(), E["hard"].tolist(), sc, GV)
    T = None
    if USE_TOK:  # S1b item 2: token file, its upstream guide / enc files and its completeness evidence verified per drive
        sj = json.load(open(f"{a.tokens}/{sc}.json"))
        if not sj.get("complete") or sj.get("guide_version") != GV or not sj.get("shared_pass_ok") or not sj.get("weights_sha"):
            raise SystemExit(f"{a.tokens}/{sc}: token cache incomplete, without shared-depth evidence, or from another guide version")
        if WC.file_sha(f"{a.tokens}/{sc}.pt") != sj["sha256"]:
            raise SystemExit(f"{a.tokens}/{sc}.pt does not match its sidecar")
        if WC.file_sha(f"{a.guides}/{sc}.pt") != sj.get("guide_sha") or WC.file_sha(f"{a.enc}/{sc}.pt") != sj.get("enc_sha"):
            raise SystemExit(f"{sc}: guide / enc file differs from the one the token cache was built from")
        T = torch.load(f"{a.tokens}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True)
        assert torch.equal(T["ticks"], E["ticks"]), f"{sc}: token ticks != enc ticks"
    return {"E": E, "G": G, "T": T, "sc": sc, "src_lr": WC.src_lowres(G["src"])}


drives = json.load(open(a.lists))[a.split]; ALL = list(drives)
WP.check_split_membership(drives, json.load(open(a.lists)), allowed=(a.split,), unseal=a.unseal)
if a.drive_list:
    want = set(WP.check_split_membership(open(a.drive_list).read().split(), json.load(open(a.lists)), allowed=(a.split,), unseal=a.unseal))
    if not want <= set(drives):
        raise SystemExit(f"--drive_list drives not in split {a.split}: {sorted(want - set(drives))}")
    drives = [d for d in drives if d in want]
si_, sn_ = map(int, a.shard.split("/"))
DONORS = json.load(open(a.donors)) if a.donors else None
if DONORS is not None and not DONORS.get("ok"):
    raise SystemExit(f"{a.donors}: swap preflight did not pass")
if a.mp4:
    os.makedirs(f"{a.out}/mp4", exist_ok=True)
SENS = None


def write_mp4(frames, keys, path):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        for i, f in enumerate(frames):
            im = f.permute(1, 2, 0).numpy()
            if keys is not None:
                im = np.concatenate([im, keys[i] if keys[i] is not None else np.zeros_like(im)], 1)
            Image.fromarray(im).save(f"{tmp}/f{i:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", f"{tmp}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", path], check=True)


for di, sc in enumerate(drives):
    if di % sn_ != si_:
        continue
    D = load_drive(sc); E = D["E"]
    other, swap_mode = ALL[(ALL.index(sc) + len(ALL) // 2) % len(ALL)], "conventional"
    n_full = len(E["ticks"]) if not a.max_ticks else min(a.max_ticks, len(E["ticks"]))
    if a.swap and DONORS is not None:  # planned donor (Astra / Neon 08:30): full = donor >= receiver; shared = receiver cut to n
        dp = DONORS["donors"].get(sc)
        if not dp:
            raise SystemExit(f"{sc}: no qualifying donor in {a.donors}")
        other, swap_mode = dp["donor"], dp["mode"]
        if swap_mode == "shared":
            n_full = min(n_full, int(dp["n"]))
    D2 = load_drive(other) if a.swap else None
    n = min(n_full, len(D2["E"]["ticks"])) if D2 is not None else n_full
    # windows (and so noise keys) of the UNSWAPPED receiver run at the reference length (full, or the shared n): a swap run must
    # reproduce them exactly
    exp_windows = [[int(E["ticks"][sg[0]]), w0, w1, n0] for sg in WP.segments(E["ticks"][:n_full].tolist(), E["hard"][:n_full].tolist())
                   for w0, w1, n0 in WP.plan_windows(1 + -(-(len(sg) - 1) // 4), a.window_latents)]
    rt, rh = E["ticks"][:n].tolist(), E["hard"][:n].tolist()
    segs0 = WP.segments(rt, rh)
    W = a.window_latents
    swap_pos = None
    if D2 is not None:
        swap_pos = WP.snap_swap(segs0, a.swap_at, W)
        if swap_pos is None:
            raise SystemExit(f"{sc}: no window start at or after tick position {a.swap_at}")
    src = [(D2, i) if (swap_pos is not None and i >= swap_pos) else (D, i) for i in range(n)]
    hards = [bool(S["E"]["hard"][j]) for S, j in src]; oks = [bool(S["E"]["ok"][j]) for S, j in src]
    srct = [int(S["E"]["ticks"][j]) for S, j in src]
    if swap_pos is None:
        segs = segs0
    else:  # receiver structure + donor hard ticks / gaps after the swap (the swap itself is not a break)
        segs, cur = [], []
        for i in range(n):
            if hards[i]:
                if cur: segs.append(cur)
                cur = []; continue
            brk = bool(cur) and (rt[i] != rt[cur[-1]] + 1 or (i > swap_pos and cur[-1] >= swap_pos and srct[i] != srct[cur[-1]] + 1))
            if brk:
                segs.append(cur); cur = []
            cur.append(i)
        if cur: segs.append(cur)
    for seed in map(int, a.seeds.split(",")):
        fo = f"{a.out}/{sc}_s{seed}.pt"
        if os.path.exists(fo):
            old = torch.load(fo, map_location="cpu", weights_only=False)
            bad = {k: (old.get(k), v) for k, v in SETTINGS.items() if old.get(k) != v}
            if bad:
                raise SystemExit(f"{fo} exists with different settings {bad}: use a new --out")
            continue
        rgb = torch.zeros(n, 3, 256, 512, dtype=torch.uint8); restart = [bool(h) for h in hards]; windows = []
        tm = {"encode_s": 0.0, "sample_s": 0.0, "decode_s": 0.0, "sd_s": 0.0, "windows": 0}
        for seg in segs:
            Ls = len(seg); Lz = 1 + -(-(Ls - 1) // 4); Fv = 4 * Lz - 3
            fr = seg + [seg[-1]] * (Fv - Ls)  # tail padded by repeating the last tick's inputs (outputs discarded)
            torch.cuda.synchronize(); t0 = time.time()
            gvid = torch.stack([src[i][0]["G"]["rgb"][src[i][1]] for i in fr], 1).float().div(127.5).sub(1)[None].to(dev)
            zg = WC.wan_encode(vae, gvid)
            slr = torch.stack([src[i][0]["src_lr"][src[i][1]] for i in fr])  # [Fv, 2, 32, 64]
            spans = WP.latent_spans(Lz, first_single=True)
            extra = torch.cat([zg, WC.fold_src(slr, spans)[None].to(dev)], 1)
            tok = ray = None
            if USE_TOK:
                tl, rl = [], []
                for sp in spans:
                    for f in WP.token_slots(sp, TT):
                        S, j = src[fr[f]]; tl.append(S["T"]["tokens"][j]); rl.append(S["T"]["rays"][j])
                tok = torch.stack(tl).view(1, Lz, S_SLOTS, *tl[0].shape).to(dev)
                ray = torch.stack(rl).view(1, Lz, S_SLOTS, *rl[0].shape).float().to(dev)
            torch.cuda.synchronize(); tm["encode_s"] += time.time() - t0
            Z = torch.zeros(1, 16, Lz, 32, 64, device=dev); t_first = rt[seg[0]]
            for w0, w1, new0 in WP.plan_windows(Lz, W):
                ex_w = extra[:, :, w0:w1]; tk_w = tok[:, w0:w1] if tok is not None else None; ry_w = ray[:, w0:w1] if ray is not None else None
                if SENS is None:  # once per run: guide / token sensitivity on a real window
                    g0 = torch.Generator().manual_seed(1)
                    x0 = torch.randn(1, 16, w1 - w0, 32, 64, generator=g0).to(dev)
                    SENS = WC.sensitivity(model, x0, ex_w, torch.tensor([600.0], device=dev), tk_w, ry_w, DT)
                    ok_s, why_s = WC.check_sensitivity(SENS, USE_TOK)
                    SENS["pass"], SENS["why"] = ok_s, why_s
                    print("sensitivity (sigma 0.6)", SENS, flush=True)
                    if int(ck.get("step", 0)) >= 1000 and not a.allow_inert and not ok_s:
                        raise SystemExit(f"trained checkpoint fails the input-sensitivity check ({why_s}): {SENS}")
                gn = torch.Generator().manual_seed(zlib.crc32(f"{sc}|{seed}|{t_first}|{w0}".encode()))
                z = torch.randn(1, 16, w1 - w0, 32, 64, generator=gn).to(dev)
                torch.cuda.synchronize(); t0 = time.time()
                z = sample(z, ex_w, tk_w, ry_w)
                torch.cuda.synchronize(); tm["sample_s"] += time.time() - t0; tm["windows"] += 1
                Z[:, :, new0:w1] = z[:, :, new0 - w0:]
                ff = 0 if new0 == 0 else 4 * new0 - 3
                if ff < Ls:
                    restart[seg[ff]] = True
                windows.append([t_first, w0, w1, new0])
            t0 = time.time()
            x = WC.wan_decode(vae, Z)[0]
            rgb[seg] = ((x[:, :Ls].permute(1, 0, 2, 3) + 1) * 127.5).round().clamp(0, 255).byte().cpu()
            torch.cuda.synchronize(); tm["decode_s"] += time.time() - t0
        rec = {"ticks": E["ticks"][:n].clone(), "src_ticks": torch.tensor(srct), "ok": torch.tensor(oks), "hard": torch.tensor(hards),
               "restart": torch.tensor(restart), "rgb": rgb, "s": -1.0, "snapshot_step": int(ck.get("step", -1)), "windows": windows,
               "swap_from": swap_pos if swap_pos is not None else -1, "other": other if D2 is not None else "", "seed": seed,
               "swap_boundaries_matched": (segs == segs0 and windows == exp_windows) if D2 is not None else None,
               "windows_match_unswapped": windows == exp_windows, "truncated_by_donor": n < n_full, "sensitivity": SENS,
               "swap_mode": swap_mode if D2 is not None else None, "swap_ref_ticks": n_full if D2 is not None else None,
               "recv_ok": E["ok"][:n].clone(), "recv_hard": E["hard"][:n].clone(),  # the receiver's own flags (post-swap common frames)
               "train_dropout": {k: (ck.get("recipe") or {}).get(k) for k in ("p_drop_both", "p_drop_guide_only", "p_drop_tokens_only")},
               "code_sha": ck.get("code_sha"), "base_sha": ck.get("base_sha"), "batch": 1, **SETTINGS}
        if sdv is not None:
            t0 = time.time(); lats = []
            with torch.no_grad():
                for i in range(0, n, 16):
                    lats.append((sdv.encode(rgb[i:i + 16].to(dev).float() / 127.5 - 1).latent_dist.mode() * 0.18215).half().cpu())
            rec["lat"] = torch.cat(lats); rec["lat"][torch.tensor(hards, dtype=torch.bool)] = 0
            torch.cuda.synchronize(); tm["sd_s"] = time.time() - t0
        rec["timing"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in tm.items()}
        torch.save(rec, fo + ".tmp"); os.replace(fo + ".tmp", fo)
        if a.mp4:
            keys = None
            if a.mp4_keys:
                keys = []
                for i, (S, j) in enumerate(src):
                    kp = f"{a.keys}/{S['sc']}/t{srct[i]:04d}.jpg"
                    keys.append(np.asarray(Image.open(kp).convert("RGB").resize((512, 288), Image.BILINEAR).crop((0, 32, 512, 288))) if os.path.exists(kp) else None)
            write_mp4(rgb, keys, f"{a.out}/mp4/{sc}_s{seed}.mp4")
        print(sc, "seed", seed, n, "ticks", len(segs), "segments", len(windows), "windows", rec["timing"], flush=True)
LEDGER.stop()
print("WAN_ROLLOUT_DONE", a.split, a.cond)
