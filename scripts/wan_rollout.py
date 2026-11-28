"""PLAN-video S1 dev panel: bidirectional sampling of whole drives with the Wan teacher (scripts/train_wan_teacher.py checkpoint).

Per drive and seed, every continuous segment (non-hard ticks, same seq, consecutive tick ids; a hard tick = black frame and a
restart) is generated as ONE latent stream, the way the training streams were encoded: the segment's guide video goes through
the frozen Wan VAE in one call (first latent = tick 0 alone, then 4 ticks per latent), and the stream's latents are generated in
WINDOWS of --window_latents (default 5 = the training clip): the first window of a segment is a true start (17 ticks), the
following ones are mid-stream windows (20 ticks), exactly the two clip kinds of training; a segment tail is covered by an
overlapping last window (only its new latents are kept). Each window is sampled independently (bidirectional inside the
window; no information crosses a window border -> restart=True on each window's first new tick). The concatenated stream is
decoded by the Wan VAE in one call (causal decoder, so window borders are decoded with real context) -> RGB per tick.
--window_latents 0 = the whole segment in one window (Wan was pretrained on up to 21 latents; try 9 / 13 / 21 on the panel).

Sampling: --sampler unipc (Wan's default solver; flow_prediction, flow sigmas, shift --shift; constructed locally, no hub
download) or euler (x <- x + (s_next - s) v on the shifted grid); --steps (default 30); --cfg (1.0 = conditional only;
> 1: rows [cond, uncond], uncond = guide AND tokens dropped, the trained 2 % case). --cond hybrid | guide | tokens | none = the
end-of-S1 ablation from the same checkpoint at fixed noise (noise depends only on drive / seed / segment / window).
Noise per window: torch.Generator(crc32("<scene>|<seed>|<segment first tick>|<window first latent>")) on the CPU.
--swap: from tick --swap_at on, the guide + tokens of another dev drive (scripts/causal_rollout.py pairing) are fed inside the
same stream (camera-swap check; the scorer reads swap_from / other / src_ticks).

Output <out>/<scene>_s<seed>.pt in the signed rollout layout: ticks, src_ticks, ok, hard, restart, "rgb" uint8 [T,3,256,512]
(read by scripts/score_rgb.py: the signed scorer on RGB directly) and, with --sd_lat (default on), "lat" = SD-VAE latents of the
same RGB (x 0.18215, posterior mode; scripts/temporal_metrics.py can read it directly, but that adds the SD codec), plus
settings / checkpoint / code hashes. --mp4 also writes <out>/mp4/<scene>_s<seed>.mp4 (decoded Wan RGB; --mp4_keys puts the
answer key on the right).
  python scripts/wan_rollout.py --ckpt runs/wan_s1/ckpt_4000.pt --split dev --seeds 0,1 --out runs/wroll/s1_4k_dev [--mp4]
"""
import argparse, hashlib, io, json, os, subprocess, tempfile, zlib
import numpy as np
import torch
from PIL import Image
from mrv import wan_cond as WC

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--split", default="dev", choices=["dev", "decision", "final", "train"])
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--guides", default="data/guides_v1")
p.add_argument("--tokens", default="data/wan_tok_v1")
p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--seeds", default="0,1")
p.add_argument("--drive_list", default="")
p.add_argument("--steps", type=int, default=30)
p.add_argument("--sampler", default="unipc", choices=["unipc", "euler"])
p.add_argument("--shift", type=float, default=None, help="sampling shift; default = the training shift stored in the checkpoint")
p.add_argument("--cfg", type=float, default=1.0)
p.add_argument("--cond", default="hybrid", choices=["hybrid", "guide", "tokens", "none"])
p.add_argument("--window_latents", type=int, default=5)
p.add_argument("--max_ticks", type=int, default=0)
p.add_argument("--swap", action="store_true")
p.add_argument("--swap_at", type=int, default=50)
p.add_argument("--sd_lat", type=int, default=1)
p.add_argument("--mp4", action="store_true")
p.add_argument("--mp4_keys", action="store_true")
p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
p.add_argument("--shard", default="0/1")
p.add_argument("--out", required=True)
a = p.parse_args()
dev = "cuda"
DT = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(a.dtype) or (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16)
os.makedirs(a.out, exist_ok=True)
_raw = open(a.ckpt, "rb").read(); CKPT_SHA = hashlib.sha256(_raw).hexdigest()[:16]
ck = torch.load(io.BytesIO(_raw), map_location="cpu", weights_only=False); del _raw
if ck.get("timing"):
    print("WARNING: timing / test checkpoint", flush=True)
CFG = ck["cfg"]; SHIFT = a.shift if a.shift is not None else CFG["shift"]
ROLL_CODE = hashlib.sha256(b"".join(open(f, "rb").read() for f in [__file__, "mrv/wan_cond.py"])).hexdigest()[:12]
prompt = torch.load(CFG["prompt_embed"], map_location="cpu") if CFG.get("prompt_embed") else None
model = WC.build(CFG["wan"], rank=CFG["rank"], tok_ticks=CFG["tok_ticks"], use_tokens=CFG["use_tokens"], dtype=DT, device=dev, prompt=prompt)
model.load_trainable(ck["trainable"]); model.eval()
if DT == torch.bfloat16 and WC.base_sha(model) != ck["base_sha"]:  # (fp16 load changes the frozen bytes; skip the check there)
    raise SystemExit("frozen base weights differ from the checkpoint's base_sha")
from diffusers import AutoencoderKLWan, AutoencoderKL
vae = AutoencoderKLWan.from_pretrained(CFG["wan"], subfolder="vae", torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
if WC.vae_sha(vae) != ck["vae_sha"]:
    raise SystemExit("Wan VAE on disk differs from the one the teacher was trained with")
sdv = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True).to(dev).eval() if a.sd_lat else None
USE_TOK = CFG["use_tokens"]; TT = CFG["tok_ticks"]; S_SLOTS = WC.SLOTS[TT]
SETTINGS = {"snapshot": a.ckpt, "ckpt_sha": CKPT_SHA, "roll_code": ROLL_CODE, "amp": str(DT), "method": f"wan_teacher_{a.cond}",
            "steps": a.steps, "sampler": a.sampler, "shift": SHIFT, "cfg": a.cfg, "cond": a.cond, "window_latents": a.window_latents,
            "swap": a.swap, "swap_at": a.swap_at, "max_ticks": a.max_ticks, "sd_lat": a.sd_lat}


def sigmas_euler(n):
    s = torch.linspace(1.0, 0.0, n + 1)
    return WC.sigma_shift(s, SHIFT)


def make_unipc():
    from diffusers import UniPCMultistepScheduler
    return UniPCMultistepScheduler(num_train_timesteps=1000, prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=SHIFT,
                                   solver_order=2, predict_x0=True, solver_type="bh2", lower_order_final=True, final_sigmas_type="zero",
                                   timestep_spacing="linspace")


SCHED = make_unipc() if a.sampler == "unipc" else None


@torch.no_grad()
def velocity(z, extra, t, tok, ray):
    """conditional (and unconditional for CFG) flow prediction for one window"""
    gk = a.cond in ("hybrid", "guide"); tk = a.cond in ("hybrid", "tokens") and USE_TOK
    ex = extra if gk else torch.zeros_like(extra)
    keep = torch.tensor([tk], device=dev)
    rows_z, rows_x, rows_k = [z], [ex], [keep]
    if a.cfg != 1.0:
        rows_z.append(z); rows_x.append(torch.zeros_like(extra)); rows_k.append(torch.tensor([False], device=dev))
    B = len(rows_z); zz = torch.cat(rows_z); xx = torch.cat(rows_x); kk = torch.cat(rows_k)
    tt = tok.expand(B, *tok.shape[1:]) if tok is not None else None
    rr = ray.expand(B, *ray.shape[1:]) if ray is not None else None
    with torch.autocast("cuda", dtype=DT):
        v = model(zz.to(DT), xx.to(DT), t.expand(B).float(), tt, rr, kk).float()
    if a.cfg != 1.0:
        vc, vu = v.chunk(2); return vu + a.cfg * (vc - vu)
    return v


@torch.no_grad()
def sample(z, extra, tok, ray):
    if a.sampler == "euler":
        sg = sigmas_euler(a.steps).to(dev)
        for i in range(a.steps):
            v = velocity(z, extra, (sg[i] * 1000).view(1), tok, ray)
            z = z + (sg[i + 1] - sg[i]) * v
        return z
    SCHED.set_timesteps(a.steps, device=dev)
    for t in SCHED.timesteps:
        v = velocity(z, extra, t.view(1).to(dev), tok, ray)
        z = SCHED.step(v, t, z, return_dict=False)[0]
    return z


def load_drive(sc):
    E = torch.load(f"{a.enc}/{sc}.pt", map_location="cpu", weights_only=False)
    G = torch.load(f"{a.guides}/{sc}.pt", map_location="cpu", weights_only=False)
    T = torch.load(f"{a.tokens}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True) if USE_TOK else None
    if T is not None:
        assert torch.equal(T["ticks"], E["ticks"]), f"{sc}: token ticks != enc ticks"
    return {"E": E, "G": G, "gpos": {int(t): k for k, t in enumerate(G["ticks"].tolist())}, "T": T, "sc": sc}


def guide_rgb(D, j):
    t = int(D["E"]["ticks"][j]); k = D["gpos"].get(t)
    if k is None:
        return torch.zeros(3, 256, 512, dtype=torch.uint8), torch.zeros(1, 256, 512)
    return D["G"]["rgb"][k], D["G"]["mask"][k].float()


drives = json.load(open(a.lists))[a.split]; ALL = list(drives)
if a.drive_list:
    want = set(open(a.drive_list).read().split())
    if not want <= set(drives):
        raise SystemExit(f"--drive_list drives not in split {a.split}: {sorted(want - set(drives))}")
    drives = [d for d in drives if d in want]
miss = [d for d in drives if not all(os.path.exists(f) for f in [f"{a.enc}/{d}.pt", f"{a.guides}/{d}.pt"] + ([f"{a.tokens}/{d}.pt"] if USE_TOK else []))]
if miss:
    raise SystemExit(f"missing enc / guide / token files: {miss}")
si_, sn_ = map(int, a.shard.split("/"))
if a.mp4:
    os.makedirs(f"{a.out}/mp4", exist_ok=True)


def write_mp4(frames, keys, path):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        for i, f in enumerate(frames):
            im = f.permute(1, 2, 0).numpy()
            if keys is not None:
                k = keys[i] if keys[i] is not None else np.zeros_like(im)
                im = np.concatenate([im, k], 1)
            Image.fromarray(im).save(f"{tmp}/f{i:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", f"{tmp}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", path], check=True)


for di, sc in enumerate(drives):
    if di % sn_ != si_:
        continue
    D = load_drive(sc); E = D["E"]
    other = ALL[(ALL.index(sc) + len(ALL) // 2) % len(ALL)]
    D2 = load_drive(other) if a.swap else None
    n = len(E["ticks"]) if not a.max_ticks else min(a.max_ticks, len(E["ticks"]))
    if D2 is not None:
        n = min(n, len(D2["E"]["ticks"]))
    src = [(D2, i) if (D2 is not None and i >= a.swap_at) else (D, i) for i in range(n)]
    for seed in map(int, a.seeds.split(",")):
        fo = f"{a.out}/{sc}_s{seed}.pt"
        if os.path.exists(fo):
            old = torch.load(fo, map_location="cpu", weights_only=False)
            bad = {k: (old.get(k), v) for k, v in SETTINGS.items() if old.get(k) != v}
            if bad:
                raise SystemExit(f"{fo} exists with different settings {bad}: use a new --out")
            continue
        rgb = torch.zeros(n, 3, 256, 512, dtype=torch.uint8); restart = [False] * n; windows = []
        oks = [bool(S["E"]["ok"][j]) for S, j in src]; hards = [bool(S["E"]["hard"][j]) for S, j in src]
        srct = [int(S["E"]["ticks"][j]) for S, j in src]
        # segments: break at a hard tick (of the source), a receiver tick gap / seq change, or a source tick gap on the same drive
        segs, cur = [], []
        for i in range(n):
            if hards[i]:
                if cur: segs.append(cur)
                cur = []; restart[i] = True; continue
            if cur:
                pi = cur[-1]; (Sa, ja), (Sb, jb) = src[pi], src[i]
                brk = int(E["ticks"][i]) != int(E["ticks"][pi]) + 1 or int(E["seq"][i]) != int(E["seq"][pi]) or \
                      (Sa is Sb and (int(Sb["E"]["ticks"][jb]) != int(Sa["E"]["ticks"][ja]) + 1 or int(Sb["E"]["seq"][jb]) != int(Sa["E"]["seq"][ja])))
                if brk:
                    segs.append(cur); cur = []
            cur.append(i)
        if cur: segs.append(cur)
        for seg in segs:
            Ls = len(seg); Lz = 1 + -(-(Ls - 1) // 4); Fv = 4 * Lz - 3
            fr = seg + [seg[-1]] * (Fv - Ls)  # pad the tail by repeating the last tick's inputs (outputs discarded)
            gr, gm = zip(*[guide_rgb(*src[i]) for i in fr])
            gvid = torch.stack(gr, 1).float().div(127.5).sub(1)[None].to(dev)  # [1,3,Fv,256,512]
            zg = WC.wan_encode(vae, gvid)  # [1,16,Lz,32,64]
            mlr = WC.mask_lowres(torch.stack(gm))  # [Fv,32,64]
            spans = WC.latent_spans(Lz, first_single=True)
            mask = WC.fold_mask(mlr, spans)[None].to(dev)  # [1,4,Lz,32,64]
            extra = torch.cat([zg, mask], 1)
            tok = ray = None
            if USE_TOK:
                tl, rl = [], []
                for sp in spans:
                    for f in WC.token_slots(sp, TT):
                        S, j = src[fr[f]]; tl.append(S["T"]["tokens"][j]); rl.append(S["T"]["rays"])
                tok = torch.stack(tl).view(1, Lz, S_SLOTS, *tl[0].shape).to(dev)
                ray = torch.stack(rl).view(1, Lz, S_SLOTS, *rl[0].shape).to(dev)
            W = a.window_latents if a.window_latents > 0 else Lz
            starts = list(range(0, Lz, W))
            if Lz >= W and starts[-1] + W > Lz:
                starts[-1] = Lz - W
            Z = torch.zeros(1, 16, Lz, 32, 64, device=dev); done = 0
            t_first = int(E["ticks"][seg[0]])
            for w0 in starts:
                w1 = min(w0 + W, Lz)
                gn = torch.Generator().manual_seed(zlib.crc32(f"{sc}|{seed}|{t_first}|{w0}".encode()))
                z = torch.randn(1, 16, w1 - w0, 32, 64, generator=gn).to(dev)
                z = sample(z, extra[:, :, w0:w1], tok[:, w0:w1] if tok is not None else None, ray[:, w0:w1] if ray is not None else None)
                new0 = max(done, w0)
                Z[:, :, new0:w1] = z[:, :, new0 - w0:]
                first_frame = 0 if new0 == 0 else 4 * new0 - 3
                if first_frame < Ls:
                    restart[seg[first_frame]] = True
                windows.append([int(E["ticks"][seg[0]]), w0, w1, new0]); done = w1
            x = WC.wan_decode(vae, Z)[0]  # [3, Fv, 256, 512]
            rgb[seg] = ((x[:, :Ls].permute(1, 0, 2, 3) + 1) * 127.5).round().clamp(0, 255).byte().cpu()
        rec = {"ticks": E["ticks"][:n].clone(), "src_ticks": torch.tensor(srct), "ok": torch.tensor(oks), "hard": torch.tensor(hards),
               "restart": torch.tensor(restart), "rgb": rgb, "s": -1.0, "snapshot_step": int(ck.get("step", -1)), "windows": windows,
               "swap_from": a.swap_at if D2 is not None else -1, "other": other if D2 is not None else "", "seed": seed,
               "code_sha": ck.get("code_sha"), "base_sha": ck.get("base_sha"), "batch": 1, **SETTINGS}
        if sdv is not None:
            lats = []
            with torch.no_grad():
                for i in range(0, n, 16):
                    xx = rgb[i:i + 16].to(dev).float() / 127.5 - 1
                    lats.append((sdv.encode(xx).latent_dist.mode() * 0.18215).half().cpu())
            rec["lat"] = torch.cat(lats)
            rec["lat"][torch.tensor(hards, dtype=torch.bool)] = 0  # hard ticks: zero latent, as scripts/causal_rollout.py
        torch.save(rec, fo + ".tmp"); os.replace(fo + ".tmp", fo)
        if a.mp4:
            keys = None
            if a.mp4_keys:
                keys = []
                for i, (S, j) in enumerate(src):
                    kp = f"{a.keys}/{S['sc']}/t{srct[i]:04d}.jpg"
                    keys.append(np.asarray(Image.open(kp).convert("RGB").resize((512, 288), Image.BILINEAR).crop((0, 32, 512, 288))) if os.path.exists(kp) else None)
            write_mp4(rgb, keys, f"{a.out}/mp4/{sc}_s{seed}.mp4")
        print(sc, "seed", seed, n, "ticks", len(segs), "segments", len(windows), "windows", flush=True)
print("WAN_ROLLOUT_DONE", a.split, a.cond)
