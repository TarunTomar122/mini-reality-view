"""PLAN-video S1 inputs, path (a) + targets: frozen-Wan-VAE latents of the game-render answer keys and of the chase-view guides.

Clip format (Fable 06:35 b2 / c3, adopted): the live stream sees Wan's single-frame first latent ONCE per drive, so most training
clips are cut MID-STREAM from latents of whole continuous segments:
  * segment = maximal run of non-hard ticks, same seq, consecutive tick ids (offset-only ticks with ok=False stay inside: the
    keys and guides exist; the trainer only cuts clips whose ticks are all ok).
  * PHASES: each segment is encoded 4 times, starting at segment offsets 0 / 1 / 2 / 3 (one call per phase, the causal VAE
    runs over the whole stream), so the 4-tick grouping falls four ways (4x distinct targets; no dependence on where the stream
    began). Stream latent j >= 1 covers stream frames 4j-3..4j; a mid-stream clip = 5 consecutive latents j0..j0+4, j0 >= 1
    (20 ticks, every latent a regular 4-tick latent with real causal history).
  * TRUE STARTS: 17-tick windows (1 + 4x4 frames, encoded alone: first latent = single frame) at every legal start with
    enc index % --start_stride == 0; the trainer mixes ~15 % of these in (the first window of every drive / after every hard
    break looks like this at rollout).
Keys: data/sim/keys_tpv10/<scene>/t%04d.jpg, 512x288 bilinear -> crop rows 32..288 (as temporal_metrics.py), [-1, 1].
Guides: data/guides_v1/<scene>.pt (mrv/guide.py: uint8 rgb [T,3,256,512], mask [T,1,256,512], ticks); a tick without a guide
= black frame, mask 0. Guide and key videos are encoded IDENTICALLY (same frames, same calls), latent = posterior mode,
normalised with the VAE's latents_mean / latents_std (mrv.wan_cond.wan_encode).

Per drive -> <out>/<scene>.pt:
  ticks / ok / hard / seq (enc arrays), mask_lr fp16 [n, 32, 64] (guide validity per latent cell, enc-indexed),
  streams: list of {"pos0": enc index of stream frame 0, "phase", "n_frames" (4L-3), "key" bf16 [16, L, 32, 64], "guide" same},
  starts LongTensor [S] (enc index of clip start), start_key / start_guide bf16 [S, 16, 5, 32, 64], meta (vae id + sha, guide
  file sha256, code sha) + <scene>.json sidecar (sha256 of the .pt).
Disk ~ 130 KB / tick for the streams + ~650 KB per true start (stride 4: ~6.5 GB for the train drives).
  python scripts/wan_cache.py --splits train,dev --out data/wan_cache_v1 --shard i/n
"""
import argparse, hashlib, json, os, time
import numpy as np
import torch
from PIL import Image
from mrv.wan_cond import WAN_ID, wan_encode, mask_lowres, vae_sha

p = argparse.ArgumentParser()
p.add_argument("--splits", default="train,dev")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--guides", default="data/guides_v1")
p.add_argument("--wan", default=WAN_ID)
p.add_argument("--out", default="data/wan_cache_v1")
p.add_argument("--phases", type=int, default=4)
p.add_argument("--start_stride", type=int, default=4)
p.add_argument("--frames", type=int, default=17)
p.add_argument("--clip_latents", type=int, default=5, help="latents per training clip (streams shorter than 1 + this are skipped)")
p.add_argument("--batch", type=int, default=4, help="true-start clips per VAE call")
p.add_argument("--shard", default="0/1")
a = p.parse_args()
assert (a.frames - 1) % 4 == 0 and 1 + (a.frames - 1) // 4 == a.clip_latents
dev = "cuda"
os.makedirs(a.out, exist_ok=True)
CODE_SHA = hashlib.sha1(b"".join(open(f, "rb").read() for f in [__file__, "mrv/wan_cond.py"])).hexdigest()[:12]
L_ = json.load(open(a.lists))
drives = sorted({d for s in a.splits.split(",") for d in L_[s]})
si, sn = map(int, a.shard.split("/"))
todo = [d for k, d in enumerate(drives) if k % sn == si and not os.path.exists(f"{a.out}/{d}.pt")]
print(f"{len(todo)} drives in shard {a.shard}", flush=True)
if not todo:
    raise SystemExit(0)
from diffusers import AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained(a.wan, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
vae.requires_grad_(False)
VAE_SHA = vae_sha(vae)


def sha_file(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def key_img(sc, t):
    f = f"{a.keys}/{sc}/t{t:04d}.jpg"
    if not os.path.exists(f):
        raise SystemExit(f"missing key {f}")  # never feed grey frames into the video codec (Fable 06:35 a3)
    im = Image.open(f).convert("RGB").resize((512, 288), Image.BILINEAR).crop((0, 32, 512, 288))
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1)


for sc in todo:
    t0 = time.time()
    E = torch.load(f"{a.enc}/{sc}.pt", map_location="cpu", weights_only=False)
    ticks, ok, hard, seq = E["ticks"].tolist(), E["ok"].tolist(), E["hard"].tolist(), E["seq"].tolist()
    n = len(ticks)
    gf = f"{a.guides}/{sc}.pt"
    G = torch.load(gf, map_location="cpu", weights_only=False)
    gpos = {int(t): k for k, t in enumerate(G["ticks"].tolist())}
    GSHA = sha_file(gf)
    mask_lr = torch.zeros(n, 32, 64)
    have = [i for i in range(n) if ticks[i] in gpos]
    if have:
        gm = G["mask"][[gpos[ticks[i]] for i in have]]
        mask_lr[have] = mask_lowres(gm)
    KEY = {}  # enc index -> uint8 [3,256,512] (lazy)

    def kframes(idx):
        out = []
        for i in idx:
            if i not in KEY:
                KEY[i] = key_img(sc, ticks[i])
            out.append(KEY[i])
        return torch.stack(out, 1).float() / 127.5 - 1  # [3, F, 256, 512]

    def gframes(idx):
        out = [G["rgb"][gpos[ticks[i]]] if ticks[i] in gpos else torch.zeros(3, 256, 512, dtype=torch.uint8) for i in idx]
        return torch.stack(out, 1).float() / 127.5 - 1

    # segments: non-hard, same seq, consecutive tick ids
    segs, cur = [], []
    for i in range(n):
        if hard[i]:
            if cur: segs.append(cur)
            cur = []; continue
        if cur and (seq[i] != seq[cur[-1]] or ticks[i] != ticks[cur[-1]] + 1):
            segs.append(cur); cur = []
        cur.append(i)
    if cur: segs.append(cur)
    streams = []
    for seg in segs:
        for ph in range(a.phases):
            fr = seg[ph:]
            Lz = 1 + (len(fr) - 1) // 4 if fr else 0
            if Lz < 1 + a.clip_latents:  # needs at least one mid-stream clip (j0 >= 1)
                continue
            fr = fr[:4 * Lz - 3]
            zk = wan_encode(vae, kframes(fr)[None].to(dev))[0].to(torch.bfloat16).cpu()
            zg = wan_encode(vae, gframes(fr)[None].to(dev))[0].to(torch.bfloat16).cpu()
            assert zk.shape[1] == Lz, (zk.shape, Lz)
            streams.append({"pos0": fr[0], "phase": ph, "n_frames": len(fr), "key": zk, "guide": zg})
        KEY.clear()
    # true starts
    Fn = a.frames
    legal = [i for i in range(n - Fn + 1) if i % a.start_stride == 0 and all(
        ok[i + k] and not hard[i + k] and seq[i + k] == seq[i] and ticks[i + k] == ticks[i] + k for k in range(Fn))]
    sk, sg = [], []
    for b in range(0, len(legal), a.batch):
        bi = legal[b:b + a.batch]
        sk.append(wan_encode(vae, torch.stack([kframes(range(i, i + Fn)) for i in bi]).to(dev)).to(torch.bfloat16).cpu())
        sg.append(wan_encode(vae, torch.stack([gframes(range(i, i + Fn)) for i in bi]).to(dev)).to(torch.bfloat16).cpu())
        KEY.clear()
    out = {"scene": sc, "ticks": E["ticks"].clone(), "ok": E["ok"].clone(), "hard": E["hard"].clone(), "seq": E["seq"].clone(),
           "mask_lr": mask_lr.half(), "streams": streams, "starts": torch.tensor(legal, dtype=torch.long),
           "start_key": torch.cat(sk) if sk else torch.zeros(0, 16, a.clip_latents, 32, 64, dtype=torch.bfloat16),
           "start_guide": torch.cat(sg) if sg else torch.zeros(0, 16, a.clip_latents, 32, 64, dtype=torch.bfloat16),
           "meta": {"vae": a.wan, "vae_sha": VAE_SHA, "guide_file_sha256": GSHA, "guides": a.guides, "keys": a.keys, "code_sha": CODE_SHA,
                    "phases": a.phases, "start_stride": a.start_stride, "frames": Fn, "guide_ticks_found": len(have), "ticks": n,
                    "latent": "posterior mode, (z - latents_mean) / latents_std"}}
    f = f"{a.out}/{sc}.pt"
    torch.save(out, f + ".tmp"); os.replace(f + ".tmp", f)
    json.dump({"scene": sc, "sha256": sha_file(f), "streams": len(streams), "stream_latents": sum(s["key"].shape[1] for s in streams),
               "starts": len(legal), "guide_ticks_found": len(have), "ticks": n, "vae_sha": VAE_SHA, "code_sha": CODE_SHA},
              open(f"{a.out}/{sc}.json", "w"))
    print(sc, n, "ticks", len(segs), "segments", len(streams), "streams", len(legal), "starts", f"guides {len(have)}/{n - sum(hard)}",
          f"{time.time() - t0:.0f}s", flush=True)
print("WAN_CACHE_DONE", a.shard)
