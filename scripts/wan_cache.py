"""PLAN-video S1 inputs, path (a) + targets: frozen-Wan-VAE latents of the game-render answer keys and of the chase-view guides.

Streams (Fable 06:35 b2 / c3; Astra item 3): a SEGMENT is a maximal run of non-hard ticks with consecutive tick ids
(mrv.wan_plan.segments): offset-only soft rejects (ok = False, new manifest seq) stay INSIDE the codec stream; only hard ticks
and tick gaps restart it. Each segment is encoded in 4 PHASES (start offsets 0-3, one VAE call per phase over the whole stream)
so the 4-tick grouping falls four ways; stream latent j >= 1 covers frames 4j-3..4j. Training eligibility is separate: the
trainer cuts 5-latent mid-stream clips (j0 >= 1) and 17-tick true starts (every --start_stride-th legal start; first latent =
single frame) whose ticks are ALL ok (mrv.wan_plan.mid_candidates / start_candidates); candidate counts are recorded here.
Keys: data/sim/keys_tpv10/<scene>/t%04d.jpg, 512x288 bilinear -> crop rows 32..288 (as temporal_metrics.py), [-1, 1]; a missing
key on a non-hard tick is an error (no grey frames in the video codec). Guides: data/guides_v1 (mrv/guide.py with the flat-
ground underlay), checked by mrv.wan_cond.check_guide (complete sidecar, camera-only, underlay, every non-hard tick built,
ticks / hard == enc, no LiDAR / preview paths) and ONE guide version (depth model, weights, scale sha, underlay, code ...) for
every drive of this run (--expect_version pins it across shards / train vs dev; the trainer re-checks all files).
Guide and key videos are encoded identically (posterior mode, latents_mean / latents_std normalisation). The guide source map
(1 depth / 2 ground prior) is kept as two shares per latent cell (src_lr), folded per latent by the trainer / rollout.

Per drive -> <out>/<scene>.pt: ticks / ok / hard / seq, src_lr fp16 [n, 2, 32, 64], streams [{pos0, phase, n_frames, key bf16
[16, L, 32, 64], guide}], starts, start_key / start_guide bf16 [S, 16, 5, 32, 64], meta; + <scene>.json sidecar {sha256 of the
.pt, guide / enc / keys / vae / code hashes, guide_version, candidate counts, complete}. An existing drive is skipped only if the
sidecar and all upstream hashes still match (else error; --rebuild re-encodes).
  python scripts/wan_cache.py --splits train,dev --out data/wan_cache_v1 --shard i/n
"""
import argparse, hashlib, json, os, time
import numpy as np
import torch
from PIL import Image
from mrv import guide as GD
from mrv import wan_cond as WC
from mrv import wan_plan as WP

p = argparse.ArgumentParser()
p.add_argument("--splits", default="train,dev")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--guides", default="data/guides_v1")
p.add_argument("--wan", default=WC.WAN_ID)
p.add_argument("--out", default="data/wan_cache_v1")
p.add_argument("--phases", type=int, default=4)
p.add_argument("--start_stride", type=int, default=4)
p.add_argument("--frames", type=int, default=17)
p.add_argument("--clip_latents", type=int, default=5)
p.add_argument("--batch", type=int, default=4, help="true-start clips per VAE call")
p.add_argument("--expect_version", default="", help="json file with the guide version every drive must have")
p.add_argument("--rebuild", action="store_true")
p.add_argument("--shard", default="0/1")
a = p.parse_args()
assert (a.frames - 1) % 4 == 0 and 1 + (a.frames - 1) // 4 == a.clip_latents
dev = "cuda"
os.makedirs(a.out, exist_ok=True)
# Neon 09:30 P0: the cache provenance hashes only the code that shapes the cached content (this script, mrv/wan_cond.py and the
# SOURCE of the four self-contained planning functions it calls), not the whole mrv/wan_plan.py, whose policy / ledger edits
# would otherwise invalidate every cache
import hashlib as _hl, inspect as _insp
CODE_SHA = _hl.sha1(open(__file__, "rb").read() + open("mrv/wan_cond.py", "rb").read()
                    + "".join(_insp.getsource(fn_) for fn_ in (WP.segments, WP.stream_specs, WP.start_candidates, WP.mid_candidates)).encode()).hexdigest()[:12]
drives = WP.check_split_membership(GD.drive_list(a.splits, a.lists), json.load(open(a.lists)))  # explicit ids / files too
si, sn = map(int, a.shard.split("/"))
mine = drives[si::sn]
VERSION = json.load(open(a.expect_version)) if a.expect_version else None


def key_path(sc, t):
    return f"{a.keys}/{sc}/t{t:04d}.jpg"


def keys_sha(sc, ticks, hard):
    h = hashlib.sha256()
    for t, hd in zip(ticks, hard):
        if hd:
            continue
        f = key_path(sc, int(t))
        if not os.path.exists(f):
            raise SystemExit(f"missing key {f} on a non-hard tick")
        h.update(open(f, "rb").read())
    return h.hexdigest()


from diffusers import AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained(a.wan, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
vae.requires_grad_(False)
VAE_SHA = WC.vae_sha(vae)


def upstream(sc, E):
    return {"guide_sha": WC.file_sha(f"{a.guides}/{sc}.pt"), "enc_sha": WC.file_sha(f"{a.enc}/{sc}.pt"),
            "keys_sha": keys_sha(sc, E["ticks"].tolist(), E["hard"].tolist()), "vae_sha": VAE_SHA, "code_sha": CODE_SHA,
            "recipe": {"phases": a.phases, "start_stride": a.start_stride, "frames": a.frames, "clip_latents": a.clip_latents}}


def key_img(sc, t):
    im = Image.open(key_path(sc, t)).convert("RGB").resize((512, 288), Image.BILINEAR).crop((0, 32, 512, 288))
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1)


tot = {"mid": 0, "start": 0, "drives_with_clips": 0}
for sc in mine:
    t0 = time.time()
    E = torch.load(f"{a.enc}/{sc}.pt", map_location="cpu", weights_only=False)
    ticks, ok, hard = E["ticks"].tolist(), E["ok"].tolist(), E["hard"].tolist()
    n = len(ticks)
    gf = f"{a.guides}/{sc}.pt"
    G = torch.load(gf, map_location="cpu", weights_only=False, mmap=True)
    VERSION = WC.check_guide(gf, G, json.load(open(f"{a.guides}/{sc}.json")), ticks, hard, sc, VERSION)
    up = upstream(sc, E)
    f = f"{a.out}/{sc}.pt"
    if os.path.exists(f) and not a.rebuild:
        sj = json.load(open(f"{a.out}/{sc}.json")) if os.path.exists(f"{a.out}/{sc}.json") else {}
        bad = [k for k, v in {**up, "guide_version": VERSION}.items() if sj.get(k) != v]
        if not sj.get("complete") or WC.file_sha(f) != sj.get("sha256") or bad:
            raise SystemExit(f"{f}: stale or corrupt cache ({bad[:4]}); rerun with --rebuild")
        print("valid, skip", sc, flush=True)
        tot["mid"] += sj["mid_candidates"]; tot["start"] += sj["start_candidates"]; tot["drives_with_clips"] += int(sj["mid_candidates"] + sj["start_candidates"] > 0)
        continue
    src_lr = WC.src_lowres(G["src"]).half()  # [n, 2, 32, 64]; hard ticks: zeros
    KEY = {}

    def kframes(idx):
        out = []
        for i in idx:
            if i not in KEY:
                KEY[i] = key_img(sc, ticks[i])
            out.append(KEY[i])
        return torch.stack(out, 1).float() / 127.5 - 1  # [3, F, 256, 512]

    def gframes(idx):
        return torch.stack([G["rgb"][i] for i in idx], 1).float() / 127.5 - 1

    segs = WP.segments(ticks, hard)
    specs = WP.stream_specs(segs, a.phases, a.clip_latents)
    streams = []
    for sp in specs:
        fr = list(range(sp["pos0"], sp["pos0"] + sp["n_frames"]))
        assert fr == segs[sp["seg"]][sp["phase"]:sp["phase"] + sp["n_frames"]]
        zk = WC.wan_encode(vae, kframes(fr)[None].to(dev))[0].to(torch.bfloat16).cpu()
        zg = WC.wan_encode(vae, gframes(fr)[None].to(dev))[0].to(torch.bfloat16).cpu()
        assert zk.shape[1] == sp["L"], (zk.shape, sp)
        streams.append({"pos0": sp["pos0"], "phase": sp["phase"], "n_frames": sp["n_frames"], "seg": sp["seg"], "key": zk, "guide": zg})
        if sp["phase"] == a.phases - 1:
            KEY.clear()
    legal = WP.start_candidates(segs, ok, hard, a.frames, a.start_stride)
    sk, sg = [], []
    for b in range(0, len(legal), a.batch):
        bi = legal[b:b + a.batch]
        sk.append(WC.wan_encode(vae, torch.stack([kframes(range(i, i + a.frames)) for i in bi]).to(dev)).to(torch.bfloat16).cpu())
        sg.append(WC.wan_encode(vae, torch.stack([gframes(range(i, i + a.frames)) for i in bi]).to(dev)).to(torch.bfloat16).cpu())
        KEY.clear()
    n_mid = len(WP.mid_candidates([{"pos0": s_["pos0"], "L": s_["key"].shape[1]} for s_ in streams], ok, hard, a.clip_latents))
    out = {"scene": sc, "ticks": E["ticks"].clone(), "ok": E["ok"].clone(), "hard": E["hard"].clone(), "seq": E["seq"].clone(),
           "src_lr": src_lr, "streams": streams, "starts": torch.tensor(legal, dtype=torch.long),
           "start_key": torch.cat(sk) if sk else torch.zeros(0, 16, a.clip_latents, 32, 64, dtype=torch.bfloat16),
           "start_guide": torch.cat(sg) if sg else torch.zeros(0, 16, a.clip_latents, 32, 64, dtype=torch.bfloat16),
           "meta": {**up, "guide_version": VERSION, "segments": len(segs), "latent": "posterior mode, (z - latents_mean) / latents_std"}}
    torch.save(out, f + ".tmp"); os.replace(f + ".tmp", f)
    json.dump({"scene": sc, "sha256": WC.file_sha(f), **up, "guide_version": VERSION, "segments": len(segs), "streams": len(streams),
               "mid_candidates": n_mid, "start_candidates": len(legal), "soft_rejects": sum(1 for i in range(n) if not ok[i] and not hard[i]),
               "ticks": n, "complete": True}, open(f"{a.out}/{sc}.json.tmp", "w"))
    os.replace(f"{a.out}/{sc}.json.tmp", f"{a.out}/{sc}.json")
    tot["mid"] += n_mid; tot["start"] += len(legal); tot["drives_with_clips"] += int(n_mid + len(legal) > 0)
    print(sc, n, "ticks", len(segs), "segments", len(streams), "streams", n_mid, "mid clips", len(legal), "starts", f"{time.time() - t0:.0f}s", flush=True)
json.dump(VERSION, open(f"{a.out}/guide_version.{si}.json", "w"), indent=1)
print("WAN_CACHE_DONE", a.shard, json.dumps(tot))
