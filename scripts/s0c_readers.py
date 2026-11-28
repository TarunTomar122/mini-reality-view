"""PLAN-video S0c (council/PLAN-video.md "SIGNATURE ROUND RESULT"; design council/pivot-cond-r1-fable.md section 4): which input
carries car placement better, the warped chase-view guide or the raw-view tokens? No video model. <= 5 aggregate 2080 Ti-hours
including caching and scoring (every stage appends its wall time to <out>/gpu_time.jsonl; the report sums it).

Readers (matched: parameters within 1 %, same updates / batch / lr schedule / training ticks / seed; 48 train drives =
mrv.guide.drive_list("train48"), every --train_every-th accepted tick), each predicting the KEY's car mask, road mask and car
instance boundaries at 256x512:
  guide   pixel-aligned conv encoder (U-Net) on [guide RGB, depth-source mask, ground-prior mask (flat-ground underlay),
          inverse depth, own-car mask, target-ray Fourier map]
          + residual blocks + decoder                                          (data/guides_v1, camera-only)
  lidar   same as guide on the LiDAR-depth guide                               (data/guides_v1_lidar; ceiling, analysis only)
  tokens  same encoder on the target-ray map only (no scene input) -> 2048 target-ray queries (32x64) that cross-attend the
          full-resolution (unpooled 32x18 per view at 448x252) frozen DINOv2-small tokens of the 6 raw views + target-relative
          Plücker rays of every source patch (in the chase-camera frame) -> same decoder
  hybrid  encoder on guide + ray map, then the same query / cross-attention stack over the tokens (report only)
Key masks: the signed scorer's own Mask2Former function key_masks() (scripts/temporal_metrics.py imported unchanged; its key
cache). Instances from a reader: connected components of car & ~boundary, boundary pixels re-attached (<= 4 px), the scorer's
filter (own hull removed, >= 8 rows).
Evaluation on the 6 dev drives, every accepted tick: instance car F1 with the imported signed Hungarian 0.3-IoU matcher
(per drive, equal drive weights), road IoU, car-mask jitter (1 - IoU of the predicted car union between consecutive ticks;
also minus the key's own change), near / far (key car bottom row on the road plane, 30 m ahead of the ego origin), paired
drive bootstrap (2,000 draws) and leave-one-drive-out. Controls: wrong-drive inputs (dev drive i scored with drive i+1's
inputs) must collapse each reader.
Pre-registered reading (PLAN-video, signed): a lead needs >= 0.03 car F1, paired drive interval > 0, positive in every LODO
and no road-IoU loss; otherwise both inputs go to S1 (S1 trains both with independent dropout anyway: Fable 06:30, S0c is not
an S1 launch gate). Unfinished or still-improving reader (final - 75 % checkpoint >= --improve_tol car F1) = UNRESOLVED ->
rerun BOTH readers with --updates 12000. LiDAR ceiling - guide >= 0.05 car F1 -> LiDAR-supervised depth fine-tune in scope.

  python scripts/s0c_readers.py --stage cache --shard 0/2          # key masks (Mask2Former) + 448x252 views, train48 + dev
  python scripts/s0c_readers.py --stage train --arm guide          # also lidar / tokens / hybrid (one 2080 Ti each)
  python scripts/s0c_readers.py --stage report                     # runs/s0c/report.json + sheet.jpg
"""
import argparse, io, json, math, os, socket, sys, time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--stage", required=True, choices=["cache", "train", "report"])
p.add_argument("--arm", default="", choices=["", "guide", "lidar", "tokens", "hybrid"])
p.add_argument("--drives", default="train48,dev", help="cache stage drive spec")
p.add_argument("--lists", default="data/temporal_lists.json"); p.add_argument("--manifest", default="data/temporal_manifest.json")
p.add_argument("--enc", default="data/enc_tpv10"); p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--dataroot", default="data/nuscenes_full")
p.add_argument("--guides", default="data/guides_v1"); p.add_argument("--guides_lidar", default="data/guides_v1_lidar")
p.add_argument("--cache", default="data/s0c_cache"); p.add_argument("--out", default="runs/s0c")
p.add_argument("--train_every", type=int, default=2, help="train ticks: accepted ticks with tick %% n == 0")
p.add_argument("--updates", type=int, default=6000); p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--width", type=int, default=192)
p.add_argument("--blocks", type=int, default=6); p.add_argument("--seed", type=int, default=0)
p.add_argument("--workers", type=int, default=6); p.add_argument("--evals", default="0.25,0.5,0.75,1.0")
p.add_argument("--improve_tol", type=float, default=0.01); p.add_argument("--shard", default="0/1")
p.add_argument("--budget_h", type=float, default=7.5, help="S0c hard ceiling (waiver A, 28 Nov 08:51; signed cap was 5.0)")
p.add_argument("--discarded_lines", default="", help="1-based gpu_time.jsonl line range of discarded work, e.g. 2-9 (pre-underlay guides)")
p.add_argument("--ceiling_hit", action="store_true", help="readers were cancelled at the waiver ceiling: no reruns (waiver A)")
p.add_argument("--sheet_ticks", type=int, default=4)
a = p.parse_args()
T_START = time.time()
sys.path.insert(0, os.getcwd())
from mrv import guide as G  # noqa: E402

DINO = "facebook/dinov2-small"
VIEW_WH = (448, 252)        # DINOv2 input (same size as data/enc_tpv10 tokens), 32 x 18 patches of 14 px, unpooled
TOK_HW = (18, 32)
Q_HW = (32, 64)             # target query grid (8 px cells)
NF = 4                      # Fourier frequencies
RAY_CH = 3 + 3 * 2 * NF     # target-ray map channels
PLK_CH = 6 + 6 * 2 * NF


def pack(m):
    return np.packbits(np.asarray(m, bool), axis=-1)


def unpack(x):
    return np.unpackbits(x, axis=-1)[..., :512].astype(bool)


def log_time(stage):
    import torch
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/gpu_time.jsonl", "a") as f:
        f.write(json.dumps({"stage": stage, "arm": a.arm, "shard": a.shard, "seconds": round(time.time() - T_START, 1),
                            "host": socket.gethostname(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}) + "\n")


# ======================================================================================================================
# stage cache: key masks (signed scorer's Mask2Former, via its key cache) + 448x252 JPEG views, per drive
# ======================================================================================================================
def stage_cache():
    import torch
    from PIL import Image
    from scipy import ndimage
    tm = G.import_signed_scorer(a.out)
    dev = set(G.drive_list("dev", a.lists))
    drives = G.drive_list(a.drives, a.lists)
    man = json.load(open(a.manifest))
    si, sn = map(int, a.shard.split("/")); os.makedirs(a.cache, exist_ok=True)
    for sc in drives[si::sn]:
        dst = f"{a.cache}/{sc}.pt"
        if os.path.exists(dst):
            print("skip", sc, flush=True); continue
        E = torch.load(f"{a.enc}/{sc}.pt", map_location="cpu", weights_only=False); rows = man[sc]
        assert [r["tick"] for r in rows] == E["ticks"].tolist(), sc
        acc = (E["ok"] & ~E["hard"]).tolist(); is_dev = sc in dev
        pos = [k for k in range(len(rows)) if acc[k] and (is_dev or rows[k]["tick"] % a.train_every == 0)]
        ticks, seq, car, road, bnd, inst, views = [], [], [], [], [], [], []
        for k in pos:
            tick = rows[k]["tick"]; kimg = G.key_image(a.keys, sc, tick)
            if kimg is None:
                continue
            cars, rd = tm["key_masks"](sc, tick, kimg)
            un = np.zeros((256, 512), bool); edge = np.zeros((256, 512), bool)
            for m in cars:
                un |= m; edge |= m & ~ndimage.binary_erosion(m)
            edge = ndimage.binary_dilation(edge) & un
            vb = []
            for f in rows[k]["files"]:
                im = Image.open(f"{a.dataroot}/{f}"); im.draft("RGB", VIEW_WH)
                im = im.convert("RGB").resize(VIEW_WH, Image.BICUBIC); buf = io.BytesIO(); im.save(buf, "JPEG", quality=92)
                vb.append(buf.getvalue())
            ticks.append(tick); seq.append(int(E["seq"][k])); car.append(pack(un)); road.append(pack(rd)); bnd.append(pack(edge))
            inst.append(np.stack([pack(m) for m in cars]) if cars else np.zeros((0, 256, 64), np.uint8)); views.append(vb)
        out = {"scene": sc, "split": "dev" if is_dev else "train", "ticks": ticks, "seq": seq, "car": np.stack(car),
               "road": np.stack(road), "bnd": np.stack(bnd), "inst": inst, "views": views, "scorer_sha": tm["SCORER_SHA"],
               "train_every": None if is_dev else a.train_every}
        torch.save(out, dst + ".tmp"); os.replace(dst + ".tmp", dst)
        print(sc, len(ticks), "ticks", flush=True)
    log_time("cache")
    print("S0C_CACHE_DONE", flush=True)


# ======================================================================================================================
# models
# ======================================================================================================================
def build_models():
    import torch, torch.nn as nn, torch.nn.functional as F

    def fourier(x):
        fr = (2.0 ** torch.arange(NF, device=x.device, dtype=x.dtype)) * math.pi
        xx = x[..., None] * fr
        return torch.cat([x, torch.sin(xx).flatten(-2), torch.cos(xx).flatten(-2)], -1)

    def conv(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.SiLU())

    class Encoder(nn.Module):
        def __init__(s, cin, C):
            super().__init__()
            s.s3 = nn.Sequential(conv(cin, 32), conv(32, 32))            # 256x512
            s.s2 = nn.Sequential(conv(32, 48, 2), conv(48, 48))          # 128x256
            s.s1 = nn.Sequential(conv(48, 64, 2), conv(64, 64))          # 64x128
            s.s0 = nn.Sequential(conv(64, C, 2), conv(C, C))             # 32x64

        def forward(s, x):
            f3 = s.s3(x); f2 = s.s2(f3); f1 = s.s1(f2); return f3, f2, f1, s.s0(f1)

    class Res(nn.Module):
        def __init__(s, C):
            super().__init__()
            s.b = nn.Sequential(nn.GroupNorm(8, C), nn.SiLU(), nn.Conv2d(C, C, 3, 1, 1), nn.GroupNorm(8, C), nn.SiLU(), nn.Conv2d(C, C, 3, 1, 1))

        def forward(s, x):
            return x + s.b(x)

    class PadMLP(nn.Module):  # 1x1 residual MLP used only to match parameter counts exactly
        def __init__(s, C, h):
            super().__init__()
            s.b = nn.Sequential(nn.Conv2d(C, h, 1), nn.SiLU(), nn.Conv2d(h, C, 1)) if h > 0 else None

        def forward(s, x):
            return x if s.b is None else x + s.b(x)

    class Dec(nn.Module):
        def __init__(s, C):
            super().__init__()
            s.u1 = nn.Sequential(conv(C + 64, 96), conv(96, 96))
            s.u2 = nn.Sequential(conv(96 + 48, 64), conv(64, 64))
            s.u3 = nn.Sequential(conv(64 + 32, 32), conv(32, 32))
            s.head = nn.Conv2d(32, 3, 1)

        def forward(s, x, f1, f2, f3):
            up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
            x = s.u1(torch.cat([up(x), f1], 1)); x = s.u2(torch.cat([up(x), f2], 1)); x = s.u3(torch.cat([up(x), f3], 1))
            return s.head(x)

    class Attn(nn.Module):
        def __init__(s, C, heads):
            super().__init__()
            s.h = heads; s.q = nn.Linear(C, C); s.kv = nn.Linear(C, 2 * C); s.o = nn.Linear(C, C)

        def forward(s, x, ctx):
            B, N, C = x.shape; M = ctx.shape[1]
            q = s.q(x).view(B, N, s.h, C // s.h).transpose(1, 2)
            k, v = s.kv(ctx).view(B, M, 2, s.h, C // s.h).permute(2, 0, 3, 1, 4)
            return s.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))

    class XBlock(nn.Module):
        def __init__(s, C, heads=6):
            super().__init__()
            s.n1, s.n2, s.n3, s.nk = nn.LayerNorm(C), nn.LayerNorm(C), nn.LayerNorm(C), nn.LayerNorm(C)
            s.x, s.s = Attn(C, heads), Attn(C, heads)
            s.m = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))

        def forward(s, q, kv):
            q = q + s.x(s.n1(q), s.nk(kv)); h = s.n2(q); q = q + s.s(h, h); return q + s.m(s.n3(q))

    class Reader(nn.Module):
        def __init__(s, arm, C, blocks, n_res=0, pad_h=0):
            super().__init__()
            s.arm = arm; s.use_g = arm != "tokens"; s.use_t = arm in ("tokens", "hybrid")
            s.enc = Encoder(RAY_CH + (7 if s.use_g else 0), C)
            if s.use_t:
                s.tok = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, C))
                s.plk = nn.Sequential(nn.Linear(PLK_CH, C), nn.SiLU(), nn.Linear(C, C))
                s.blocks = nn.ModuleList([XBlock(C) for _ in range(blocks)]); s.qn = nn.LayerNorm(C)
            else:
                s.res = nn.Sequential(*[Res(C) for _ in range(n_res)], PadMLP(C, pad_h))
            s.dec = Dec(C)
            d = G.chase_pixel_dirs(256, 512)
            s.register_buffer("ray", fourier(d).permute(2, 0, 1)[None].contiguous(), persistent=False)
            s.register_buffer("own", torch.from_numpy(G.OWN_NP.copy()).float()[None, None], persistent=False)

        def forward(s, g=None, tok=None, plk=None):
            B = (g if g is not None else tok).shape[0]
            x = s.ray.expand(B, -1, -1, -1)
            if s.use_g:
                gf = g.float() / 255.0
                x = torch.cat([gf[:, :3] * 2 - 1, gf[:, 3:6], s.own.expand(B, -1, -1, -1), x], 1)  # rgb, depth / prior masks, invd
            f3, f2, f1, x0 = s.enc(x)
            if s.use_t:
                q = x0.flatten(2).transpose(1, 2)
                kv = s.tok(tok) + s.plk(fourier(plk))
                for b in s.blocks:
                    q = b(q, kv)
                x0 = s.qn(q).transpose(1, 2).reshape(B, -1, *Q_HW)
            else:
                x0 = s.res(x0)
            return s.dec(x0, f1, f2, f3)

    count = lambda m: sum(p_.numel() for p_ in m.parameters())

    def make(arm, C=a.width, blocks=a.blocks):
        """guide / lidar readers are padded to the token reader's parameter count (residual blocks + a 1x1 MLP)"""
        if arm in ("tokens", "hybrid"):
            m = Reader(arm, C, blocks); return m, {"params": count(m), "blocks": blocks}
        target = count(Reader("tokens", C, blocks))
        p0 = count(Reader(arm, C, blocks, 0, 0)); per = count(Reader(arm, C, blocks, 1, 0)) - p0
        n = max(0, int((target - p0) // per)); rem = target - p0 - n * per
        h = max(0, int(round((rem - C) / (2 * C + 1))))
        m = Reader(arm, C, blocks, n, h); pc = count(m)
        assert abs(pc - target) / target < 0.01, (pc, target)
        return m, {"params": pc, "token_reader_params": target, "n_res": n, "pad_h": h}
    return make, fourier


def plucker(c2t, K):
    """source patch rays (target-relative): per 14-px DINOv2 patch centre of each 448x252 view, the ray in the CHASE camera
    frame as Plücker (d, o x d). c2t [B,6,4,4] cam -> tick ego, K [B,6,3,3] at 1600x900."""
    import torch
    dev_ = c2t.device
    v, u = torch.meshgrid((torch.arange(TOK_HW[0], device=dev_) + 0.5) * 14 * 900 / VIEW_WH[1],
                          (torch.arange(TOK_HW[1], device=dev_) + 0.5) * 14 * 1600 / VIEW_WH[0], indexing="ij")
    pix = torch.stack([u, v, torch.ones_like(u)], -1).view(-1, 3)                       # [576,3]
    rc = torch.einsum("bvij,nj->bvni", torch.linalg.inv(K.float()), pix)               # camera frame
    Tch = torch.as_tensor(np.linalg.inv(G.chase_cam_to_ego()), dtype=torch.float32, device=dev_)
    R = Tch[:3, :3] @ c2t[:, :, :3, :3].float()                                         # cam -> chase
    o = torch.einsum("ij,bvj->bvi", Tch[:3, :3], c2t[:, :, :3, 3].float()) + Tch[:3, 3]
    d = torch.einsum("bvij,bvnj->bvni", R, rc); d = d / d.norm(dim=-1, keepdim=True)
    m = torch.cross(o[:, :, None].expand_as(d), d, dim=-1)
    return torch.cat([d, m], -1).flatten(1, 2)                                          # [B, 6*576, 6]


# ======================================================================================================================
# stage train
# ======================================================================================================================
def load_split(names, arm):
    import torch
    gdir = a.guides_lidar if arm == "lidar" else a.guides
    use_g, use_t = arm != "tokens", arm in ("tokens", "hybrid")
    D = []
    for sc in names:
        C = torch.load(f"{a.cache}/{sc}.pt", weights_only=False)
        Gf = torch.load(f"{gdir}/{sc}.pt", weights_only=False)
        gp = {int(t): i for i, t in enumerate(Gf["ticks"].tolist())}
        missing = [t for t in C["ticks"] if t not in gp or not bool(Gf["built"][gp[t]])]
        if missing:
            raise SystemExit(f"{sc}: guide missing / not built for ticks {missing[:5]}... ({gdir})")
        r = torch.tensor([gp[t] for t in C["ticks"]])
        d = {"scene": sc, "ticks": C["ticks"], "seq": C["seq"], "car": C["car"], "road": C["road"], "bnd": C["bnd"],
             "c2t": Gf["cam2tick"][r].clone(), "K": Gf["K"][r].clone(), "prov": Gf["provenance"]}
        if use_g:
            if "src" not in Gf:
                raise SystemExit(f"{sc}: guide file has no source map (pre-underlay version); rebuild with scripts/build_guides.py")
            sr = Gf["src"][r]
            d["g"] = torch.cat([Gf["rgb"][r], (sr == 1).to(torch.uint8) * 255, (sr == 2).to(torch.uint8) * 255, Gf["invd"][r]], 1).contiguous()
        if use_t:
            d["views"] = C["views"]
        del Gf, C
        if D:  # one guide version per arm (no mixed caches)
            pk = ("depth_model_id", "scale_sha", "code_sha", "depth_res_hw", "splat", "rmax", "chase_dz", "lidar_ceiling",
                  "lidar_fill", "densify", "tick_pose", "underlay")
            diff = {k: (D[0]["prov"].get(k), d["prov"].get(k)) for k in pk if D[0]["prov"].get(k) != d["prov"].get(k)}
            if diff:
                raise SystemExit(f"{sc}: guide provenance differs from {D[0]['scene']}: {diff}")
        D.append(d)
    return D


def make_dataset(D, items, with_tgt, use_g, use_t):
    import torch
    from torch.utils.data import Dataset
    from PIL import Image

    class DS(Dataset):
        def __len__(s):
            return len(items)

        def __getitem__(s, i):
            di, j = items[i]; d = D[di]
            o = {"c2t": d["c2t"][j], "K": d["K"][j]}
            if with_tgt:
                o["tgt"] = torch.from_numpy(np.stack([unpack(d["car"][j]), unpack(d["road"][j]), unpack(d["bnd"][j])]))
            if use_g:
                o["g"] = d["g"][j]
            if use_t:
                o["v"] = torch.from_numpy(np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in d["views"][j]]))
            return o
    return DS()


def stage_train():
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    assert a.arm, "--arm required"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = f"{a.out}/{a.arm}"; os.makedirs(out, exist_ok=True)
    use_g, use_t = a.arm != "tokens", a.arm in ("tokens", "hybrid")
    make, _ = build_models()
    model, pinfo = make(a.arm); model = model.cuda()
    print(a.arm, pinfo, flush=True)
    dino = None
    if use_t:
        from transformers import AutoModel
        dino = AutoModel.from_pretrained(DINO, local_files_only=True).cuda().half().eval()
        for q in dino.parameters():
            q.requires_grad_(False)
    MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    tr = load_split(G.drive_list("train48", a.lists), a.arm); dv = load_split(G.drive_list("dev", a.lists), a.arm)
    for k_ in ("depth_model_id", "scale_sha", "code_sha", "splat", "chase_dz", "lidar_ceiling", "lidar_fill", "tick_pose", "underlay"):
        if tr[0]["prov"].get(k_) != dv[0]["prov"].get(k_):
            raise SystemExit(f"train / dev guides differ in {k_}: {tr[0]['prov'].get(k_)} vs {dv[0]['prov'].get(k_)}")
    items = [(di, j) for di, d in enumerate(tr) for j in range(len(d["ticks"]))]
    print(f"train ticks {len(items)} on {len(tr)} drives; dev ticks {sum(len(d['ticks']) for d in dv)}", flush=True)
    own = torch.from_numpy(G.OWN_NP.copy()).cuda()[None]; valid = (~own).float()

    def inputs(b):
        kw = {}
        if use_g:
            kw["g"] = b["g"].cuda(non_blocking=True)
        if use_t:
            v = b["v"].cuda(non_blocking=True)                                       # [B,6,252,448,3]
            B = v.shape[0]
            x = (v.flatten(0, 1).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD
            with torch.no_grad():
                tok = dino(pixel_values=x.half()).last_hidden_state[:, 1:]          # [B*6, 576, 384]
            kw["tok"] = tok.reshape(B, -1, 384)
            kw["plk"] = plucker(b["c2t"].cuda(), b["K"].cuda())
        return kw

    def loss_fn(lg, tgt):
        t = tgt.float(); n = valid.sum() * lg.shape[0]
        bce = lambda x, y, pw=None: (F.binary_cross_entropy_with_logits(x, y, reduction="none", pos_weight=pw) * valid).sum() / n
        pc = torch.sigmoid(lg[:, 0]) * valid
        dice = 1 - (2 * (pc * t[:, 0]).sum((1, 2)) + 1) / (pc.sum((1, 2)) + (t[:, 0] * valid).sum((1, 2)) + 1)
        return bce(lg[:, 0], t[:, 0]) + dice.mean() + 0.5 * bce(lg[:, 1], t[:, 1]) + 0.5 * bce(lg[:, 2], t[:, 2], torch.tensor(5.0, device=lg.device))

    @torch.no_grad()
    def evaluate(src_shift=0):
        """predictions on every dev tick; src_shift k: drive i is fed drive (i+k)'s inputs (wrong-drive control)"""
        model.eval(); res = {}
        for di, d in enumerate(dv):
            si_ = (di + src_shift) % len(dv); s_ = dv[si_]; n = len(d["ticks"])
            its = [(si_, j % len(s_["ticks"])) for j in range(n)]
            dl = DataLoader(make_dataset(dv, its, False, use_g, use_t), batch_size=a.batch, shuffle=False, num_workers=a.workers)
            car, road, bnd = [], [], []
            for b in dl:
                with torch.autocast("cuda", dtype=torch.float16):
                    pr = torch.sigmoid(model(**inputs(b)).float()) > 0.5
                pr = pr.cpu().numpy()
                car.append(pack(pr[:, 0])); road.append(pack(pr[:, 1])); bnd.append(pack(pr[:, 2]))
            res[d["scene"]] = {"ticks": d["ticks"], "src": s_["scene"], "car": np.concatenate(car), "road": np.concatenate(road), "bnd": np.concatenate(bnd)}
        model.train(); return res

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.99))
    warm = min(300, a.updates // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.updates)))))
    scaler = torch.cuda.amp.GradScaler()
    gen = torch.Generator(); gen.manual_seed(a.seed)
    dl = DataLoader(make_dataset(tr, items, True, use_g, use_t), batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=a.workers, generator=gen, persistent_workers=a.workers > 0, pin_memory=True)
    evals = sorted({max(1, int(round(float(f) * a.updates))) for f in a.evals.split(",")})
    step, t0, losses, curve = 0, time.time(), [], []
    model.train()
    while step < a.updates:
        for b in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(**inputs(b))
            loss = loss_fn(lg.float(), b["tgt"].cuda(non_blocking=True))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            step += 1; losses.append(float(loss))
            if step % 100 == 0:
                curve.append([step, float(np.mean(losses[-100:]))])
                print(f"{a.arm} step {step} loss {curve[-1][1]:.4f} {(time.time() - t0) / step:.3f} s/update", flush=True)
            if step in evals:
                te = time.time(); pr = evaluate()
                torch.save({"step": step, "pred": pr}, f"{out}/pred_{step}.pt")
                torch.save({"step": step, "model": model.state_dict(), "pinfo": pinfo}, f"{out}/ckpt.pt")
                print(f"{a.arm} eval at {step} ({time.time() - te:.0f} s)", flush=True)
            if step >= a.updates:
                break
    torch.save({"step": step, "pred": evaluate(src_shift=1)}, f"{out}/pred_wrong_{step}.pt")
    info = {"arm": a.arm, "params": pinfo, "updates": a.updates, "batch": a.batch, "lr": a.lr, "seed": a.seed, "width": a.width,
            "blocks": a.blocks, "train_every": a.train_every, "train_ticks": len(items), "train_drives": [d["scene"] for d in tr],
            "dev_drives": [d["scene"] for d in dv], "evals": evals, "loss_curve": curve, "s_per_update": (time.time() - t0) / max(1, step),
            "guide_provenance": tr[0]["prov"], "encoder": DINO if use_t else None, "finished": step >= a.updates,
            "seconds": round(time.time() - T_START, 1)}
    json.dump(info, open(f"{out}/train.json", "w"), indent=1, default=str)
    log_time("train")
    print("S0C_TRAIN_DONE", a.arm, json.dumps({k: info[k] for k in ("params", "train_ticks", "s_per_update")}), flush=True)


# ======================================================================================================================
# stage report
# ======================================================================================================================
def stage_report():
    import torch
    from PIL import Image, ImageDraw
    from scipy import ndimage
    tm = G.import_signed_scorer(a.out)
    iou_mat, hungarian, OWN = tm["iou_mat"], tm["hungarian"], tm["OWN"]
    dev = G.drive_list("dev", a.lists)
    C = {sc: torch.load(f"{a.cache}/{sc}.pt", weights_only=False) for sc in dev}

    def ground_x(row):
        """ego-frame x where the chase ray through (cropped) row meets the road (inf above the horizon)"""
        y = (row + 0.5 - G.K_CHASE[1, 2]) / G.FX
        dz = -G.SP - y * G.CP; dx = G.CP - y * G.SP
        return -G.CH_BACK + (G.CH_UP / -dz) * dx if dz < -1e-6 else float("inf")

    def near(m):
        return ground_x(np.where(m.any(1))[0].max()) < 30.0

    def instances(car, bnd):
        full = car & ~OWN; core = full & ~bnd
        lab, n = ndimage.label(core)
        if n == 0:
            return []
        dist, ind = ndimage.distance_transform_edt(lab == 0, return_indices=True)
        L = lab[ind[0], ind[1]]; L[~full | (dist > 4)] = 0
        out = []
        for i, sl in enumerate(ndimage.find_objects(L), 1):
            if sl is None:
                continue
            m = np.zeros_like(full); m[sl] = L[sl] == i
            rws = np.where(m.any(1))[0]
            if m.sum() >= 16 and rws.max() - rws.min() + 1 >= 8:
                out.append(m)
        return out

    def score(pred):
        per = {}
        for sc in dev:
            if sc not in pred:
                return None
            P, K_ = pred[sc], C[sc]; n = len(K_["ticks"])
            assert list(P["ticks"]) == list(K_["ticks"]), sc
            tp = fp = fn = 0; roads, jit, kjit = [], [], []; nf = {"near": [0, 0, 0], "far": [0, 0, 0]}
            prev = None
            for j in range(n):
                car, bnd, rd = unpack(P["car"][j]), unpack(P["bnd"][j]), unpack(P["road"][j]) & ~OWN
                inst = instances(car, bnd)
                kc = list(unpack(K_["inst"][j])) if len(K_["inst"][j]) else []
                kr = unpack(K_["road"][j]); ku = unpack(K_["car"][j])
                m = hungarian(iou_mat(inst, kc), 0.3)
                tp += len(m); fp += len(inst) - len(m); fn += len(kc) - len(m)
                for g_, km in enumerate(kc):
                    nf["near" if near(km) else "far"][0 if g_ in m else 2] += 1
                matched_r = set(m.values())
                for r_, im in enumerate(inst):
                    if r_ not in matched_r:
                        nf["near" if near(im) else "far"][1] += 1
                roads.append((kr & rd).sum() / max(1, (kr | rd).sum()))
                pu = np.zeros_like(ku)
                for im in inst:
                    pu |= im
                if prev is not None and K_["ticks"][j] == K_["ticks"][j - 1] + 1 and K_["seq"][j] == K_["seq"][j - 1]:
                    ch = lambda x, y: 1 - (x & y).sum() / (x | y).sum() if (x | y).any() else 0.0
                    jit.append(ch(pu, prev[0])); kjit.append(ch(ku, prev[1]))
                prev = (pu, ku)
            f1 = lambda t, f, n_: 2 * t / max(1, 2 * t + f + n_)
            per[sc] = {"car_f1": f1(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn, "road_iou": float(np.mean(roads)),
                       "jitter": float(np.mean(jit)) if jit else None, "key_jitter": float(np.mean(kjit)) if kjit else None,
                       "near_f1": f1(*nf["near"]), "far_f1": f1(*nf["far"]), "ticks": n}
        agg = {k: float(np.mean([per[s][k] for s in dev if per[s][k] is not None])) for k in ("car_f1", "road_iou", "jitter", "key_jitter", "near_f1", "far_f1")}
        T_ = {k: sum(per[s][k] for s in dev) for k in ("tp", "fp", "fn")}
        agg["car_f1_pooled"] = 2 * T_["tp"] / max(1, 2 * T_["tp"] + T_["fp"] + T_["fn"])
        agg["jitter_excess"] = agg["jitter"] - agg["key_jitter"]
        return {"per_drive": per, "mean": agg}

    arms = {}
    for arm in ("guide", "tokens", "hybrid", "lidar"):
        d = f"{a.out}/{arm}"; info = json.load(open(f"{d}/train.json")) if os.path.exists(f"{d}/train.json") else None
        steps = sorted(int(f[5:-3]) for f in os.listdir(d) if f.startswith("pred_") and f[5:-3].isdigit()) if os.path.isdir(d) else []
        R = {"info": {k: info[k] for k in ("params", "updates", "train_ticks", "s_per_update", "finished", "evals")} if info else None,
             "checkpoints": {}}
        for s_ in steps:
            R["checkpoints"][s_] = score(torch.load(f"{d}/pred_{s_}.pt", weights_only=False)["pred"])
        wf = [f for f in os.listdir(d) if f.startswith("pred_wrong_")] if os.path.isdir(d) else []
        R["wrong_drive"] = score(torch.load(f"{d}/{wf[0]}", weights_only=False)["pred"]) if wf else None
        upd = info["updates"] if info else None
        R["finished"] = bool(info and info["finished"] and upd in R["checkpoints"])
        if R["finished"]:
            ev = info["evals"]; s75 = min(ev, key=lambda e: abs(e - 0.75 * upd))
            f_fin = R["checkpoints"][upd]["mean"]["car_f1"]; f_75 = R["checkpoints"][s75]["mean"]["car_f1"]
            R["still_improving"] = bool(f_fin - f_75 >= a.improve_tol); R["final"] = R["checkpoints"][upd]
            R["f1_final_minus_75pct"] = f_fin - f_75
            if R["wrong_drive"]:
                R["wrong_drive_collapsed"] = bool(R["wrong_drive"]["mean"]["car_f1"] <= 0.5 * f_fin)
        arms[arm] = R

    def compare(x, y):
        X, Y = arms[x].get("final"), arms[y].get("final")
        if not X or not Y:
            return None
        f = np.array([X["per_drive"][s]["car_f1"] - Y["per_drive"][s]["car_f1"] for s in dev])
        r = np.array([X["per_drive"][s]["road_iou"] - Y["per_drive"][s]["road_iou"] for s in dev])
        jx = [X["per_drive"][s]["jitter"] for s in dev]; jy = [Y["per_drive"][s]["jitter"] for s in dev]
        rng = np.random.default_rng(0); boots = [f[rng.integers(0, len(f), len(f))].mean() for _ in range(2000)]
        lodo = [float(np.delete(f, i).mean()) for i in range(len(f))]
        out = {"car_f1_diff": float(f.mean()), "ci95": np.percentile(boots, [2.5, 97.5]).round(4).tolist(), "lodo": [round(v, 4) for v in lodo],
               "per_drive_diff": dict(zip(dev, f.round(4).tolist())), "road_iou_diff": float(r.mean()),
               "jitter_diff": float(np.mean(jx) - np.mean(jy)) if None not in jx + jy else None,
               "near_f1_diff": X["mean"]["near_f1"] - Y["mean"]["near_f1"], "far_f1_diff": X["mean"]["far_f1"] - Y["mean"]["far_f1"]}
        out["lead"] = bool(out["car_f1_diff"] >= 0.03 and out["ci95"][0] > 0 and min(lodo) > 0 and out["road_iou_diff"] >= 0)
        return out

    cmp_ = {"guide_vs_tokens": compare("guide", "tokens"), "tokens_vs_guide": compare("tokens", "guide"),
            "hybrid_vs_guide": compare("hybrid", "guide"), "hybrid_vs_tokens": compare("hybrid", "tokens"),
            "lidar_vs_guide": compare("lidar", "guide")}
    pg, pt = (arms[x]["info"]["params"]["params"] if arms[x]["info"] else None for x in ("guide", "tokens"))
    matched = bool(pg and pt and abs(pg - pt) / pt < 0.01)
    unres = [f"{x}: unfinished" for x in ("guide", "tokens") if not arms[x]["finished"]]
    unres += [f"{x}: still improving (final - 75% = {arms[x]['f1_final_minus_75pct']:+.3f} >= {a.improve_tol})" for x in ("guide", "tokens")
              if arms[x]["finished"] and arms[x]["still_improving"]]
    if not matched:
        unres.append(f"parameter counts not matched (guide {pg}, tokens {pt})")
    if unres:
        verdict = "UNRESOLVED: " + "; ".join(unres) + (" -> waiver A ceiling hit: NO reruns; both inputs go into S1 (already the S1 recipe)"
                                                       if a.ceiling_hit else " -> rerun BOTH readers with --updates 12000 (matched) before reading")
    elif cmp_["guide_vs_tokens"]["lead"]:
        verdict = "GUIDE LEADS (>= 0.03 car F1, paired interval > 0, every LODO > 0, no road-IoU loss); tokens stay the second path"
    elif cmp_["tokens_vs_guide"]["lead"]:
        verdict = "TOKENS LEAD (>= 0.03 car F1, paired interval > 0, every LODO > 0, no road-IoU loss)"
    else:
        verdict = "NO LEAD: both inputs go into S1 (independent input dropout; hybrid / guide-only / tokens-only compared at fixed noise on the same checkpoint)"
    lid = cmp_["lidar_vs_guide"]
    lidar_rule = None if lid is None else ("LiDAR ceiling - estimated guide = %+.3f car F1: %s" % (
        lid["car_f1_diff"], "LiDAR-supervised depth fine-tune IN SCOPE (>= 0.05)" if lid["car_f1_diff"] >= 0.05 else "below 0.05: estimated depth kept"))
    gpu_s = 0.0; tl = []
    if os.path.exists(f"{a.out}/gpu_time.jsonl"):
        tl = [json.loads(x) for x in open(f"{a.out}/gpu_time.jsonl") if x.strip()]
        gpu_s = sum(x["seconds"] for x in tl)
    gpu_h = (gpu_s + (time.time() - T_START)) / 3600
    disc_s = 0.0
    if a.discarded_lines:
        lo, hi = map(int, a.discarded_lines.split("-"))
        disc_s = sum(x["seconds"] for i, x in enumerate(tl, 1) if lo <= i <= hi)
    canc_s = sum(x["seconds"] for x in tl if x.get("cancelled"))
    hours = {"total_h": round(gpu_h, 3), "discarded_builder_error_h": round(disc_s / 3600, 3), "cancelled_at_ceiling_h": round(canc_s / 3600, 3),
             "clean_h": round(gpu_h - (disc_s + canc_s) / 3600, 3)}
    controls = {x: {"car_f1": arms[x]["final"]["mean"]["car_f1"] if arms[x].get("final") else None,
                    "wrong_drive_car_f1": arms[x]["wrong_drive"]["mean"]["car_f1"] if arms[x].get("wrong_drive") else None,
                    "collapsed (wrong <= 0.5 x right)": arms[x].get("wrong_drive_collapsed")} for x in arms}
    rep = {"verdict": verdict, "lidar_rule": lidar_rule, "comparisons": cmp_, "controls": controls,
           "params_matched": matched, "params": {x: arms[x]["info"]["params"] if arms[x]["info"] else None for x in arms},
           "arms": {x: {k: v for k, v in arms[x].items() if k not in ("checkpoints",)} | {
               "curve": {s_: (c["mean"] if c else None) for s_, c in arms[x]["checkpoints"].items()}} for x in arms},
           "gpu_hours_aggregate": round(gpu_h, 2), "gpu_budget_h": a.budget_h, "signed_cap_h": 5.0, "over_budget": gpu_h > a.budget_h, "hours": hours, "time_log": tl,
           "rules": {"lead": ">= 0.03 car F1 (mean of per-drive F1), paired drive bootstrap 95% interval > 0, every LODO > 0, road IoU diff >= 0",
                     "unresolved": f"unfinished or final - 75% checkpoint car F1 >= {a.improve_tol}", "lidar": "ceiling - guide >= 0.05 car F1",
                     "wrong_drive": "operational check: wrong-drive car F1 <= 0.5 x right-drive car F1", "dev_ticks": "every accepted dev tick",
                     "s1_gate": "S0c is not an S1 launch gate (Fable 06:30): S1 trains both inputs with dropout regardless"},
           "scorer_sha": tm["SCORER_SHA"], "dev_drives": dev}
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # comparison sheet: key | guide (est) | guide (LiDAR) | key masks | guide reader | token reader | hybrid reader
    preds = {}
    for x in ("guide", "tokens", "hybrid"):
        if arms[x].get("final"):
            preds[x] = torch.load(f"{a.out}/{x}/pred_{arms[x]['info']['updates']}.pt", weights_only=False)["pred"]

    def ov(key, cars, road, kcars):
        img = np.asarray(key).astype(np.float32) * 0.55
        img[road] = img[road] * 0.5 + np.array([40, 90, 255]) * 0.5
        for m in cars:
            img[m] = img[m] * 0.4 + np.array([255, 40, 40]) * 0.6
        for m in kcars:
            img[ndimage.binary_dilation(m) & ~m] = (0, 255, 0)
        return Image.fromarray(img.clip(0, 255).astype(np.uint8))
    cols = ["key", "guide (camera depth)", "guide (LiDAR)", "key masks", "guide reader", "token reader", "hybrid reader"]
    rows_ = []
    for sc in dev[: a.sheet_ticks]:
        K_ = C[sc]; j = len(K_["ticks"]) // 2; tick = K_["ticks"][j]
        key = G.key_image(a.keys, sc, tick); kc = list(unpack(K_["inst"][j])) if len(K_["inst"][j]) else []
        row = [key]
        for gd in (a.guides, a.guides_lidar):
            try:
                Gf = torch.load(f"{gd}/{sc}.pt", weights_only=False); gi = Gf["ticks"].tolist().index(tick)
                row.append(Image.fromarray(Gf["rgb"][gi].permute(1, 2, 0).numpy())); del Gf
            except Exception:  # noqa: BLE001
                row.append(Image.new("RGB", (512, 256)))
        row.append(ov(key, kc, unpack(K_["road"][j]), []))
        for x in ("guide", "tokens", "hybrid"):
            if x in preds:
                P = preds[x][sc]
                row.append(ov(key, instances(unpack(P["car"][j]), unpack(P["bnd"][j])), unpack(P["road"][j]) & ~OWN, kc))
            else:
                row.append(Image.new("RGB", (512, 256)))
        rows_.append((sc, tick, row))
    W_, H_ = 384, 192
    sheet = Image.new("RGB", (W_ * len(cols), 20 + H_ * len(rows_)), (20, 20, 20)); dr = ImageDraw.Draw(sheet)
    for c, t in enumerate(cols):
        dr.text((c * W_ + 4, 4), t, fill=(255, 255, 255))
    for r_, (sc, tick, row) in enumerate(rows_):
        for c, im in enumerate(row):
            sheet.paste(im.resize((W_, H_), Image.BILINEAR), (c * W_, 20 + r_ * H_))
        dr.text((4, 22 + r_ * H_), f"{sc} t{tick}", fill=(255, 255, 0))
    sheet.save(f"{a.out}/sheet.jpg", quality=90)
    log_time("report")
    print("S0C_REPORT", json.dumps({"verdict": verdict, "lidar_rule": lidar_rule, "gpu_hours": round(gpu_h, 2),
                                    "car_f1": {x: (arms[x]["final"]["mean"]["car_f1"] if arms[x].get("final") else None) for x in arms},
                                    "guide_vs_tokens": cmp_["guide_vs_tokens"]}), flush=True)


if __name__ == "__main__":
    {"cache": stage_cache, "train": stage_train, "report": stage_report}[a.stage]()
