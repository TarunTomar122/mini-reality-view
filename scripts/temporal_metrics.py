"""Temporal evaluation, step 2 (council/PLAN-temporal.md + amendments): score rollouts against the 10 Hz game keys.

  python scripts/temporal_metrics.py --roll runs/troll/warm07_dev --cold runs/troll/indep_dev --out runs/troll/warm07_dev.json
  python scripts/temporal_metrics.py --compare runs/troll/warm07_dev.json runs/troll/indep_dev.json

Per (drive, seed) rollout: decode latents, segment render and key (Mask2Former, own car masked; key masks cached per
tick), only ACCEPTED ticks (ok and not hard) are scored.
- placement: car F1 (Hungarian IoU >= 0.3) and road IoU on every 5th accepted tick.
- key car tracks: key masks linked across adjacent accepted ticks (Hungarian IoU >= 0.5). Edge tracks: centred in the
  outer thirds. Per edge-track transition (t -> t+1): matched = render has a car with IoU >= 0.3 at both ticks;
  ID error = |cos-dist(DINO CLS of render crops t, t+1) - cos-dist(key crops t, t+1)| (crops = key box + 10%, 96x96);
  W = mean |render(t+1) - warp(render(t), key flow t->t+1)| on key car pixels at t+1 inside the outer 20% columns
  (RAFT-small flow on the keys; valid share reported). Stored per transition so comparisons can use SHARED transitions.
- edge recall per track tick; flips = changes of matched status; ghosts = unmatched render cars per tick.
- drift (rollouts >= 170 ticks): LPIPS and car F1 on ticks 150-169 vs the same ticks of the cold rollout (--cold);
  early window ticks 5-24 alongside.
- lag (rollouts made with --swap): ticks after the swap until 5-tick car F1 vs the new keys reaches 80% of the cold
  value of that other drive.
--compare A B: paired drive bootstrap (2,000 draws, seeds together) of ID / W on shared transitions, recall, flips,
car F1, road IoU, late LPIPS gap; prints point differences and 95% intervals.
"""
import argparse, glob, json, math, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--roll", default="")
p.add_argument("--cold", default="")
p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--out", default="")
p.add_argument("--compare", nargs=2, default=None)
p.add_argument("--shard", default="0/1", help="i/n: score every n-th rollout file (merge shards in --compare with a glob)")
p.add_argument("--lag_ref", default="", help="lag test: metrics json (or glob) of the cold / independent rollouts of the same split; the donor drive's car F1 (same seed) is the reference")
p.add_argument("--render_keys", action="store_true", help="metric check: score the key video itself as the render (must be best)")
a = p.parse_args()

if a.compare:
    def load_merged(pat):  # a json, or a glob of shard jsons
        out = {"drives": {}}
        for f in sorted(glob.glob(pat)):
            out["drives"].update(json.load(open(f))["drives"])
        return out
    A, B = (load_merged(f) for f in a.compare)
    runs = sorted(set(A["drives"]) & set(B["drives"]))  # "<scene>_s<seed>"
    seeds_of = lambda D, s_: {r.rsplit("_s", 1)[1] for r in D["drives"] if r.rsplit("_s", 1)[0] == s_}
    all_sc = sorted({r.rsplit("_s", 1)[0] for r in runs})
    # paired seeds: a drive enters only if both methods have the same full seed set for it
    scenes = [s_ for s_ in all_sc if seeds_of(A, s_) == seeds_of(B, s_)]
    unpaired = [s_ for s_ in all_sc if s_ not in scenes]
    by_scene = {s_: [r for r in runs if r.rsplit("_s", 1)[0] == s_] for s_ in scenes}
    PER = ("recall", "flips", "car_f1", "road_iou", "late_lpips_gap", "late_f1_gap", "early_lpips_gap", "lag_ticks")

    def per_drive(s_):
        """one value per drive (seeds pooled inside the drive): ID / W on SHARED transitions, others averaged over seeds"""
        rs = by_scene[s_]; v = {}
        for name in ("ID", "W"):
            ea, eb, elig = [], [], 0
            for d in rs:
                ta = {tuple(t[:4]): t[4] for t in A["drives"][d]["trans_" + name] if t[5] and t[4] == t[4]}
                tb = {tuple(t[:4]): t[4] for t in B["drives"][d]["trans_" + name] if t[5] and t[4] == t[4]}
                elig += len({tuple(t[:4]) for t in A["drives"][d]["trans_" + name]} | {tuple(t[:4]) for t in B["drives"][d]["trans_" + name]})
                for k in ta.keys() & tb.keys():
                    ea.append(ta[k]); eb.append(tb[k])
            v[name] = (np.mean(ea), np.mean(eb), len(ea), elig) if ea else (np.nan, np.nan, 0, elig)
        for name in PER:
            va = [A["drives"][d][name] for d in rs if A["drives"][d].get(name) is not None]
            vb = [B["drives"][d][name] for d in rs if B["drives"][d].get(name) is not None]
            v[name] = (np.mean(va), np.mean(vb), len(va), 0) if va and vb and len(va) == len(vb) else (np.nan, np.nan, 0, 0)
        return v
    PD = {s_: per_drive(s_) for s_ in scenes}
    no_shared = {name: [s_ for s_ in scenes if PD[s_][name][2] == 0 and PD[s_][name][3] > 0] for name in ("ID", "W")}

    def stats(pick_scenes):  # equal drive weights
        res = {}
        for name in ("ID", "W") + PER:
            xs = [PD[s_][name] for s_ in pick_scenes if np.isfinite(PD[s_][name][0]) and np.isfinite(PD[s_][name][1])]
            res[name] = (np.mean([x[0] for x in xs]) if xs else np.nan, np.mean([x[1] for x in xs]) if xs else np.nan,
                         sum(x[2] for x in xs), len(xs))
        return res
    full = stats(scenes)
    rng = np.random.default_rng(0); boots = {k: [] for k in full}
    for _ in range(2000):
        s = stats(list(rng.choice(scenes, len(scenes))))
        for k in s:
            boots[k].append(s[k][0] - s[k][1])
    out = {"A": a.compare[0], "B": a.compare[1], "drives": len(scenes), "runs": sum(len(v) for v in by_scene.values()),
           "unpaired_drives": unpaired, "weighting": "equal per drive; seeds pooled within drive",
           "no_shared_transition_drives": no_shared,
           "shared_per_drive": {name: {s_: PD[s_][name][2] for s_ in scenes} for name in ("ID", "W")}}
    for k, (x, y, n, nd) in full.items():
        bb = np.array(boots[k]); bb = bb[np.isfinite(bb)]
        out[k] = {"A": round(float(x), 4), "B": round(float(y), 4), "diff": round(float(x - y), 4), "n": int(n), "drives": int(nd),
                  "ci95": np.percentile(bb, [2.5, 97.5]).round(4).tolist() if len(bb) else None,
                  "rel": round(float((x - y) / y), 3) if y and np.isfinite(y) and y > 0 else None,
                  "can_pass": (not no_shared.get(k)) if k in ("ID", "W") else None}
    print("TCOMPARE", json.dumps(out)); raise SystemExit

import torch
import torch.nn.functional as F
import lpips
from PIL import Image, ImageDraw, ImageFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from diffusers import AutoencoderKL
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, AutoModel
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

dev = "cuda"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
name = "facebook/mask2former-swin-large-cityscapes-panoptic"
proc = AutoImageProcessor.from_pretrained(name); seg = Mask2FormerForUniversalSegmentation.from_pretrained(name).to(dev).eval()
lab = {v.lower(): int(k) for k, v in seg.config.id2label.items()}; CAR, ROAD = lab["car"], lab["road"]
dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(dev).eval()
lp = lpips.LPIPS(net="alex").to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
# own car hull (same as judge_pair.py)
W_, H_, CROP = 512, 288, 32
fpx = W_ / 2 / math.tan(math.radians(30)); cp, sp = math.cos(math.radians(8)), math.sin(math.radians(8)); pts = []
for x in (-1.05, 1.05):
    for y in (-2.5, 2.5):
        for z in (0.0, 1.6):
            Yf, Zu = y + 7.5, z - 2.4; dep = Yf * cp - Zu * sp; up = Yf * sp + Zu * cp
            pts.append((W_ / 2 + fpx * x / dep, H_ / 2 - fpx * up / dep - CROP))
pts = np.array(pts); hull = pts[ConvexHull(pts).vertices]
_m = Image.new("L", (512, 256), 0); ImageDraw.Draw(_m).polygon([tuple(q) for q in hull], fill=255)
OWN = np.asarray(_m.filter(ImageFilter.MaxFilter(9))) > 0
CACHE = "runs/judge/keycache10_m2f"; os.makedirs(CACHE, exist_ok=True)


@torch.no_grad()
def segment(img):
    inp = proc(images=img.resize((1024, 512), Image.BICUBIC), return_tensors="pt").to(dev)
    pan = proc.post_process_panoptic_segmentation(seg(**inp), threshold=0.5, target_sizes=[(256, 512)])[0]
    sm = pan["segmentation"].cpu().numpy(); cars = []; road = np.zeros((256, 512), bool)
    for s in pan["segments_info"]:
        m = sm == s["id"]
        if s["label_id"] == ROAD:
            road |= m
        elif s["label_id"] == CAR and (m & OWN).sum() <= 0.5 * m.sum():
            m = m & ~OWN; rows = np.where(m.any(1))[0]
            if len(rows) and rows.max() - rows.min() + 1 >= 8:
                cars.append(m)
    return cars, road & ~OWN


def key_masks(sc, tick, img):
    c = f"{CACHE}/{sc}_t{tick:04d}.npz"
    if os.path.exists(c):
        try:
            z = np.load(c)
            return (list(np.unpackbits(z["cars"], axis=-1)[..., :512].astype(bool)) if len(z["cars"]) else []), np.unpackbits(z["road"], axis=-1)[..., :512].astype(bool)
        except Exception:  # partial / corrupt cache entry (e.g. written concurrently): recompute
            pass
    cars, road = segment(img)
    tmp = f"{c[:-4]}.{os.getpid()}.tmp.npz"  # atomic write: several metric jobs share this cache
    np.savez_compressed(tmp, cars=np.packbits(np.stack(cars), axis=-1) if cars else np.zeros((0, 256, 64), np.uint8), road=np.packbits(road, axis=-1))
    os.replace(tmp, c)
    return cars, road


def iou_mat(P, G):
    if not P or not G:
        return np.zeros((len(P), len(G)))
    A = np.stack(P).reshape(len(P), -1).astype(np.float32); B = np.stack(G).reshape(len(G), -1).astype(np.float32)
    inter = A @ B.T; return inter / np.maximum(A.sum(1)[:, None] + B.sum(1)[None] - inter, 1)


def hungarian(M, thr):
    if M.size == 0:
        return {}
    r, c = linear_sum_assignment(-M); return {int(j): int(i) for i, j in zip(r, c) if M[i, j] >= thr}  # key -> render


@torch.no_grad()
def cls(img, box):
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
    b = (max(0, x0 - 0.1 * w), max(0, y0 - 0.1 * h), min(512, x1 + 0.1 * w), min(256, y1 + 0.1 * h))
    t = torch.from_numpy(np.asarray(img.crop(tuple(int(v) for v in b)).resize((96, 96)))).permute(2, 0, 1)[None].float().to(dev) / 255
    f = dino(pixel_values=(t - MEAN) / STD).last_hidden_state[:, 0]; return F.normalize(f, dim=-1)[0]


def to_img(lat):
    with torch.no_grad():
        x = vae.decode(lat[None].to(dev).float() / 0.18215).sample[0].clamp(-1, 1)
    return Image.fromarray(((x + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())


tens = lambda im: torch.from_numpy(np.asarray(im)).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
result = {"roll": a.roll, "cold": a.cold, "drives": {}}
_si, _sn = map(int, a.shard.split("/"))
for _k, fr in enumerate(sorted(glob.glob(f"{a.roll}/*_s*.pt"))):
    if _k % _sn != _si:
        continue
    b = os.path.basename(fr)[:-3]; sc = b.rsplit("_s", 1)[0]
    R = torch.load(fr); ticks = R["ticks"].tolist(); acc = (R["ok"] & ~R["hard"]).tolist()
    swap = R.get("swap_from", -1); other = R.get("other", "")
    keysc = lambda i: other if (swap >= 0 and i >= swap) else sc
    src_ticks = R["src_ticks"].tolist() if "src_ticks" in R else ticks  # after a swap: the donor drive's tick
    kt = lambda i: src_ticks[i]
    rimg, kimg, rseg, kseg = {}, {}, {}, {}
    for i in range(len(ticks)):
        if not acc[i]:
            continue
        rimg[i] = to_img(R["lat"][i]); ks = keysc(i)
        kp = f"{a.keys}/{ks}/t{kt(i):04d}.jpg"
        if not os.path.exists(kp):
            continue
        kimg[i] = Image.open(kp).convert("RGB").resize((512, 288), Image.BILINEAR).crop((0, 32, 512, 288))
        kseg[i] = key_masks(ks, kt(i), kimg[i])
        if a.render_keys:
            rimg[i] = kimg[i]; rseg[i] = kseg[i]
        else:
            rseg[i] = segment(rimg[i])
    idx = sorted(kimg)
    tp = fp = fn = 0; roads = []
    for n_, i in enumerate(idx):
        if n_ % 5:
            continue
        M = iou_mat(rseg[i][0], kseg[i][0]); m = hungarian(M, 0.3); tp += len(m); fp += len(rseg[i][0]) - len(m); fn += len(kseg[i][0]) - len(m)
        rk, rr = kseg[i][1], rseg[i][1]; roads.append((rk & rr).sum() / max(1, (rk | rr).sum()))
    # tracks on the keys + matches
    trans_ID, trans_W, rec, flips, ghosts = [], [], [], 0, []
    tid, nxt_id = {}, 0
    for n_, i in enumerate(idx):
        G = kseg[i][0]; M = iou_mat(rseg[i][0], G); m = hungarian(M, 0.3)
        ghosts.append(len(rseg[i][0]) - len(m))
        ids = {}
        if n_ and idx[n_ - 1] == i - 1:
            j = i - 1; link = hungarian(iou_mat(G, kseg[j][0]), 0.5)  # key t-1 index -> key t index
            prev = {v: k for k, v in link.items()}
            for g in range(len(G)):
                ids[g] = tid.get((j, prev[g])) if g in prev else None
        for g in range(len(G)):
            if ids.get(g) is None:
                ids[g] = nxt_id; nxt_id += 1
            tid[(i, g)] = ids[g]
        for g, mask in enumerate(G):
            xs = np.nonzero(mask.any(0))[0]; cx = xs.mean() if len(xs) else 256
            if 171 <= cx <= 341:
                continue
            rec.append(g in m)
            if n_ and idx[n_ - 1] == i - 1:
                j = i - 1; pg = [k for (jj, k), v in tid.items() if jj == j and v == ids[g]]
                if not pg:
                    continue
                pg = pg[0]; Mj = iou_mat(rseg[j][0], kseg[j][0]); mj = hungarian(Mj, 0.3)
                matched = (g in m) and (pg in mj)
                flips += int((g in m) != (pg in mj))
                bb = lambda mk: (np.nonzero(mk.any(0))[0].min(), np.nonzero(mk.any(1))[0].min(), np.nonzero(mk.any(0))[0].max() + 1, np.nonzero(mk.any(1))[0].max() + 1)
                kb_j, kb_i = bb(kseg[j][0][pg]), bb(mask)
                dk = 1 - float(cls(kimg[j], kb_j) @ cls(kimg[i], kb_i)); dr = 1 - float(cls(rimg[j], kb_j) @ cls(rimg[i], kb_i))
                trans_ID.append([sc, b, int(kt(i)), int(ids[g]), abs(dr - dk), bool(matched)])
                with torch.no_grad():
                    # backward warp: for each pixel x of frame i, flow i -> j on the KEYS gives where it was at j;
                    # sample render_j there. Valid = inside the image and forward-backward consistent.
                    ki, kj = tens(kimg[i]), tens(kimg[j])
                    fb = raft(ki, kj)[-1]; ff = raft(kj, ki)[-1]
                    gy, gx = torch.meshgrid(torch.arange(256, device=dev, dtype=torch.float32), torch.arange(512, device=dev, dtype=torch.float32), indexing="ij")
                    sx, sy = gx + fb[0, 0], gy + fb[0, 1]
                    grid = torch.stack([sx / 255.5 - 1, sy / 127.5 - 1], -1)[None]
                    wr = F.grid_sample(tens(rimg[j]), grid, align_corners=True)
                    ffw = F.grid_sample(ff, grid, align_corners=True)
                    cyc = (fb + ffw).pow(2).sum(1)[0]; mag = fb.pow(2).sum(1)[0] + ffw.pow(2).sum(1)[0]
                    valid = ((sx >= 0) & (sx <= 511) & (sy >= 0) & (sy <= 255) & (cyc < 0.01 * mag + 0.5)).cpu().numpy()
                    diff = (wr - tens(rimg[i])).abs().mean(1)[0].cpu().numpy() * 127.5
                cm = mask.copy(); cm[:, 102:410] = False; cmv = cm & valid
                trans_W.append([sc, b, int(kt(i)), int(ids[g]), float(diff[cmv].mean()) if cmv.any() else float("nan"),
                                bool(matched and cmv.any()), int(cmv.sum()), int(cm.sum())])
    d = {"car_f1": 2 * tp / max(1, 2 * tp + fp + fn), "road_iou": float(np.mean(roads)) if roads else None,
         "recall": float(np.mean(rec)) if rec else None, "flips": flips / max(1, len(trans_ID)), "ghosts": float(np.mean(ghosts)) if ghosts else None,
         "accepted": len(idx), "ticks": len(ticks), "trans_ID": trans_ID, "trans_W": trans_W}
    # drift vs cold rollout of the same drive / seed: only rollouts whose history chain runs unbroken through tick 169
    rst = R["restart"].tolist() if "restart" in R else [False] * len(ticks)
    contiguous = len(ticks) >= 170 and not any(rst[1:170])
    d["contiguous_170"] = contiguous
    if a.cold and contiguous and os.path.exists(f"{a.cold}/{b}.pt"):
        C = torch.load(f"{a.cold}/{b}.pt")
        def win(lo, hi):
            lr, lc, tr, tc = [], [], [0, 0, 0], [0, 0, 0]
            for i in range(lo, hi):
                if i in kimg:
                    K = tens(kimg[i]); ci = to_img(C["lat"][i])
                    with torch.no_grad():
                        lr.append(float(lp(tens(rimg[i]), K))); lc.append(float(lp(tens(ci), K)))
                    for acc_, cars in ((tr, rseg[i][0]), (tc, segment(ci)[0])):
                        mm = hungarian(iou_mat(cars, kseg[i][0]), 0.3)
                        acc_[0] += len(mm); acc_[1] += len(cars) - len(mm); acc_[2] += len(kseg[i][0]) - len(mm)
            if not lr:
                return None, None
            f1 = lambda q: 2 * q[0] / max(1, 2 * q[0] + q[1] + q[2])
            return float(np.mean(lr) - np.mean(lc)), float(f1(tr) - f1(tc))
        d["late_lpips_gap"], d["late_f1_gap"] = win(150, 170); d["early_lpips_gap"], d["early_f1_gap"] = win(5, 25)
    if swap >= 0:
        if not a.lag_ref:
            raise SystemExit("lag test needs --lag_ref (plan: 80% of the donor drive's cold car F1)")
        if "LAGREF" not in globals():
            LAGREF = {}
            for f_ in sorted(glob.glob(a.lag_ref)):
                LAGREF.update(json.load(open(f_))["drives"])
        ref_ = LAGREF.get(f"{other}_s{b.rsplit('_s', 1)[1]}", {}).get("car_f1")
        d["cold_ref_f1"] = ref_
        lag = None
        if not ref_:  # plan: nonzero reference only
            d["lag_ticks"] = None; result["drives"][b] = d; continue
        for i in range(swap, len(ticks) - 5):
            w = [j for j in range(i, i + 5) if j in kseg]
            t_ = sum(len(hungarian(iou_mat(rseg[j][0], kseg[j][0]), 0.3)) for j in w)
            n_ = sum(len(rseg[j][0]) + len(kseg[j][0]) for j in w)
            if n_ and 2 * t_ / n_ >= 0.8 * ref_:
                lag = i - swap; break
        d["lag_ticks"] = lag
    result["drives"][b] = d
    print(b, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if not k.startswith("trans")}, flush=True)
json.dump(result, open(a.out, "w"))
print("TMETRICS_DONE", len(result["drives"]))
