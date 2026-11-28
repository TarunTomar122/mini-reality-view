"""Pinned-drive guide check (Fable 06:30 review (c1)/(d); runs BEFORE any guide fan-out or reader training, no training).

One dev drive (default scene-0396 if it is a dev drive, else the first dev drive), ~20 ticks (keyframe ticks, evenly spread
over the accepted ones). Three guides per tick, same renderer (mrv/guide.py):
  flat   no depth net: every pixel below the horizon on the road plane z = 0 (known camera height)
  est    Depth Anything V2-Small metric depth (primary checkpoint), per-camera frozen scale if the scale file exists
         (else unscaled, flagged)
  lidar  nuScenes LiDAR depth (nearest sweep, densified; the S0c ceiling arm definition; ANALYSIS ONLY)
Training-free placement score against the key (same Mask2Former segmentation as the signed scorer, imported unchanged):
  - photo cars: the scorer's Mask2Former finds car instances in each of the 6 camera photos; only those pixels' labels ride
    through the splat; per-view pieces of one car are merged (touching, depth within 15 %); instances get the scorer's
    filter (own hull removed, >= 8 rows). Matched to the key's car masks with the signed Hungarian 0.3-IoU matcher ->
    car F1 / precision / recall (instance level), union car-mask IoU.
  - coverage = share of key car pixels covered by valid guide pixels; spill = share of guide car pixels outside key cars.
Also: per-camera LiDAR / estimate depth ratio on these ticks (diagnostic only; dev drive, never used for fitting), timing.
Writes <out>/<scene>/sheet.jpg (all ticks, half size: key | flat | est | lidar, guide over a dimmed key, key cars outlined
green, guide-derived cars red), sheet_full.jpg (4 ticks, full size) and score.json; prints GUIDE_PREVIEW {...}.
  python scripts/guide_preview.py [--scene scene-0396] [--ticks 20]
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from nuscenes.nuscenes import NuScenes

from mrv import guide as G

p = argparse.ArgumentParser()
p.add_argument("--scene", default=""); p.add_argument("--ticks", type=int, default=20)
p.add_argument("--lists", default="data/temporal_lists.json"); p.add_argument("--manifest", default="data/temporal_manifest.json")
p.add_argument("--enc", default="data/enc_tpv10"); p.add_argument("--keys", default="data/sim/keys_tpv10")
p.add_argument("--dataroot", default="data/nuscenes_full"); p.add_argument("--version", default="v1.0-trainval")
p.add_argument("--scale_file", default="data/guides_v1/depth_scale.json")
p.add_argument("--depth_model", default=G.PRIMARY); p.add_argument("--res", default="294x518")
p.add_argument("--splat", default="zbuf", choices=["zbuf", "soft"]); p.add_argument("--chase_dz", type=float, default=0.0)
p.add_argument("--tick_pose", default="keyframe_interp", choices=["keyframe_interp", "sweep"])
p.add_argument("--lidar_fill", default="est", choices=["est", "none"]); p.add_argument("--densify", type=int, default=4)
p.add_argument("--out", default="runs/guide_preview")
a = p.parse_args()
res = tuple(int(x) for x in a.res.split("x"))
dev_list = G.drive_list("dev", a.lists)
sc = a.scene or ("scene-0396" if "scene-0396" in dev_list else dev_list[0])
if sc not in dev_list:
    print(f"WARNING {sc} is not a dev drive", flush=True)
od = f"{a.out}/{sc}"; os.makedirs(od, exist_ok=True)
tm = G.import_signed_scorer(a.out)
iou_mat, hungarian, OWN = tm["iou_mat"], tm["hungarian"], tm["OWN"]
proc, seg, CAR = tm["proc"], tm["seg"], tm["CAR"]
nusc = NuScenes(a.version, a.dataroot, verbose=False)
scale_file = a.scale_file if os.path.exists(a.scale_file) else None
R = G.GuideRenderer(scale_file=scale_file, res=res, splat=a.splat, chase_dz=a.chase_dz, lidar_fill=a.lidar_fill,
                    model_id=a.depth_model)
idx = G.SceneIndex(nusc, sc)
rows = json.load(open(a.manifest))[sc]
E = torch.load(f"{a.enc}/{sc}.pt", map_location="cpu", weights_only=False)
assert [r["tick"] for r in rows] == E["ticks"].tolist()
acc = (E["ok"] & ~E["hard"]).tolist()
cand = [k for k in range(len(rows)) if acc[k] and rows[k]["tick"] % 5 == 0 and G.key_image(a.keys, sc, rows[k]["tick"]) is not None]
pick = [cand[i] for i in np.unique(np.linspace(0, len(cand) - 1, min(a.ticks, len(cand))).round().astype(int))]
MODES = ["flat", "est", "lidar"]


@torch.no_grad()
def photo_cars(views):
    """scorer's Mask2Former on the 6 photos (at the depth resolution) -> instance label map [6,Hd,Wd] (-1 none), ids v*1000+i"""
    lab = np.full((6, res[0], res[1]), -1, np.int64)
    for v in range(6):
        im = Image.fromarray(views[v]).resize((1024, 576), Image.BICUBIC)
        inp = proc(images=im, return_tensors="pt").to("cuda")
        pan = proc.post_process_panoptic_segmentation(seg(**inp), threshold=0.5, target_sizes=[res])[0]
        sm = pan["segmentation"].cpu().numpy()
        for i, s_ in enumerate(pan["segments_info"]):
            if s_["label_id"] == CAR:
                lab[v][sm == s_["id"]] = v * 1000 + i
    return lab


def chase_instances(labmap, invd):
    """splatted label map -> merged, filtered car instance masks (list of [256,512] bool)"""
    L = labmap.cpu().numpy(); iv = invd[0].cpu().numpy()
    ids = [int(i) for i in np.unique(L) if i >= 0]
    masks = {i: L == i for i in ids}
    par = {i: i for i in ids}

    def root(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    dil = {i: ndimage.binary_dilation(m, iterations=2) for i, m in masks.items()}
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            i, j = ids[x], ids[y]
            if i // 1000 == j // 1000 or (dil[i] & masks[j]).sum() < 5:
                continue
            di, dj = np.median(iv[masks[i]]), np.median(iv[masks[j]])
            if di > 0 and dj > 0 and 0.85 <= di / dj <= 1 / 0.85:
                par[root(i)] = root(j)
    groups = {}
    for i in ids:
        groups.setdefault(root(i), []).append(i)
    out = []
    for g in groups.values():
        m = np.zeros_like(L, dtype=bool)
        for i in g:
            m |= masks[i]
        m = ndimage.binary_closing(m, iterations=2) & ~OWN
        rws = np.where(m.any(1))[0]
        if m.sum() >= 16 and len(rws) and rws.max() - rws.min() + 1 >= 8:
            out.append(m)
    return out


def outline(m):
    return m & ~ndimage.binary_erosion(m)


def overlay(key, rgb=None, vmask=None, kcars=(), gcars=()):
    base = np.asarray(key).astype(np.float32)
    img = base * 0.35
    if rgb is not None:
        vm = vmask[0].cpu().numpy(); g = rgb.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        img[vm] = g[vm] * 0.9 + base[vm] * 0.1
    else:
        img = base.copy()
    for m in kcars:
        img[ndimage.binary_dilation(outline(m))] = (0, 255, 0)
    for m in gcars:
        img[outline(m)] = (255, 40, 40)
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


stats = {m: {"tp": 0, "fp": 0, "fn": 0, "cov_num": 0, "cov_den": 0, "spill_num": 0, "spill_den": 0, "iou": [], "ms": [], "valid": []} for m in MODES}
ratio = {c: [] for c in G.CAMS}
panels = []
t0 = time.time()
for k in pick:
    r = rows[k]; tick = r["tick"]
    key = G.key_image(a.keys, sc, tick)
    kcars, kroad = tm["key_masks"](sc, tick, key)
    kunion = np.zeros((256, 512), bool)
    for m in kcars:
        kunion |= m
    Tw = idx.tick_pose(tick, a.tick_pose, r["t_us"])
    K, c2t, c2e, c2w = idx.cam_geom(r["files"], Tw)
    views = G.load_views(a.dataroot, r["files"], (res[1], res[0]))
    lab = photo_cars(views)
    lsd = idx.lidar_at(r["t_us"])
    Dl = G.lidar_depth(G.lidar_points_world(nusc, a.dataroot, lsd), c2w, K, res, device=R.device)
    row = [overlay(key, kcars=kcars)]
    for mode in MODES:
        if mode == "flat":
            o = R.render(views, K, c2t, c2e, mode="flat", labels=lab)
        elif mode == "est":
            o = R.render(views, K, c2t, c2e, labels=lab)
            zs = o["depth"]
            for v, c in enumerate(G.CAMS):
                mm = torch.isfinite(Dl[v]) & (Dl[v] >= 2) & (Dl[v] <= 60) & torch.isfinite(zs[v]) & (zs[v] > 0.1)
                if bool(mm.any()):
                    ratio[c].append(float((Dl[v][mm] / zs[v][mm]).median()))
        else:
            D = G.densify(Dl, a.densify)
            o = R.render(views, K, c2t, c2e, labels=lab, depth_override=torch.where(torch.isfinite(D), D, torch.full_like(D, float("nan"))))
        gcars = chase_instances(o["lab"], o["invd"])
        S = stats[mode]
        mt = hungarian(iou_mat(gcars, kcars), 0.3)
        S["tp"] += len(mt); S["fp"] += len(gcars) - len(mt); S["fn"] += len(kcars) - len(mt)
        vm = o["mask"][0].cpu().numpy()
        gun = np.zeros((256, 512), bool)
        for m in gcars:
            gun |= m
        S["cov_num"] += int((kunion & vm).sum()); S["cov_den"] += int(kunion.sum())
        S["spill_num"] += int((gun & ~kunion).sum()); S["spill_den"] += int(gun.sum())
        S["iou"].append(float((gun & kunion).sum() / max(1, (gun | kunion).sum())))
        S["ms"].append(o["ms"]["total"]); S["valid"].append(float(vm[~OWN].mean()))
        row.append(overlay(key, o["rgb"], o["mask"], kcars, gcars))
    panels.append((tick, row))
    print(sc, tick, {m: (stats[m]["tp"], stats[m]["fp"], stats[m]["fn"]) for m in MODES}, flush=True)

score = {"scene": sc, "ticks": [t for t, _ in panels], "scale_file": scale_file, "scale_sha": R.scale_sha,
         "depth": R.depth.info, "chase": G.CHASE, "chase_dz": a.chase_dz, "tick_pose": a.tick_pose, "splat": a.splat,
         "lidar_fill": a.lidar_fill, "scorer_sha": tm["SCORER_SHA"], "matcher": "signed Hungarian IoU >= 0.3 (imported)",
         "est_unscaled": scale_file is None,
         "lidar_over_est_ratio_median_per_cam (dev diagnostic only)": {c: (float(np.median(v)) if v else None) for c, v in ratio.items()}}
for m in MODES:
    S = stats[m]
    score[m] = {"car_f1": 2 * S["tp"] / max(1, 2 * S["tp"] + S["fp"] + S["fn"]), "precision": S["tp"] / max(1, S["tp"] + S["fp"]),
                "recall": S["tp"] / max(1, S["tp"] + S["fn"]), "tp": S["tp"], "fp": S["fp"], "fn": S["fn"],
                "key_car_coverage": S["cov_num"] / max(1, S["cov_den"]), "guide_car_spill": S["spill_num"] / max(1, S["spill_den"]),
                "car_union_iou_mean": float(np.mean(S["iou"])) if S["iou"] else None,
                "valid_share_mean": float(np.mean(S["valid"])) if S["valid"] else None,
                "render_ms_p50": float(np.median(S["ms"])) if S["ms"] else None}
score["lidar_minus_est_car_f1"] = score["lidar"]["car_f1"] - score["est"]["car_f1"]
score["reading"] = ("training-free and instance-level on photo-car pixels only; a pointer for S0c (whose 0.05 LiDAR rule is the "
                    "signed one), not a verdict")
score["seconds"] = round(time.time() - t0, 1)
json.dump(score, open(f"{od}/score.json", "w"), indent=1)

# sheets
labels = ["key (green = key cars)", "flat-ground warp", "estimated depth" + (" (UNSCALED)" if scale_file is None else ""), "LiDAR depth (ceiling)"]
hw, hh = 256, 128
sheet = Image.new("RGB", (hw * 4, 18 + hh * len(panels)), (20, 20, 20)); dr = ImageDraw.Draw(sheet)
for c, t in enumerate(labels):
    dr.text((c * hw + 4, 3), t, fill=(255, 255, 255))
for r_, (tick, row) in enumerate(panels):
    for c, im in enumerate(row):
        sheet.paste(im.resize((hw, hh), Image.BILINEAR), (c * hw, 18 + r_ * hh))
    dr.text((4, 18 + r_ * hh + 2), f"t{tick}", fill=(255, 255, 0))
sheet.save(f"{od}/sheet.jpg", quality=90)
sel = panels[:: max(1, len(panels) // 4)][:4]
full = Image.new("RGB", (512 * 4, 20 + 256 * len(sel)), (20, 20, 20)); dr = ImageDraw.Draw(full)
for c, t in enumerate(labels):
    dr.text((c * 512 + 6, 4), t, fill=(255, 255, 255))
for r_, (tick, row) in enumerate(sel):
    for c, im in enumerate(row):
        full.paste(im, (c * 512, 20 + r_ * 256))
    dr.text((6, 22 + r_ * 256), f"{sc} t{tick}", fill=(255, 255, 0))
full.save(f"{od}/sheet_full.jpg", quality=92)
print("GUIDE_PREVIEW", json.dumps({m: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in score[m].items()} for m in MODES}
                                  | {"scene": sc, "lidar_minus_est_car_f1": round(score["lidar_minus_est_car_f1"], 3),
                                     "est_unscaled": scale_file is None, "out": od}), flush=True)
