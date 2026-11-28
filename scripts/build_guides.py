"""PLAN-video guides: render the camera-only chase-view guide (mrv/guide.py) for every selected tick of the given drives.

  python scripts/build_guides.py --drives train48,dev --shard 0/4                     # -> data/guides_v1/<scene>.pt
  python scripts/build_guides.py --drives train48,dev --shard 0/4 --lidar_ceiling     # -> data/guides_v1_lidar/<scene>.pt

Ticks come from data/temporal_manifest.json (6 camera files per tick, t_us) aligned with data/enc_tpv10/<scene>.pt
(ticks / ok / hard / seq). Hard ticks (camera > 100 ms off or a repeated bundle) are not rendered (built = False, zeros).
--lidar_ceiling (S0c ceiling arm; TRAINING / ANALYSIS ONLY, never a run-time input): the LiDAR sweep nearest the tick, moved
into each camera through both ego poses, nearest return per depth pixel, densified 4 rings (nearest depth wins); those pixels
replace the estimated depth, the rest keep the estimate (--lidar_fill est) or are dropped (--lidar_fill none).
Production default: flat-ground UNDERLAY (mrv/guide.py; Fable C9 / Astra S1 review): depth points over the road-plane warp;
src uint8 [n,1,256,512] = 1 depth / 2 ground prior / 0 nothing (--no_underlay = depth points only, a different cache version).
A drive is written only when EVERY non-hard selected tick was rendered (else the job fails and nothing is written); per-tick
coverage [n,3] = shares of the non-own-car pixels that are valid / from depth / from the ground prior (0 for hard ticks);
ticks with valid coverage < --min_coverage are listed in the sidecar and printed as a warning.
File per drive: ticks, built, ok, hard, seq, rgb uint8 [n,3,256,512], mask bool [n,1,256,512], src, coverage, invd uint8 [n,1,256,512]
(= clamp(510 / z, 0, 255), z in m), cam2tick float32 [n,6,4,4], K float32 [n,6,3,3] (1600x900 px), tick_pose float64 [n,4,4]
(world <- tick), own bool [256,512], provenance (depth model id, scale file sha, code sha, ...), timing (ms p50 / p95).
A sidecar <scene>.json holds the provenance; an existing drive with different provenance is an error unless --overwrite.
"""
import argparse, json, os, socket, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes

from mrv import guide as G

p = argparse.ArgumentParser()
p.add_argument("--drives", default="train48,dev"); p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--unseal", action="store_true", help="allow the sealed decision / final drives")
p.add_argument("--manifest", default="data/temporal_manifest.json"); p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--dataroot", default="data/nuscenes_full"); p.add_argument("--version", default="v1.0-trainval")
p.add_argument("--out", default="", help="default data/guides_v1 (data/guides_v1_lidar with --lidar_ceiling)")
p.add_argument("--scale_file", default="data/guides_v1/depth_scale.json")
p.add_argument("--depth_model", default=G.PRIMARY); p.add_argument("--no_fallback", action="store_true")
p.add_argument("--res", default="294x518", help="depth input HxW, multiples of 14 (S0b timed 294x518)")
p.add_argument("--splat", default="zbuf", choices=["zbuf", "soft"]); p.add_argument("--rmax", type=int, default=2)
p.add_argument("--chase_dz", type=float, default=0.0, help="chase camera height correction (m); 0 = 2.4 m as rendered")
p.add_argument("--tick_pose", default="keyframe_interp", choices=["keyframe_interp", "sweep"])
p.add_argument("--every", type=int, default=1, help="render ticks with tick %% every == 0")
p.add_argument("--max_ticks", type=int, default=0, help="smoke test: at most this many ticks per drive")
p.add_argument("--shard", default="0/1"); p.add_argument("--workers", type=int, default=4)
p.add_argument("--lidar_ceiling", action="store_true"); p.add_argument("--lidar_fill", default="est", choices=["est", "none"])
p.add_argument("--densify", type=int, default=4)
p.add_argument("--no_underlay", action="store_true", help="depth points only (no flat-ground underlay); NOT the production guide")
p.add_argument("--min_coverage", type=float, default=0.05, help="warn / list ticks whose valid coverage is below this")
p.add_argument("--overwrite", action="store_true"); p.add_argument("--time_log", default="")
a = p.parse_args()
t_start = time.time()
out_dir = a.out or ("data/guides_v1_lidar" if a.lidar_ceiling else "data/guides_v1")
os.makedirs(out_dir, exist_ok=True)
res = tuple(int(x) for x in a.res.split("x"))
drives = G.drive_list(a.drives, a.lists, unseal=a.unseal)
si, sn = map(int, a.shard.split("/")); mine = drives[si::sn]
print(f"drives {len(drives)}, this shard {len(mine)} -> {out_dir}", flush=True)
man = json.load(open(a.manifest))
nusc = NuScenes(a.version, a.dataroot, verbose=False)
R = G.GuideRenderer(scale_file=a.scale_file, res=res, splat=a.splat, rmax=a.rmax, chase_dz=a.chase_dz,
                    lidar_fill=a.lidar_fill, model_id=a.depth_model, allow_fallback=not a.no_fallback, underlay=not a.no_underlay)
if R.scales is None:
    raise SystemExit("no scale file: run scripts/fit_depth_scale.py first (the per-camera scale is part of the recipe)")
PROV = R.provenance()
PROV.update(lidar_ceiling=a.lidar_ceiling, lidar_fill=a.lidar_fill if a.lidar_ceiling else None,
            densify=a.densify if a.lidar_ceiling else None, tick_pose=a.tick_pose, every=a.every,
            code_sha=G.code_sha(os.path.abspath(__file__), os.path.abspath(G.__file__)),
            note="LiDAR ceiling: training / analysis only" if a.lidar_ceiling else "camera-only")
KEYS = ("depth_model_id", "scale_sha", "code_sha", "depth_res_hw", "splat", "rmax", "chase_dz", "lidar_ceiling", "lidar_fill",
        "densify", "tick_pose", "every", "underlay")


class Views(Dataset):
    def __init__(s, jobs):
        s.jobs = jobs

    def __len__(s):
        return len(s.jobs)

    def __getitem__(s, i):
        files, lfile = s.jobs[i]
        v = G.load_views(a.dataroot, files, (res[1], res[0]))
        pts = None
        if lfile is not None:
            from nuscenes.utils.data_classes import LidarPointCloud
            pts = LidarPointCloud.from_file(f"{a.dataroot}/{lfile}").points[:3].T.astype(np.float64)
        return i, v, pts


all_ms = []
for sc in mine:
    side = f"{out_dir}/{sc}.json"
    if os.path.exists(f"{out_dir}/{sc}.pt") and os.path.exists(side) and not a.overwrite:
        old = json.load(open(side))
        if not old.get("complete"):
            raise SystemExit(f"{sc}: existing sidecar is not marked complete (use --overwrite)")
        diff = {k: (old.get(k), PROV.get(k)) for k in KEYS if old.get(k) != PROV.get(k)}
        if diff:
            raise SystemExit(f"{sc}: existing guide has different provenance {diff} (use --overwrite or another --out)")
        print("skip", sc, flush=True); continue
    E = torch.load(f"{a.enc}/{sc}.pt", map_location="cpu", weights_only=False)
    rows = man[sc]
    assert [r["tick"] for r in rows] == E["ticks"].tolist(), f"{sc}: manifest / enc tick mismatch"
    hard = E["hard"].tolist()
    sel = [k for k in range(len(rows)) if rows[k]["tick"] % a.every == 0]
    if a.max_ticks:
        sel = sel[: a.max_ticks]
    n = len(sel)
    idx = G.SceneIndex(nusc, sc)
    rgb = torch.zeros(n, 3, G.GH, G.GW, dtype=torch.uint8); mask = torch.zeros(n, 1, G.GH, G.GW, dtype=torch.bool)
    invd = torch.zeros(n, 1, G.GH, G.GW, dtype=torch.uint8); built = torch.zeros(n, dtype=torch.bool)
    src = torch.zeros(n, 1, G.GH, G.GW, dtype=torch.uint8); cover = torch.zeros(n, 3)
    c2t_all = torch.zeros(n, 6, 4, 4); K_all = torch.zeros(n, 6, 3, 3); tp_all = torch.zeros(n, 4, 4, dtype=torch.float64)
    geo, jobs, jpos = {}, [], []
    for q, k in enumerate(sel):
        r = rows[k]
        Tw = idx.tick_pose(r["tick"], a.tick_pose, r["t_us"])
        K, c2t, c2e, c2w = idx.cam_geom(r["files"], Tw)
        c2t_all[q] = torch.from_numpy(c2t).float(); K_all[q] = torch.from_numpy(K).float(); tp_all[q] = torch.from_numpy(Tw)
        if hard[k]:
            continue
        lsd = idx.lidar_at(r["t_us"]) if a.lidar_ceiling else None
        geo[len(jobs)] = (q, K, c2t, c2e, c2w, lsd)
        jobs.append((r["files"], lsd["filename"] if lsd is not None else None))
    dl = DataLoader(Views(jobs), batch_size=None, shuffle=False, num_workers=a.workers, prefetch_factor=4 if a.workers else None)
    t_wait = time.perf_counter(); ms = []
    for i, views, pts in dl:
        wait = (time.perf_counter() - t_wait) * 1000
        q, K, c2t, c2e, c2w, lsd = geo[i]
        override = None
        if a.lidar_ceiling:
            tl = time.perf_counter()
            Tl = G.sensor_to_world(nusc, lsd)
            pts = pts.numpy() if torch.is_tensor(pts) else pts
            Pw = pts @ Tl[:3, :3].T + Tl[:3, 3]
            D = G.densify(G.lidar_depth(Pw, c2w, K, res, device=R.device), a.densify)
            override = torch.where(torch.isfinite(D), D, torch.full_like(D, float("nan")))
            lid_ms = (time.perf_counter() - tl) * 1000
        o = R.render(views, K, c2t, c2e, depth_override=override)
        rgb[q] = o["rgb"].cpu(); mask[q] = o["mask"].cpu(); src[q] = o["src"].cpu()
        cover[q] = torch.tensor([o["coverage"]["valid"], o["coverage"]["depth"], o["coverage"]["prior"]])
        invd[q] = (o["invd"] * 510.0).clamp(0, 255).round().to(torch.uint8).cpu()  # 0 where no guide pixel
        built[q] = True
        m_ = dict(o["ms"], load_wait=wait)
        if a.lidar_ceiling:
            m_["lidar"] = lid_ms
        ms.append(m_); t_wait = time.perf_counter()
    need = torch.tensor([not hard[k] for k in sel])
    if not torch.equal(built, need):  # refuse partial builds: every non-hard selected tick must have been rendered
        miss = [rows[sel[q]]["tick"] for q in range(n) if bool(need[q]) and not bool(built[q])]
        raise SystemExit(f"{sc}: {len(miss)} non-hard ticks not rendered (e.g. {miss[:5]}); nothing written")
    low = [int(rows[sel[q]]["tick"]) for q in range(n) if bool(need[q]) and float(cover[q, 0]) < a.min_coverage]
    if low:
        print(f"WARNING {sc}: {len(low)} ticks with valid coverage < {a.min_coverage}: {low[:10]}", flush=True)
    all_ms += ms
    summ = {k: {"p50": float(np.percentile([m[k] for m in ms], 50)), "p95": float(np.percentile([m[k] for m in ms], 95))}
            for k in (ms[0] if ms else {})}
    out = {"scene": sc, "ticks": torch.tensor([rows[k]["tick"] for k in sel]), "built": built,
           "ok": E["ok"][sel].clone(), "hard": E["hard"][sel].clone(), "seq": E["seq"][sel].clone(),
           "rgb": rgb, "mask": mask, "src": src, "coverage": cover, "invd": invd, "cam2tick": c2t_all, "K": K_all, "tick_pose": tp_all,
           "own": torch.from_numpy(G.OWN_NP.copy()), "provenance": PROV, "timing": summ, "gpu": torch.cuda.get_device_name()}
    tmp = f"{out_dir}/{sc}.{os.getpid()}.tmp"
    torch.save(out, tmp); os.replace(tmp, f"{out_dir}/{sc}.pt")
    cm = cover[need].mean(0).tolist() if bool(need.any()) else [0.0, 0.0, 0.0]
    json.dump({**PROV, "scene": sc, "ticks": n, "built": int(built.sum()), "non_hard": int(need.sum()), "complete": True,
               "coverage_mean": {"valid": cm[0], "depth": cm[1], "prior": cm[2]}, "low_coverage_ticks": low, "timing": summ},
              open(side + ".tmp", "w"), indent=1)
    os.replace(side + ".tmp", side)
    print(sc, f"{int(built.sum())}/{n} ticks built", {k: round(v["p50"], 1) for k, v in summ.items()},
          "coverage valid / depth / prior", [round(v, 3) for v in cm], flush=True)
if all_ms:
    tot = {k: {"p50": round(float(np.percentile([m[k] for m in all_ms], 50)), 1),
               "p95": round(float(np.percentile([m[k] for m in all_ms], 95)), 1)} for k in all_ms[0]}
    print("GUIDE_TIMING", json.dumps({"gpu": torch.cuda.get_device_name(), "ticks": len(all_ms), "ms": tot}), flush=True)
print("BUILD_GUIDES_DONE", len(mine), "drives", round(time.time() - t_start), "s", flush=True)
if a.time_log:
    os.makedirs(os.path.dirname(a.time_log) or ".", exist_ok=True)
    with open(a.time_log, "a") as f:
        f.write(json.dumps({"stage": "build_guides" + ("_lidar" if a.lidar_ceiling else ""), "shard": a.shard,
                            "seconds": round(time.time() - t_start, 1), "host": socket.gethostname(),
                            "gpu": torch.cuda.get_device_name()}) + "\n")
