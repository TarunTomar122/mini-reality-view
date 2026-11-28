"""Camera-only geometric guide for the chase view (council/PLAN-video.md: conditioning "HYBRID, GEOMETRY FIRST"; S0c / S1).

Per tick: the 6 nuScenes camera images -> Depth Anything V2-Small depth per view (one batch of 6) -> frozen per-camera scale
(fitted on TRAIN drives against LiDAR by scripts/fit_depth_scale.py; LiDAR is supervision only) -> unproject every depth pixel
with the exact nuScenes intrinsics / extrinsics and that image's own ego pose -> tick ego frame -> the answer-key chase camera
-> z-buffered splat (nearest wins, small footprint-sized squares) -> guide RGB [3,256,512] uint8 + validity mask [1,256,512]
+ source map [1,256,512] + inverse depth [1,256,512].
Production default (Fable C9, Astra S1 review item 2): a FLAT-GROUND UNDERLAY. The same pixels are also put on the road
plane z = 0 of their own ego frame (known camera height) and splatted under the depth points: depth wins wherever a depth
point lands, the ground prior fills only what no depth point reaches; src = 1 (depth) / 2 (ground prior) / 0 (nothing).

Chase camera = the camera that rendered the answer keys data/sim/keys_tpv10 (scripts/sim_render.py --view tpv: MetaDrive
camera parented to the agent origin at panda pos [0, -7.5, 2.4] (x right, y forward, z up), hpr [0, -8, 0], lens 60 deg HFOV
at 512x288; scripts/temporal_metrics.py then crops the top 32 rows -> 512x256). Same projection as mrv/carmap.py project()
(the judge's projection). Sim ego origin = logged nuScenes ego origin (rear axle, 0.0 m: scripts/sim_verify.py, PROGRESS.md).
Pinhole in the cropped image: fx = fy = 256 / tan(30 deg) = 443.405 px, cx = 256, cy = 144 - 32 = 112.
Tick pose: ScenarioNet replays nuScenes keyframes (2 Hz) linearly interpolated to 10 Hz, so the chase camera at tick t sits on
the linear interpolation of the LIDAR_TOP keyframe ego poses k = t // 5, k + 1 (yaw-only: MetaDrive is flat); "sweep" mode
uses the ego pose of the LiDAR sweep nearest to the tick time instead.
Own car: the keys show our own car; the signed scorer removes the fixed own-car hull (mrv/carmap.OWN, identical to the
scorer's OWN). The guide sets mask = 0, RGB = 0, inverse depth = 0 inside that hull and drops every 3D point inside the ego
box, so nothing is painted where the key shows our car.

Depth models (Neon licence check, council/PLAN-video-neon.md; coordinator 28 Nov):
  PRIMARY  depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf = the transformers port of
           depth-anything/Depth-Anything-V2-Metric-VKITTI-Small (Apache-2.0 tag on the original; trained on Virtual KITTI 2,
           whose data licence is NC). If the original .pth is in the local HF cache, every original tensor is checked
           bit-identical in the port and the result is recorded; otherwise the provenance says "not verified".
  FALLBACK depth-anything/Depth-Anything-V2-Small-hf (relative disparity): per tick and view, an affine disparity -> inverse
           depth fit anchored on the known camera height (pixels whose ray meets the road plane 3-25 m away, robust trimmed
           least squares), then the frozen per-camera scale.
  "flat"   no depth net: every pixel below the horizon is put on the road plane z = 0 (known camera height); the
           flat-ground warp row of scripts/guide_preview.py.
All from_pretrained / hub calls use local_files_only=True (jobs run with HF_HUB_OFFLINE=1).
"""
import hashlib, json, math, os, time
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pyquaternion import Quaternion

from mrv import carmap
from mrv.nusc import tf, sensor_to_world

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
PRIMARY = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
ORIG_REPO = "depth-anything/Depth-Anything-V2-Metric-VKITTI-Small"
ORIG_FILE = "depth_anything_v2_metric_vkitti_vits.pth"
PRIMARY_NOTE = ("transformers port of depth-anything/Depth-Anything-V2-Metric-VKITTI-Small (Apache-2.0 tag on the original "
                "checkpoint; trained on Virtual KITTI 2, data licence CC BY-NC-SA 3.0)")
FALLBACK = "depth-anything/Depth-Anything-V2-Small-hf"

# ---- chase (answer-key) camera ------------------------------------------------------------------------------------------
CH_BACK, CH_UP, CH_PITCH_DEG, CH_HFOV_DEG = 7.5, 2.4, 8.0, 60.0
KW, KH, CROP = 512, 288, 32          # key render size and the scorer's top crop
GW, GH = 512, 256                    # guide size (= scored key size)
FX = KW / 2 / math.tan(math.radians(CH_HFOV_DEG / 2))
CP, SP = math.cos(math.radians(CH_PITCH_DEG)), math.sin(math.radians(CH_PITCH_DEG))
K_CHASE = np.array([[FX, 0, KW / 2], [0, FX, KH / 2 - CROP], [0, 0, 1.0]])
assert (carmap.W, carmap.H, carmap.CROP) == (KW, KH, CROP) and abs(carmap._f - FX) < 1e-9, "chase camera differs from mrv/carmap.py"
OWN_NP = carmap.OWN                  # [256,512] bool, same hull + MaxFilter(9) as scripts/temporal_metrics.py
EGO_BOX = dict(x=(-2.6, 3.9), y=1.2, z=2.0)  # 3D points inside this ego-frame box are our own car -> dropped
CHASE = dict(back=CH_BACK, up=CH_UP, pitch_deg=-CH_PITCH_DEG, hfov_deg=CH_HFOV_DEG, render_wh=[KW, KH], crop_top=CROP,
             out_wh=[GW, GH], fx=FX, fy=FX, cx=KW / 2, cy=KH / 2 - CROP, source="scripts/sim_render.py VIEW['tpv'] + mrv/carmap.py project()")


def chase_cam_to_ego(dz=0.0):
    """4x4 chase-camera (x right, y down, z forward, OpenCV) -> tick ego frame (x fwd, y left, z up)."""
    T = np.eye(4)
    T[:3, 0] = [0, -1, 0]                # image right = ego -y
    T[:3, 1] = [-SP, 0, -CP]             # image down
    T[:3, 2] = [CP, 0, -SP]              # optical axis, pitched 8 deg down
    T[:3, 3] = [-CH_BACK, 0, CH_UP + dz]
    return T


def chase_project(P, dz=0.0):
    """P [..., 3] tick-ego points (torch) -> u, v (continuous px in the cropped 512x256 image), depth along the axis.
    Identical arithmetic to mrv/carmap.project()."""
    Yf, R_, Zu = P[..., 0] + CH_BACK, -P[..., 1], P[..., 2] - (CH_UP + dz)
    d = Yf * CP - Zu * SP
    up = Yf * SP + Zu * CP
    dd = d.clamp(min=1e-3)
    return KW / 2 + FX * R_ / dd, KH / 2 - FX * up / dd - CROP, d


def chase_pixel_dirs(h, w):
    """unit ray directions (chase camera frame) through the centres of an h x w grid over the cropped 512x256 image"""
    v, u = torch.meshgrid((torch.arange(h) + 0.5) * GH / h, (torch.arange(w) + 0.5) * GW / w, indexing="ij")
    d = torch.stack([(u - K_CHASE[0, 2]) / FX, (v - K_CHASE[1, 2]) / FX, torch.ones_like(u)], -1)
    return d / d.norm(dim=-1, keepdim=True)


def selftest():
    """chase_project == mrv/carmap.project on random points (run once on the cluster: python -m mrv.guide)"""
    rng = np.random.default_rng(0)
    P = np.stack([rng.uniform(-5, 60, 2000), rng.uniform(-20, 20, 2000), rng.uniform(-1, 4, 2000)], 1)
    uv_ref, d_ref = carmap.project(P)
    u, v, d = chase_project(torch.tensor(P))
    ok = d_ref > 0.5
    err = max(np.abs(u.numpy() - uv_ref[:, 0])[ok].max(), np.abs(v.numpy() - uv_ref[:, 1])[ok].max(), np.abs(d.numpy() - d_ref).max())
    # the 4x4 pose must give the same pixels
    T = np.linalg.inv(chase_cam_to_ego()); Pc = P @ T[:3, :3].T + T[:3, 3]
    uu = K_CHASE[0, 0] * Pc[:, 0] / Pc[:, 2] + K_CHASE[0, 2]; vv = K_CHASE[1, 1] * Pc[:, 1] / Pc[:, 2] + K_CHASE[1, 2]
    err2 = max(np.abs(uu - uv_ref[:, 0])[ok].max(), np.abs(vv - uv_ref[:, 1])[ok].max())
    assert err < 1e-6 and err2 < 1e-6, (err, err2)
    return float(max(err, err2))


# ---- bookkeeping --------------------------------------------------------------------------------------------------------
def code_sha(*paths):
    h = hashlib.sha1()
    for p_ in paths:
        h.update(open(p_, "rb").read())
    return h.hexdigest()[:12]


def file_sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def drive_list(spec, lists_path="data/temporal_lists.json", unseal=False):
    """comma list of: train48 (sorted train[::3][:48], the S0c drives), train, dev, a scene name, or a file of scene names.
    decision / final are the sealed exam drives and need unseal=True."""
    L = json.load(open(lists_path)); out = []
    for s in [x for x in spec.split(",") if x]:
        if s == "train48":
            out += sorted(L["train"])[::3][:48]
        elif s in ("decision", "final"):
            if not unseal:
                raise SystemExit(f"{s} drives are sealed (PLAN-video: exam sealed until recipe lock + N1)")
            out += sorted(L[s])
        elif s in L and isinstance(L[s], list):
            out += sorted(L[s])
        elif s.startswith("scene-"):
            out.append(s)
        elif os.path.exists(s):
            out += open(s).read().split()
        else:
            raise SystemExit(f"unknown drive spec {s}")
    return list(dict.fromkeys(out))


def import_signed_scorer(workdir):
    """Load the UNCHANGED signed scorer scripts/temporal_metrics.py as a module namespace (its functions segment, key_masks,
    iou_mat, hungarian, OWN, proc / seg, CAR, ROAD, SCORER_SHA, ...). The scorer is a script: it is run with an empty rollout
    folder so its scoring loop does nothing (same runpy pattern as scripts/score_rgb.py). Run from the repo root (its key-mask
    cache path runs/judge/keycache10_m2f is relative). Loads its models on cuda."""
    import runpy, sys
    empty = os.path.join(workdir, "_empty_roll"); os.makedirs(empty, exist_ok=True)
    argv = sys.argv
    sys.argv = ["scripts/temporal_metrics.py", "--roll", empty, "--out", os.path.join(workdir, f"_empty_metrics.{os.getpid()}.json")]
    try:
        return runpy.run_path("scripts/temporal_metrics.py", run_name="signed_temporal_metrics")
    finally:
        sys.argv = argv


def key_image(keys_dir, sc, tick):
    """answer key exactly as the signed scorer loads it: 512x288 -> crop top 32 rows -> 512x256 (None if missing)"""
    f = f"{keys_dir}/{sc}/t{tick:04d}.jpg"
    if not os.path.exists(f):
        return None
    return Image.open(f).convert("RGB").resize((KW, KH), Image.BILINEAR).crop((0, CROP, KW, KH))


def yaw_of(q):
    return Quaternion(q).yaw_pitch_roll[0]


class SceneIndex:
    """per-scene lookups: filename -> sample_data for the 6 cameras and LIDAR_TOP, keyframe ego poses, tick poses."""
    def __init__(s, nusc, scene_name):
        s.nusc = nusc
        scene = next(x for x in nusc.scene if x["name"] == scene_name)
        s.samples, t = [], scene["first_sample_token"]
        while t:
            s.samples.append(t); t = nusc.get("sample", t)["next"]
        s0 = nusc.get("sample", s.samples[0])
        s.by_file, s.lidar = {}, []
        for ch in CAMS + ["LIDAR_TOP"]:
            sd = nusc.get("sample_data", s0["data"][ch])
            while sd["prev"]:
                sd = nusc.get("sample_data", sd["prev"])
            while True:
                s.by_file[sd["filename"]] = sd
                if ch == "LIDAR_TOP":
                    s.lidar.append(sd)
                if not sd["next"]:
                    break
                sd = nusc.get("sample_data", sd["next"])
        s.lidar_ts = np.array([x["timestamp"] for x in s.lidar])
        s.key_poses = []
        for tok in s.samples:
            ep = nusc.get("ego_pose", nusc.get("sample_data", nusc.get("sample", tok)["data"]["LIDAR_TOP"])["ego_pose_token"])
            s.key_poses.append((np.array(ep["translation"], float), yaw_of(ep["rotation"])))

    def lidar_at(s, t_us):
        return s.lidar[int(np.argmin(np.abs(s.lidar_ts - t_us)))]

    def tick_pose(s, tick, mode="keyframe_interp", t_us=None):
        """4x4 world <- tick-ego frame (yaw-only, gravity aligned)"""
        if mode == "sweep":
            ep = s.nusc.get("ego_pose", s.lidar_at(t_us)["ego_pose_token"])
            tr, yaw = np.array(ep["translation"], float), yaw_of(ep["rotation"])
        else:
            k, fr = tick // 5, (tick % 5) / 5.0
            tr, yaw = s.key_poses[min(k, len(s.key_poses) - 1)]
            if k + 1 < len(s.key_poses) and fr > 0:
                t1, y1 = s.key_poses[k + 1]
                tr = tr + fr * (t1 - tr)
                yaw = yaw + fr * math.atan2(math.sin(y1 - yaw), math.cos(y1 - yaw))
        T = np.eye(4)
        T[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
        T[:3, 3] = tr
        return T

    def cam_geom(s, files, T_world_tick):
        """files: 6 camera filenames (manifest order = CAMS) -> K [6,3,3] (1600x900 px), cam2tick, cam2ego, cam2world [6,4,4]"""
        Ks, c2t, c2e, c2w = [], [], [], []
        inv = np.linalg.inv(T_world_tick)
        for f in files:
            sd = s.by_file[f]
            cs = s.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            Ks.append(np.array(cs["camera_intrinsic"], float))
            ce = tf(cs["rotation"], cs["translation"]); cw = sensor_to_world(s.nusc, sd)
            c2e.append(ce); c2w.append(cw); c2t.append(inv @ cw)
        return np.stack(Ks), np.stack(c2t), np.stack(c2e), np.stack(c2w)


def load_views(dataroot, files, wh, resample=Image.BILINEAR):
    """6 JPEGs -> uint8 [6, H, W, 3] at size wh (JPEG draft decoding: DCT-scaled, then resized)"""
    out = []
    for f in files:
        im = Image.open(f"{dataroot}/{f}")
        im.draft("RGB", wh)
        im = im.convert("RGB")
        if im.size != tuple(wh):
            im = im.resize(tuple(wh), resample)
        out.append(np.asarray(im))
    return np.stack(out)


# ---- depth network ------------------------------------------------------------------------------------------------------
def _hash(t):
    return hashlib.sha1(t.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def verify_same_weights(orig_sd, hf_sd):
    """every tensor of the original checkpoint must appear bit-identical (flattened) in the HF model (fused qkv re-joined)"""
    H = Counter(_hash(v) for v in hf_sd.values() if torch.is_tensor(v))
    groups = {}
    for k, v in hf_sd.items():
        for part in ("query", "key", "value"):
            tag = f".attention.attention.{part}."
            if tag in k:
                pre, suf = k.split(tag)
                groups.setdefault((pre, suf), {})[part] = v
    for d in groups.values():
        if len(d) == 3:
            H[_hash(torch.cat([d["query"], d["key"], d["value"]], 0))] += 1
    P = Counter(_hash(v) for v in orig_sd.values() if torch.is_tensor(v))
    missing = sum((P - H).values())
    return missing == 0, {"orig_tensors": int(sum(P.values())), "missing_in_hf": int(missing)}


def load_depth_model(model_id=PRIMARY, allow_fallback=True):
    from transformers import AutoModelForDepthEstimation
    info = {"requested": model_id}
    try:
        m = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        if not allow_fallback or model_id == FALLBACK:
            raise
        print(f"WARNING depth: {model_id} not loadable ({type(e).__name__}: {e}); FALLING BACK to {FALLBACK}", flush=True)
        info["primary_error"] = f"{type(e).__name__}: {e}"
        model_id = FALLBACK
        m = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=True)
    kind = "metric" if getattr(m.config, "depth_estimation_type", "relative") == "metric" else "relative"
    info.update(model_id=model_id, kind=kind, max_depth=float(getattr(m.config, "max_depth", 80.0) or 80.0))
    try:
        from huggingface_hub import snapshot_download
        info["weights_sha"] = file_sha(os.path.join(snapshot_download(model_id, local_files_only=True), "model.safetensors"))
    except Exception:  # noqa: BLE001
        info["weights_sha"] = None
    if model_id == PRIMARY:
        info["note"] = PRIMARY_NOTE
        try:  # optional: original-format checkpoint, only if already cached
            from huggingface_hub import hf_hub_download
            pth = hf_hub_download(ORIG_REPO, ORIG_FILE, local_files_only=True)
            sd = torch.load(pth, map_location="cpu", weights_only=True)
            sd = sd["model"] if isinstance(sd, dict) and isinstance(sd.get("model"), dict) else sd
            same, det = verify_same_weights(sd, m.state_dict())
            info["vkitti_original_check"] = {"identical": bool(same), **det, "orig_sha": file_sha(pth)}
        except Exception as e:  # noqa: BLE001
            info["vkitti_original_check"] = f"not verified ({type(e).__name__})"
    return m, info


class DepthNet:
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(s, model_id=PRIMARY, device="cuda", dtype=torch.float16, allow_fallback=True):
        s.model, s.info = load_depth_model(model_id, allow_fallback)
        s.model = s.model.to(device=device, dtype=dtype).eval()
        s.kind, s.device, s.dtype = s.info["kind"], device, dtype
        s.mean = torch.tensor(s.MEAN, device=device).view(1, 3, 1, 1); s.std = torch.tensor(s.STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def raw(s, x):
        """x uint8 [B,3,H,W] (H, W multiples of 14) -> metric depth (m) or relative disparity [B,H,W] float32"""
        y = (x.float() / 255 - s.mean) / s.std
        out = s.model(pixel_values=y.to(s.dtype)).predicted_depth
        if out.dim() == 4:
            out = out[:, 0]
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out[:, None].float(), size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return out.float()


# ---- LiDAR (supervision / ceiling only) ---------------------------------------------------------------------------------
def lidar_points_world(nusc, dataroot, lidar_sd):
    from nuscenes.utils.data_classes import LidarPointCloud
    P = LidarPointCloud.from_file(f"{dataroot}/{lidar_sd['filename']}").points[:3].T.astype(np.float64)
    T = sensor_to_world(nusc, lidar_sd)
    return P @ T[:3, :3].T + T[:3, 3]


def lidar_depth(P_world, cam2world, K, res, full_wh=(1600, 900), device="cuda"):
    """world points -> sparse per-view z-depth maps at depth resolution [6,Hd,Wd] (inf where no return; nearest return wins)"""
    Hd, Wd = res
    P = torch.as_tensor(P_world, dtype=torch.float64, device=device)
    D = torch.full((len(cam2world), Hd * Wd), float("inf"), dtype=torch.float32, device=device)
    for v in range(len(cam2world)):
        T = torch.as_tensor(np.linalg.inv(cam2world[v]), device=device)
        Pc = P @ T[:3, :3].T + T[:3, 3]
        z = Pc[:, 2]; m = z > 1.0
        Pc, z = Pc[m], z[m]
        Kv = torch.as_tensor(K[v], device=device)
        u = Kv[0, 0] * Pc[:, 0] / z + Kv[0, 2]; w = Kv[1, 1] * Pc[:, 1] / z + Kv[1, 2]
        j = torch.floor(u * Wd / full_wh[0]).long(); i = torch.floor(w * Hd / full_wh[1]).long()
        ok = (j >= 0) & (j < Wd) & (i >= 0) & (i < Hd)
        D[v].scatter_reduce_(0, (i * Wd + j)[ok], z[ok].float(), reduce="amin", include_self=True)
    return D.view(-1, Hd, Wd)


def densify(D, iters=4):
    """fill empty pixels from 3x3 neighbours, nearest depth wins (foreground-preferring), `iters` rings; filled pixels stay"""
    for _ in range(iters):
        mp = -F.max_pool2d(-D[:, None], 3, 1, 1)[:, 0]
        D = torch.where(torch.isfinite(D), D, mp)
    return D


# ---- renderer -----------------------------------------------------------------------------------------------------------
class GuideRenderer:
    def __init__(s, depth=None, scale_file=None, res=(294, 518), device="cuda", splat="zbuf", rmax=2, chase_dz=0.0,
                 max_depth=None, min_depth=0.5, soft_eps=0.05, lidar_fill="est", model_id=PRIMARY, allow_fallback=True,
                 no_depth_net=False, underlay=True):
        if no_depth_net:  # flat-ground warp only
            class _NoNet:
                info = {"model_id": "none (flat ground)", "kind": "flat", "max_depth": 80.0}; kind = "flat"
            s.depth = _NoNet()
        else:
            s.depth = depth if depth is not None else DepthNet(model_id, device, allow_fallback=allow_fallback)
        s.res, s.device, s.splat_mode, s.rmax, s.dz = tuple(res), device, splat, int(rmax), float(chase_dz)
        assert s.res[0] % 14 == 0 and s.res[1] % 14 == 0, "depth resolution must be a multiple of 14"
        s.max_depth = float(max_depth or s.depth.info.get("max_depth", 80.0)); s.min_depth = min_depth
        s.soft_eps, s.lidar_fill, s.underlay = soft_eps, lidar_fill, bool(underlay)
        s.scales, s.scale_sha, s.scale_file = None, None, scale_file
        if scale_file and s.depth.kind != "flat":
            J = json.load(open(scale_file))
            if J["model_id"] != s.depth.info["model_id"] or J["kind"] != s.depth.kind:
                raise SystemExit(f"scale file {scale_file} was fitted for {J['model_id']} ({J['kind']}), depth net is "
                                 f"{s.depth.info['model_id']} ({s.depth.kind})")
            if tuple(J["res"]) != s.res:
                raise SystemExit(f"scale file res {J['res']} != renderer res {s.res}")
            s.scales = torch.tensor([J["scales"][c] for c in CAMS], device=device, dtype=torch.float32)
            s.scale_sha = file_sha(scale_file)
        s.own = torch.from_numpy(OWN_NP.copy()).to(device)
        s._rays_key, s._rays = None, None

    def provenance(s):
        return {"depth_model_id": s.depth.info["model_id"], "depth_info": s.depth.info, "depth_kind": s.depth.kind,
                "depth_res_hw": list(s.res), "scale_file": s.scale_file, "scale_sha": s.scale_sha,
                "scales": None if s.scales is None else dict(zip(CAMS, s.scales.tolist())), "splat": s.splat_mode,
                "rmax": s.rmax, "soft_eps": s.soft_eps, "chase": CHASE, "chase_dz": s.dz, "max_depth": s.max_depth,
                "min_depth": s.min_depth, "ego_box": EGO_BOX,
                "underlay": "flat_ground (road plane z = 0 at the known camera height; depth points win; src 1 = depth, 2 = ground prior)"
                if s.underlay else None,
                "code_sha": code_sha(os.path.abspath(__file__))}

    def rays(s, K, full_wh=(1600, 900)):
        """camera-frame rays (z = 1) through the depth-pixel centres [6,Hd,Wd,3]"""
        key = (np.asarray(K).round(4).tobytes(), tuple(full_wh))
        if key != s._rays_key:
            Hd, Wd = s.res
            v, u = torch.meshgrid((torch.arange(Hd, device=s.device) + 0.5) * full_wh[1] / Hd,
                                  (torch.arange(Wd, device=s.device) + 0.5) * full_wh[0] / Wd, indexing="ij")
            Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=s.device)
            x = (u[None] - Kt[:, 0, 2, None, None]) / Kt[:, 0, 0, None, None]
            y = (v[None] - Kt[:, 1, 2, None, None]) / Kt[:, 1, 1, None, None]
            s._rays, s._rays_key = torch.stack([x, y, torch.ones_like(x)], -1), key
        return s._rays

    def ground_depth(s, K, cam2ego, full_wh=(1600, 900), dmin=3.0, dmax=25.0):
        """camera depth of the road plane z = 0 (ego frame) for pixels whose ray meets it dmin..dmax m away (else nan)"""
        r = s.rays(K, full_wh)
        Ce = torch.as_tensor(np.asarray(cam2ego), dtype=torch.float32, device=s.device)
        d = torch.einsum("vij,vhwj->vhwi", Ce[:, :3, :3], r); o = Ce[:, :3, 3]
        t = -o[:, 2, None, None] / d[..., 2]
        hit = o[:, None, None, :2] + t[..., None] * d[..., :2]
        dist = (hit - o[:, None, None, :2]).norm(dim=-1)
        ok = (d[..., 2] < -1e-3) & (t > 0) & (dist >= dmin) & (dist <= dmax)
        return torch.where(ok, t, torch.full_like(t, float("nan")))

    def anchor_relative(s, disp, K, cam2ego):
        """relative disparity -> metric depth per view: 1/z = a * disp + b on road-plane pixels (camera height), trimmed LS"""
        zg = s.ground_depth(K, cam2ego); out = torch.full_like(disp, float("nan"))
        for v in range(disp.shape[0]):
            m = torch.isfinite(zg[v])
            if m.sum() < 200:
                continue
            x, y = disp[v][m], 1.0 / zg[v][m]; keep = torch.ones_like(x, dtype=torch.bool)
            for _ in range(3):
                A = torch.stack([x[keep], torch.ones_like(x[keep])], 1)
                sol = torch.linalg.lstsq(A, y[keep][:, None]).solution[:, 0]
                r = (sol[0] * x + sol[1] - y).abs(); keep = r <= torch.quantile(r, 0.5)
            if sol[0] <= 0:
                continue
            out[v] = 1.0 / (sol[0] * disp[v] + sol[1]).clamp(min=1.0 / s.max_depth)
        return out

    def metric_depth(s, raw, K, cam2ego, scaled=True):
        z = raw if s.depth.kind == "metric" else s.anchor_relative(raw, K, cam2ego)
        if scaled and s.scales is not None:
            z = z * s.scales[:, None, None]
        return z

    def splat(s, P, col, fp, lab=None, prio=None):
        """P [N,3] tick-ego points, col [N,3] uint8, fp [N] metres per source pixel, lab [N] long (optional), prio [N] long
        (0 = depth point, 1 = ground-prior point; optional). Per chase pixel the winner is the lowest priority, then the
        nearest. Returns dict rgb uint8 [3,H,W], mask bool [1,H,W] (any source), src uint8 [1,H,W] (0 none, 1 depth,
        2 ground prior), invd float [1,H,W] (1 / chase depth), lab long [H,W] (-1 none)."""
        u, v, d = chase_project(P, s.dz)
        ok = (d > 0.5) & torch.isfinite(u) & torch.isfinite(v) & (u > -s.rmax - 1) & (u < GW + s.rmax + 1) & (v > -s.rmax - 1) & (v < GH + s.rmax + 1)
        u, v, d, col, fp = u[ok], v[ok], d[ok], col[ok], fp[ok]
        lab = lab[ok] if lab is not None else None
        prio = prio[ok] if prio is not None else torch.zeros_like(d, dtype=torch.long)
        N = int(d.numel()); HW = GH * GW
        labmap = torch.full((HW,), -1, dtype=torch.long, device=s.device)
        src = torch.zeros(HW, dtype=torch.uint8, device=s.device)
        rgb = torch.zeros(HW, 3, dtype=torch.uint8, device=s.device); invd = torch.zeros(HW, device=s.device)
        hit = torch.zeros(HW, dtype=torch.bool, device=s.device)
        if N:
            r = ((FX * fp / d - 1) / 2).ceil().clamp(0, s.rmax).long()   # square side 2r+1 >= projected footprint
            x0, y0 = u.floor().long(), v.floor().long()
            bits = max(1, N.bit_length()); lowmask = (1 << bits) - 1
            # key = priority (bit 61) | depth in mm | point index: amin = depth points first, then nearest, ties by index
            key = (prio << 61) | ((d * 1000).round().long().clamp(0, (1 << (61 - bits)) - 1) << bits) | torch.arange(N, device=s.device)
            LMAX = torch.iinfo(torch.long).max
            zb = torch.full((HW,), LMAX, dtype=torch.long, device=s.device); entries = []
            for dy in range(-s.rmax, s.rmax + 1):
                for dx in range(-s.rmax, s.rmax + 1):
                    xx, yy = x0 + dx, y0 + dy
                    m = (r >= max(abs(dx), abs(dy))) & (xx >= 0) & (xx < GW) & (yy >= 0) & (yy < GH)
                    if not bool(m.any()):
                        continue
                    pix, kk = (yy * GW + xx)[m], key[m]
                    zb.scatter_reduce_(0, pix, kk, reduce="amin", include_self=True)
                    if s.splat_mode == "soft":
                        entries.append((pix, kk))
            hit = zb != LMAX
            win = zb[hit] & lowmask
            rgb[hit] = col[win]; invd[hit] = 1.0 / d[win]; src[hit] = (prio[win] + 1).to(torch.uint8)
            if lab is not None:
                labmap[hit] = lab[win]
            if s.splat_mode == "soft":  # blend near-tied surfaces of the winning source (within soft_eps of the nearest)
                zmin = torch.full((HW,), float("inf"), device=s.device); zmin[hit] = d[win]
                wp = torch.full((HW,), -1, dtype=torch.long, device=s.device); wp[hit] = prio[win]
                acc = torch.zeros(HW, 3, device=s.device); ws = torch.zeros(HW, device=s.device)
                for pix, kk in entries:
                    idx = kk & lowmask; dd = d[idx]; zm = zmin[pix]
                    m = (dd <= zm * (1 + s.soft_eps)) & (prio[idx] == wp[pix])
                    w = torch.exp(-3.0 * (dd[m] - zm[m]) / (s.soft_eps * zm[m]))
                    acc.index_add_(0, pix[m], col[idx[m]].float() * w[:, None]); ws.index_add_(0, pix[m], w)
                rgb[hit] = (acc[hit] / ws[hit, None]).round().clamp(0, 255).to(torch.uint8)
        rgb = rgb.view(GH, GW, 3).permute(2, 0, 1).contiguous(); hit = hit.view(1, GH, GW); invd = invd.view(1, GH, GW)
        src = src.view(1, GH, GW); labmap = labmap.view(GH, GW)
        rgb[:, s.own] = 0; hit[:, s.own] = False; invd[:, s.own] = 0; src[:, s.own] = 0; labmap[s.own] = -1
        return {"rgb": rgb, "mask": hit, "src": src, "invd": invd, "lab": labmap}

    def _points(s, z, rays, cam2tick, K, full_wh):
        """camera-frame depth z [6,Hd,Wd] -> tick-ego points [6,Hd,Wd,3], footprint (m / source px), valid (range, not ego box)"""
        Tt = torch.as_tensor(np.asarray(cam2tick), dtype=torch.float32, device=s.device)
        P = torch.einsum("vij,vhwj->vhwi", Tt[:, :3, :3], rays * z[..., None]) + Tt[:, None, None, :3, 3]
        fx_d = torch.as_tensor(np.asarray(K)[:, 0, 0] * s.res[1] / full_wh[0], dtype=torch.float32, device=s.device)
        inbox = (P[..., 0] > EGO_BOX["x"][0]) & (P[..., 0] < EGO_BOX["x"][1]) & (P[..., 1].abs() < EGO_BOX["y"]) & (P[..., 2] < EGO_BOX["z"])
        valid = torch.isfinite(z) & (z > s.min_depth) & (z < s.max_depth) & ~inbox
        return P, z / fx_d[:, None, None], valid

    @torch.no_grad()
    def render(s, views, K, cam2tick, cam2ego, depth_override=None, full_wh=(1600, 900), timing=True, mode="est", labels=None,
               underlay=None):
        """views uint8 [6,Hd,Wd,3] (numpy or tensor) at the depth resolution; K [6,3,3] full-res; cam2tick / cam2ego [6,4,4].
        depth_override [6,Hd,Wd] (LiDAR ceiling): finite pixels replace the estimate; elsewhere the estimate (lidar_fill='est')
        or nothing (lidar_fill='none'). mode='flat': no depth net, every pixel on the road plane z = 0 of its own ego frame
        (pixels above the horizon are dropped). underlay (default: the renderer's setting, production = True; ignored in
        mode 'flat'): the flat-ground warp of the same pixels is splatted UNDER the depth points (Fable C9 / Astra S1 review):
        depth points win wherever they land, ground-prior points only fill chase pixels no depth point reaches; "src" tells
        which (1 depth, 2 ground prior). labels [6,Hd,Wd] long (-1 = none, optional): the label of the winning DEPTH point per
        chase pixel is returned as "lab" [256,512] (ground-prior points carry -1). Returns dict rgb [3,256,512] u8,
        mask [1,256,512] bool (any source), src [1,256,512] u8, invd [1,256,512] f32, coverage {valid, depth, prior} (shares
        of the non-own-car pixels), depth [6,Hd,Wd], lab, ms."""
        underlay = s.underlay if underlay is None else underlay
        sync = (lambda: torch.cuda.synchronize()) if (timing and str(s.device).startswith("cuda")) else (lambda: None)
        sync(); t0 = time.perf_counter()
        x = torch.as_tensor(views).to(s.device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        assert tuple(x.shape[-2:]) == s.res, (x.shape, s.res)
        if mode == "flat":
            z = s.ground_depth(K, cam2ego, full_wh, dmin=0.0, dmax=s.max_depth); sync(); t1 = time.perf_counter()
        else:
            raw = s.depth.raw(x); sync(); t1 = time.perf_counter()
            z = s.metric_depth(raw, K, cam2ego)
        if depth_override is not None:
            o = torch.as_tensor(depth_override, device=s.device, dtype=torch.float32)
            z = torch.where(torch.isfinite(o), o, z if s.lidar_fill == "est" else torch.full_like(z, float("nan")))
        rays = s.rays(K, full_wh)
        col = x.permute(0, 2, 3, 1)
        P, fp, valid = s._points(z, rays, cam2tick, K, full_wh)
        lab_all = torch.as_tensor(labels, device=s.device).long() if labels is not None else None
        Ps, cs, fs, ls, ps = [P[valid]], [col[valid]], [fp[valid]], [], [torch.zeros(int(valid.sum()), dtype=torch.long, device=s.device)]
        if lab_all is not None:
            ls.append(lab_all[valid])
        if underlay and mode != "flat":
            zf = s.ground_depth(K, cam2ego, full_wh, dmin=0.0, dmax=s.max_depth)
            Pf, fpf, vf = s._points(zf, rays, cam2tick, K, full_wh)
            Ps.append(Pf[vf]); cs.append(col[vf]); fs.append(fpf[vf]); ps.append(torch.ones(int(vf.sum()), dtype=torch.long, device=s.device))
            if lab_all is not None:
                ls.append(torch.full((int(vf.sum()),), -1, dtype=torch.long, device=s.device))
        o = s.splat(torch.cat(Ps), torch.cat(cs), torch.cat(fs), torch.cat(ls) if ls else None, torch.cat(ps))
        sync(); t2 = time.perf_counter()
        free = ~s.own; nf = float(free.sum())
        cov = {"valid": float(o["mask"][0][free].sum()) / nf, "depth": float((o["src"][0][free] == 1).sum()) / nf,
               "prior": float((o["src"][0][free] == 2).sum()) / nf}
        return {"rgb": o["rgb"], "mask": o["mask"], "src": o["src"], "invd": o["invd"], "coverage": cov, "depth": z,
                "lab": o["lab"] if labels is not None else None,
                "ms": {"depth": (t1 - t0) * 1000, "splat": (t2 - t1) * 1000, "total": (t2 - t0) * 1000}}


if __name__ == "__main__":
    print("GUIDE_SELFTEST max |err| px", selftest())
