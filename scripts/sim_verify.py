"""Experiment S, step 1: verify that ScenarioNet replays line up with the real nuScenes data before collecting pairs.

Checks (council/PLAN-sim.md):
  a) cars: replayed track positions at keyframes vs nuScenes annotation centres (after one constant frame offset per
     scene) -> median error for cars within 25 m of the ego must be <= 0.3 m;
  b) ego origin: sim agent position vs logged ego pose (rear axle) -> offset in metres along/across the car;
  c) cameras: each sim camera is placed from that log's calibrated_sensor (position, full rotation, intrinsics -> FOV
     and principal point); 3D points projected by the Panda3D lens vs by K must agree within 2 px;
  d) hiding our own car in the drone view: render with and without hiding and report the changed-pixel fraction.
usage: python sim_verify.py <scenario_dir> <nuscenes_root> <version> <out_dir> [n_scenes]
"""
import os, sys, json, math
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
from panda3d.core import NodePath, Point3, Vec3, Point2

sdir, nroot, ver, out = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 5
os.makedirs(out, exist_ok=True)
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
W, H = 448, 252
nusc = NuScenes(ver, nroot, verbose=False)
env = ScenarioEnv(dict(data_directory=sdir, num_scenarios=n, agent_policy=ReplayEgoCarPolicy, reactive_traffic=False,
                       use_render=False, image_observation=True, norm_pixel=False, sensors={"rgb": (RGBCamera, W, H)},
                       vehicle_config=dict(image_source="rgb", show_navi_mark=False, show_dest_mark=False,
                                           show_line_to_dest=False), log_level=50))


def ego_to_panda(v):  # nuScenes ego (x fwd, y left, z up) -> Panda3D vehicle frame (x right, y fwd, z up)
    return np.array([-v[1], v[0], v[2]])


def cam_pose(cs):
    """position and hpr (Panda3D, relative to the vehicle origin) + hfov/vfov + principal-point shift, at 448x252."""
    R = Quaternion(cs["rotation"]).rotation_matrix
    pos = ego_to_panda(np.array(cs["translation"]))
    fwd = ego_to_panda(R @ np.array([0, 0, 1.0]))
    up = ego_to_panda(R @ np.array([0, -1.0, 0]))
    tmp = NodePath("tmp")
    tmp.setPos(*pos)
    tmp.lookAt(Point3(*(pos + fwd)), Vec3(*up))
    K = np.array(cs["camera_intrinsic"]) * np.array([[448 / 1600], [252 / 900], [1]])
    hfov = 2 * math.degrees(math.atan(W / 2 / K[0, 0]))
    vfov = 2 * math.degrees(math.atan(H / 2 / K[1, 1]))
    shift = (-(K[0, 2] - W / 2) / (W / 2), (K[1, 2] - H / 2) / (H / 2))  # Panda3D film offset (sign checked: 0 px vs K)
    return pos, tmp.getHpr(), hfov, vfov, shift, K, R


report = {"scenes": []}
car_err, ego_off, px_err, hide_frac = [], [], [], []
for si in range(n):
    env.reset(seed=si)
    sc = env.engine.data_manager.current_scenario
    meta = sc["metadata"]
    name = meta.get("scenario_id", meta.get("id", str(si)))
    scene = next((s for s in nusc.scene if s["name"] in str(name) or s["token"] in str(name)), None)
    if scene is None:
        print("no nuScenes match for", name, flush=True); continue
    cam = env.engine.get_sensor("rgb")
    toks, t = [], scene["first_sample_token"]
    while t:
        toks.append(t); t = nusc.get("sample", t)["next"]
    s0 = nusc.get("sample", toks[0])
    cs = {c: nusc.get("calibrated_sensor", nusc.get("sample_data", s0["data"][c])["calibrated_sensor_token"]) for c in CAMS}
    # --- (a)+(b): positions at keyframes (scenario steps are 10 Hz; keyframes every 5 steps) ---
    tracks = sc["tracks"]; sdc = meta["sdc_id"]
    ts = np.array(meta["ts"]) if "ts" in meta else None
    errs, offs = [], []
    for k, tok in enumerate(toks):
        step = min(k * 5, len(tracks[sdc]["state"]["position"]) - 1)
        smp = nusc.get("sample", tok)
        lsd = nusc.get("sample_data", smp["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", lsd["ego_pose_token"])
        e_log = np.array(pose["translation"][:2])
        e_sim = np.array(tracks[sdc]["state"]["position"][step][:2])
        offs.append(e_sim - e_log)
        for ann_tok in smp["anns"]:
            a = nusc.get("sample_annotation", ann_tok)
            if not a["category_name"].startswith("vehicle.car"):
                continue
            tid = a["instance_token"]
            if tid not in tracks or not tracks[tid]["state"]["valid"][step]:
                continue
            p_sim = np.array(tracks[tid]["state"]["position"][step][:2])
            if np.linalg.norm(np.array(a["translation"][:2]) - e_log) > 25:
                continue
            errs.append((p_sim - np.array(a["translation"][:2])))
    offs = np.array(offs); off0 = np.median(offs, 0)
    if errs:
        errs = np.linalg.norm(np.array(errs) - off0, axis=1); car_err += list(errs)
    # ego origin: compare sim agent (after a few steps) with the logged ego track at the same step
    env.step([0, 0]); env.step([0, 0])
    ag = np.array(env.agent.position); tr = np.array(tracks[sdc]["state"]["position"][2][:2])
    hd = env.agent.heading_theta
    d = ag - tr
    ego_off.append([d[0] * math.cos(hd) + d[1] * math.sin(hd), -d[0] * math.sin(hd) + d[1] * math.cos(hd)])
    # --- (c): lens vs K projection of a grid of 3D points in front of each camera ---
    for c in CAMS:
        pos, hpr, hfov, vfov, shift, K, R = cam_pose(cs[c])
        cam.lens.setFov(hfov, vfov); cam.lens.setFilmOffset(shift[0] * 0.5 * cam.lens.getFilmSize()[0], shift[1] * 0.5 * cam.lens.getFilmSize()[1])
        for x in (-3, 0, 3):
            for y in (-1, 0, 1):
                pc = np.array([x, y, 12.0])  # camera frame (OpenCV)
                uv = (K @ pc)[:2] / pc[2]
                # same point via Panda lens: camera space Panda = (x right, y fwd, z up) = (pc_x, pc_z, -pc_y)
                p2 = Point2()
                if cam.lens.project(Point3(pc[0], pc[2], -pc[1]), p2):
                    u = (p2[0] + 1) / 2 * W; v = (1 - p2[1]) / 2 * H
                    px_err.append(float(np.hypot(u - uv[0], v - uv[1])))
    # --- (d): hide our car in the drone view, and save a verification sheet ---
    for _ in range(20):
        env.step([0, 0])
    cam.lens.setFilmOffset(0, 0)
    ims = []
    for c in CAMS:
        pos, hpr, hfov, vfov, shift, K, R = cam_pose(cs[c])
        cam.lens.setFov(hfov, vfov)
        cam.lens.setFilmOffset(shift[0] * 0.5 * cam.lens.getFilmSize()[0], shift[1] * 0.5 * cam.lens.getFilmSize()[1])
        ims.append(cam.perceive(to_float=False, new_parent_node=env.agent.origin, position=list(pos), hpr=list(hpr)))
    cam.lens.setFilmOffset(0, 0); cam.lens.setFov(60, 2 * math.degrees(math.atan(math.tan(math.radians(30)) * H / W)))
    dpos, dhpr = [0, -9.0, 4.5], [0, -math.degrees(math.atan2(4.0, 24.0)), 0]
    shown = cam.perceive(to_float=False, new_parent_node=env.agent.origin, position=dpos, hpr=dhpr)
    hidden_nodes = [ch for ch in env.agent.origin.getChildren() if "camera" not in ch.getName().lower()]
    for ch in hidden_nodes:
        ch.hide()
    hidden = cam.perceive(to_float=False, new_parent_node=env.agent.origin, position=dpos, hpr=dhpr)
    for ch in hidden_nodes:
        ch.show()
    frac = float((np.abs(shown.astype(int) - hidden.astype(int)).sum(-1) > 30).mean())
    hide_frac.append(frac)
    grid = np.concatenate([np.concatenate(ims[:3], 1), np.concatenate(ims[3:], 1)], 0)
    Image.fromarray(grid[..., ::-1]).save(f"{out}/{scene['name']}_cams.jpg")
    Image.fromarray(np.concatenate([shown, hidden], 1)[..., ::-1]).save(f"{out}/{scene['name']}_drone_shown_hidden.jpg")
    report["scenes"].append({"scene": scene["name"], "cars": len(errs) if len(errs) else 0, "frame_offset_m": off0.tolist(),
                             "offset_spread_m": float(np.linalg.norm(offs - off0, axis=1).max()), "hide_changed_frac": frac,
                             "hfov": {c: round(cam_pose(cs[c])[2], 1) for c in CAMS}})
    print(report["scenes"][-1], flush=True)
report["car_centre_err_median_m"] = float(np.median(car_err)) if car_err else None
report["car_centre_err_p90_m"] = float(np.percentile(car_err, 90)) if car_err else None
report["ego_offset_along_across_m"] = np.median(np.array(ego_off), 0).tolist() if ego_off else None
report["lens_vs_K_px_median"] = float(np.median(px_err)) if px_err else None
report["lens_vs_K_px_max"] = float(np.max(px_err)) if px_err else None
report["hide_changed_frac_median"] = float(np.median(hide_frac)) if hide_frac else None
json.dump(report, open(f"{out}/verify.json", "w"), indent=1)
print("VERIFY", json.dumps({k: v for k, v in report.items() if k != "scenes"}))
env.close()
