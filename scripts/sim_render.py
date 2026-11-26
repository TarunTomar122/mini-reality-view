"""Experiment S, step 2: render sim pairs from ScenarioNet replays of official nuScenes TRAIN scenes.

For every chosen keyframe (scenario step = keyframe * 5 at 10 Hz) of a scene:
  - 6 rig cameras placed from that log's calibrated_sensor (position, rotation, FOV, principal point; checked to
    0 px against K by sim_verify.py), 448x252 -> data/sim/cams/<scene>/<k>_<token>_<CAM>.jpg
  - the low drone camera (9 m back, 4.5 m up, looking at (15, 0, 0.5), 60 deg HFOV) with our own car hidden,
    512x288 -> data/sim/keys/<scene>/<k>_<token>.png (alpha 255 = fully trusted: the sim key is complete)
usage: python sim_render.py <scenario_dir> <nuscenes_root> <version> <out_dir> --scenes 100 --every 2 --shard i/n
"""
import argparse, faulthandler, math, os
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train as TRAIN, val as VALS
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
from panda3d.core import NodePath, Point3, Vec3

p = argparse.ArgumentParser()
p.add_argument("sdir"); p.add_argument("nroot"); p.add_argument("ver"); p.add_argument("out")
p.add_argument("--scenes", type=int, default=100)
p.add_argument("--every", type=int, default=2, help="use every n-th keyframe")
p.add_argument("--shard", default="0/1")
p.add_argument("--skip", default="", help="comma list of scenes to skip (e.g. ones where MetaDrive hangs)")
p.add_argument("--view", default="drone", choices=["drone", "tpv"],
               help="drone: 9 m back, 4.5 m up, own car hidden; tpv: third-person chase camera 7.5 m behind the car centre, 2.4 m up, pitched 8 deg down, own car shown")
p.add_argument("--split", default="train", choices=["train", "val"])
p.add_argument("--dense", action="store_true",
               help="temporal data (council/PLAN-temporal.md): a key at EVERY 10 Hz sim tick of the whole drive, keys only, "
                    "saved as t<tick>.jpg plus ticks.json (keyframe k = tick 5k, with its sample token)")
p.add_argument("--lit", action="store_true", help="keyframe mode: render each key twice and keep the second (same lighting and shadows as --dense)")
p.add_argument("--only_file", default="", help="file with scene names (one per line): render just these")
p.add_argument("--keys_name", default="keys", help="sub-folder for the answer keys (e.g. keys_tpv)")
p.add_argument("--keys_only", action="store_true", help="do not render the 6 rig cameras (already rendered)")
p.add_argument("--only", default="", help="comma list of scenes: render just these (any train scene in sdir)")
a = p.parse_args()
VIEW = {"drone": dict(pos=[0, -9.0, 4.5], pitch=-math.degrees(math.atan2(4.0, 24.0)), hide_ego=True),
        "tpv": dict(pos=[0, -7.5, 2.4], pitch=-8.0, hide_ego=False)}[a.view]
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
W, H, DW, DH = 448, 252, 512, 288
nusc = NuScenes(a.ver, a.nroot, verbose=False)
by_name = {s["name"]: s for s in nusc.scene}


def to_panda(v):
    return np.array([-v[1], v[0], v[2]])


def cam_pose(cs):
    R = Quaternion(cs["rotation"]).rotation_matrix
    pos = to_panda(np.array(cs["translation"]))
    fwd, up = to_panda(R @ np.array([0, 0, 1.0])), to_panda(R @ np.array([0, -1.0, 0]))
    t = NodePath("t"); t.setPos(*pos); t.lookAt(Point3(*(pos + fwd)), Vec3(*up))
    K = np.array(cs["camera_intrinsic"]) * np.array([[W / 1600], [H / 900], [1]])
    hf, vf = 2 * math.degrees(math.atan(W / 2 / K[0, 0])), 2 * math.degrees(math.atan(H / 2 / K[1, 1]))
    shift = (-(K[0, 2] - W / 2) / (W / 2), (K[1, 2] - H / 2) / (H / 2))
    return list(pos), list(t.getHpr()), hf, vf, shift


# pick scenarios: official train scenes only, deterministic spread over the list, then shard
import glob, pickle
summ = None
for f in glob.glob(f"{a.sdir}/**/dataset_summary.pkl", recursive=True):
    if os.path.dirname(f) == a.sdir.rstrip("/"):
        summ = f
summ = summ or glob.glob(f"{a.sdir}/dataset_summary.pkl")[0]
names = list(pickle.load(open(summ, "rb")).keys())
env = ScenarioEnv(dict(data_directory=a.sdir, num_scenarios=len(names), agent_policy=ReplayEgoCarPolicy,
                       reactive_traffic=False, use_render=False, image_observation=True, norm_pixel=False,
                       sensors={"rgb": (RGBCamera, W, H), "drone": (RGBCamera, DW, DH)},
                       vehicle_config=dict(image_source="rgb", show_navi_mark=False, show_dest_mark=False,
                                           show_line_to_dest=False), log_level=50))
cands = []
for i, fn in enumerate(names):
    sc_name = next((n for n in by_name if n in fn), None)
    if sc_name and sc_name in set(TRAIN if a.split == "train" else VALS):
        cands.append((i, sc_name))
step = max(1, len(cands) // a.scenes)
chosen = cands[::step][: a.scenes]
if a.only or a.only_file:
    want = set(a.only.split(",")) if a.only else set(open(a.only_file).read().split())
    chosen = [c for c in cands if c[1] in want]
si, sn = map(int, a.shard.split("/"))
chosen = chosen[si::sn]
print("scenarios", len(names), "train candidates", len(cands), "this shard", len(chosen), flush=True)
os.makedirs(f"{a.out}/cams", exist_ok=True); os.makedirs(f"{a.out}/{a.keys_name}", exist_ok=True)
n_pairs = 0
SKIP = set(x for x in a.skip.split(",") if x)
for idx, sc_name in chosen:
    kd = f"{a.out}/{a.keys_name}/{sc_name}"
    # resume: DONE = finished; STARTED without DONE = a previous attempt hung or died here -> skip it;
    # a folder without markers is from the first (marker-less) run: finished unless listed in --skip
    if sc_name in SKIP or os.path.exists(f"{kd}/DONE") or os.path.exists(f"{kd}/STARTED") or \
            (os.path.isdir(kd) and not os.path.exists(f"{kd}/STARTED")):
        print("skip", sc_name, flush=True)
        continue
    scene = by_name[sc_name]
    toks, t = [], scene["first_sample_token"]
    while t:
        toks.append(t); t = nusc.get("sample", t)["next"]
    s0 = nusc.get("sample", toks[0])
    poses = {c: cam_pose(nusc.get("calibrated_sensor", nusc.get("sample_data", s0["data"][c])["calibrated_sensor_token"])) for c in CAMS}
    env.reset(seed=idx)
    os.makedirs(kd, exist_ok=True); open(f"{kd}/STARTED", "w").close()
    # per-scene watchdog in C (works even if MetaDrive hangs inside native code holding the GIL): dump the stack and
    # exit after 5 min; the wrapper restarts and this scene is skipped (STARTED without DONE)
    faulthandler.dump_traceback_later(1200 if a.dense else 300, exit=True)  # after reset: a crash at start-up is not a bad scene
    cam, drone = env.engine.get_sensor("rgb"), env.engine.get_sensor("drone")
    drone.lens.setFov(60, 2 * math.degrees(math.atan(math.tan(math.radians(30)) * DH / DW)))
    od_c, od_k = f"{a.out}/cams/{sc_name}", f"{a.out}/{a.keys_name}/{sc_name}"
    os.makedirs(od_c, exist_ok=True); os.makedirs(od_k, exist_ok=True)
    cur = 0
    if a.dense:  # every sim tick from the first to the last keyframe
        hid = [ch for ch in env.agent.origin.getChildren() if "camera" not in ch.getName().lower()] if VIEW["hide_ego"] else []
        last, done, ticks = (len(toks) - 1) * 5, False, {}
        while cur <= last and not done:
            for ch in hid:
                ch.hide()
            im = drone.perceive(to_float=False, new_parent_node=env.agent.origin, position=VIEW["pos"], hpr=[0, VIEW["pitch"], 0])
            for ch in hid:
                ch.show()
            Image.fromarray(np.ascontiguousarray(im[..., ::-1])).save(f"{od_k}/t{cur:04d}.jpg", quality=95)
            ticks[cur] = toks[cur // 5] if cur % 5 == 0 else None
            n_pairs += 1
            if cur == last:
                break
            *_, term, trunc, _ = env.step([0, 0]); cur += 1; done = term or trunc
        import json as _json
        _json.dump({"ticks": ticks, "last_tick": cur, "complete": not done}, open(f"{od_k}/ticks.json", "w"))
        faulthandler.cancel_dump_traceback_later()
        open(f"{od_k}/DONE", "w").close()
        print(sc_name, "dense ticks", cur + 1, "frames so far", n_pairs, flush=True)
        continue
    for k, tok in enumerate(toks):
        if k % a.every:
            continue
        target = k * 5
        done = False
        while cur < target and not done:
            *_, term, trunc, _ = env.step([0, 0]); cur += 1; done = term or trunc
        if done:
            break
        for c in ([] if a.keys_only else CAMS):
            pos, hpr, hf, vf, sh = poses[c]
            cam.lens.setFov(hf, vf)
            fs = cam.lens.getFilmSize(); cam.lens.setFilmOffset(sh[0] * 0.5 * fs[0], sh[1] * 0.5 * fs[1])
            im = cam.perceive(to_float=False, new_parent_node=env.agent.origin, position=pos, hpr=hpr)
            Image.fromarray(im[..., ::-1]).save(f"{od_c}/{k:03d}_{tok}_{c}.jpg", quality=92)
        hid = [ch for ch in env.agent.origin.getChildren() if "camera" not in ch.getName().lower()] if VIEW["hide_ego"] else []
        for ch in hid:
            ch.hide()
        im = drone.perceive(to_float=False, new_parent_node=env.agent.origin, position=VIEW["pos"], hpr=[0, VIEW["pitch"], 0])
        if a.lit:  # render twice, keep the second: MetaDrive's first render after several unrendered ticks has no shadows /
            # flat lighting; the second matches frames rendered every tick (dense mode). Use for keys mixed with dense ones.
            im = drone.perceive(to_float=False, new_parent_node=env.agent.origin, position=VIEW["pos"], hpr=[0, VIEW["pitch"], 0])
        for ch in hid:
            ch.show()
        rgba = np.concatenate([im[..., ::-1], np.full(im.shape[:2] + (1,), 255, np.uint8)], -1)
        Image.fromarray(rgba, "RGBA").save(f"{od_k}/{k:03d}_{tok}.png")
        n_pairs += 1
    faulthandler.cancel_dump_traceback_later()
    open(f"{od_k}/DONE", "w").close()
    print(sc_name, "pairs so far", n_pairs, flush=True)
env.close()
print("SIM_RENDER_DONE", n_pairs)
