"""Experiment S feasibility: can MetaDrive render 6 nuScenes-rig cameras + our low drone camera headless on a GPU node?

Cameras are attached to the ego car. MetaDrive/Panda3D frame: x right, y forward, z up; hpr = (heading/yaw left-positive,
pitch up-positive, roll) in degrees. The nuScenes rig (ego frame x forward, y left) is converted below.
"""
import os, sys, time, json
import numpy as np
from PIL import Image

out = sys.argv[1] if len(sys.argv) > 1 else "runs/sim_test"
os.makedirs(out, exist_ok=True)
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

# approximate nuScenes rig (ego frame: x fwd, y left, z up; yaw left-positive), from calibrated_sensor averages
RIG = {"CAM_FRONT_LEFT": ((1.5, 0.5, 1.55), 55), "CAM_FRONT": ((1.7, 0.0, 1.55), 0), "CAM_FRONT_RIGHT": ((1.5, -0.5, 1.55), -55),
       "CAM_BACK_LEFT": ((1.0, 0.5, 1.55), 110), "CAM_BACK": ((0.0, 0.0, 1.55), 180), "CAM_BACK_RIGHT": ((1.0, -0.5, 1.55), -110)}
W, H = 448, 252
cfg = dict(use_render=False, image_observation=True, norm_pixel=False, num_scenarios=20, start_seed=0, map=4,
           traffic_density=0.25, random_lane_width=True, random_agent_model=False,
           sensors={"rgb": (RGBCamera, W, H)}, vehicle_config=dict(image_source="rgb"), log_level=50)
t0 = time.time()
env = MetaDriveEnv(cfg)
obs, _ = env.reset(seed=3)
for _ in range(40):
    obs, *_ = env.step([0.0, 0.3])
cam = env.engine.get_sensor("rgb")
print("env up in", round(time.time() - t0, 1), "s", flush=True)


def shot(pos_ego, yaw_deg, pitch_deg=0.0, fov=None):
    x, y, z = pos_ego
    if fov is not None:
        cam.lens.setFov(fov)
    return cam.perceive(to_float=False, new_parent_node=env.agent.origin, position=[-y, x, z], hpr=[yaw_deg, pitch_deg, 0])


t1 = time.time()
ims = [shot(p, yaw, fov=70) for p, yaw in RIG.values()]
# low drone: 9 m back, 4.5 m up, looking at (15, 0, 0.5) -> pitch = -atan(4/24)
pitch = -np.degrees(np.arctan2(4.0, 24.0))
drone = shot((-9.0, 0.0, 4.5), 0, pitch, fov=60)
print("7 shots in", round(time.time() - t1, 2), "s", [i.shape for i in ims[:1]], drone.shape, flush=True)
row1 = np.concatenate(ims[:3], 1); row2 = np.concatenate(ims[3:], 1)
Image.fromarray(np.concatenate([row1, row2], 0)[..., ::-1]).save(f"{out}/cams.jpg")
Image.fromarray(drone[..., ::-1]).save(f"{out}/drone.jpg")
env.close()
print("SIM_OK")
