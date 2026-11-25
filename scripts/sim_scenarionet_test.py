"""Experiment S, step 2: replay real nuScenes layouts + traffic in MetaDrive (ScenarioNet) and shoot the 6-camera rig
plus our low drone camera. Our own car is hidden in the drone shot (real keys never contain it).
usage: python sim_scenarionet_test.py <converted_scenario_dir> <out_dir> [n_scenarios]
"""
import os, sys, time
import numpy as np
from PIL import Image
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera

data, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
os.makedirs(out, exist_ok=True)
RIG = [((1.5, 0.5, 1.55), 55), ((1.7, 0.0, 1.55), 0), ((1.5, -0.5, 1.55), -55),
       ((1.0, 0.5, 1.55), 110), ((0.0, 0.0, 1.55), 180), ((1.0, -0.5, 1.55), -110)]
W, H = 448, 252
env = ScenarioEnv(dict(data_directory=data, num_scenarios=n, agent_policy=ReplayEgoCarPolicy, reactive_traffic=False,
                       use_render=False, image_observation=True, norm_pixel=False, sensors={"rgb": (RGBCamera, W, H)},
                       vehicle_config=dict(image_source="rgb", show_navi_mark=False), log_level=50))
cam = None
for sc in range(n):
    t0 = time.time()
    env.reset(seed=sc)
    cam = env.engine.get_sensor("rgb")
    for k in range(60):
        env.step([0, 0])
        if k in (10, 40):
            cam.lens.setFov(70)
            ims = [cam.perceive(to_float=False, new_parent_node=env.agent.origin, position=[-p[1], p[0], p[2]], hpr=[yaw, 0, 0]) for p, yaw in RIG]
            env.agent.origin.hide()
            cam.lens.setFov(60)
            drone = cam.perceive(to_float=False, new_parent_node=env.agent.origin, position=[0, -9.0, 4.5],
                                 hpr=[0, -np.degrees(np.arctan2(4.0, 24.0)), 0])
            env.agent.origin.show()
            grid = np.concatenate([np.concatenate(ims[:3], 1), np.concatenate(ims[3:], 1)], 0)
            Image.fromarray(grid[..., ::-1]).save(f"{out}/s{sc}_f{k}_cams.jpg")
            Image.fromarray(drone[..., ::-1]).save(f"{out}/s{sc}_f{k}_drone.jpg")
    print("scenario", sc, "ok in", round(time.time() - t0, 1), "s", flush=True)
env.close()
print("SCN_OK")
