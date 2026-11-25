"""Inspect the RGBCamera lens MetaDrive uses and compare its projection with a fresh PerspectiveLens."""
import math
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
from panda3d.core import Point3, Point2, PerspectiveLens
env = ScenarioEnv(dict(data_directory="data/scn_mini", num_scenarios=1, agent_policy=ReplayEgoCarPolicy, use_render=False,
                       image_observation=True, norm_pixel=False, sensors={"rgb": (RGBCamera, 448, 252)},
                       vehicle_config=dict(image_source="rgb"), log_level=50))
env.reset(seed=0)
cam = env.engine.get_sensor("rgb")
L = cam.lens
print("type", type(L).__name__, "film", L.getFilmSize(), "fov", L.getFov(), "offset", L.getFilmOffset(),
      "viewhpr", L.getViewHpr(), "viewvec", L.getViewVector(), "upvec", L.getUpVector(), "aspect", L.getAspectRatio())
print("attrs", [a for a in dir(cam) if not a.startswith("__")][:60])
L.setFov(64.6, 39.15)
for p in [(0, 12, 0), (3, 12, -1)]:
    p2 = Point2(); ok = L.project(Point3(*p), p2); print(p, ok, (p2[0] + 1) / 2 * 448, (1 - p2[1]) / 2 * 252)
env.close()
