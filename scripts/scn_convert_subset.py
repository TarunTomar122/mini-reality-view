"""Convert a subset of official TRAIN (or VAL, 4th arg) scenes, evenly spread, to ScenarioNet scenarios.
The stock converter loads one full nuScenes index per worker (~50 GB for 8 workers) and has no subset option."""
import sys
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train as TRAIN, val as VALS
from scenarionet.converter.nuscenes.utils import convert_nuscenes_scenario
from scenarionet.converter.utils import write_to_directory

out, n, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
split = sys.argv[4] if len(sys.argv) > 4 else "train"
nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes_full", verbose=False)
cands = [s for s in nusc.scene if s["name"] in set(TRAIN if split == "train" else VALS)]
step = max(1, len(cands) // n)
scenes = cands[::step][:n]
print("converting", len(scenes), "train scenes, e.g.", [s["name"] for s in scenes[:5]], flush=True)
write_to_directory(convert_func=convert_nuscenes_scenario, scenarios=scenes, output_path=out, dataset_version="v1.0-trainval",
                   dataset_name="nuscenes", overwrite=True, num_workers=workers, nuscenes=[nusc] * workers,
                   past=[2] * workers, future=[6] * workers, prediction=[False] * workers, map_radius=[500] * workers)
print("SUBSET_DONE")
