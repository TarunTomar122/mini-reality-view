"""Experiment W (council/PLAN-bev.md): cache BEVFormer-base BEV features for the renderer, and the sanity gate.

Run inside envs/bev from bev/BEVFormer (slurm/bev_extract.sh does this):
  python ../../scripts/bev_extract.py --split val --eval        # val features + official nuScenes detection metrics
  python ../../scripts/bev_extract.py --split train --shard 0/4 # train features, one of 4 shards

The model runs recurrently through each scene in time order (2 Hz keyframes, memory reset at scene start), exactly
as in BEVFormer's own test. After each frame we take its BEV embedding (200 x 200 x 256, 0.512 m cells over
+-51.2 m in the LIDAR_TOP frame) and resample it onto our ego-frame grid in front of the car: x (forward)
-8 .. +41 m, y (left) -16.4 .. +16.4 m, sampled at 96 x 64 points and 2x2 average-pooled -> 48 x 32 cells of
~1.02 m, 256 channels, fp16. Saved as data/bev/<scene>.pt {bev (n,256,32,48), sample_tokens, x_range, y_range}:
row 0 is the left edge (+16.4 m), rows go to the right; column 0 is 8 m behind the car, columns go forward.
"""
import argparse, importlib, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--config", default="projects/configs/bevformer/bevformer_base.py")
p.add_argument("--ckpt", default="../ckpts/bevformer_r101_dcn_24ep.pth")
p.add_argument("--split", default="val", choices=["train", "val"])
p.add_argument("--shard", default="0/1")
p.add_argument("--eval", action="store_true", help="also run the official nuScenes detection evaluation (val, all scenes)")
p.add_argument("--out", default="../../data/bev")
p.add_argument("--limit_scenes", type=int, default=0)
p.add_argument("--scenes_file", default="", help="only these scenes (one name per line)")
p.add_argument("--boxes_out", default="", help="also save detected 3D boxes per sample token to this JSON (lidar frame)")
p.add_argument("--no_feats", action="store_true", help="do not save BEV feature caches (boxes only)")
p.add_argument("--zero_canbus", action="store_true", help="P2 (PLAN-carmap): zero the can_bus ego-motion input (no non-camera motion)")
p.add_argument("--no_prev", action="store_true", help="P2: no previous BEV grid (video_test_mode off)")
a = p.parse_args()

sys.path.insert(0, os.getcwd())
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from pyquaternion import Quaternion

cfg = Config.fromfile(a.config)
importlib.import_module(cfg.plugin_dir.rstrip("/").replace("/", "."))
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

X0, X1, Y0, Y1, NX, NY = -8.0, 41.0, -16.4, 16.4, 48, 32
PC = cfg.point_cloud_range  # [-51.2, -51.2, -5, 51.2, 51.2, 3]
BH, BW = cfg.bev_h_, cfg.bev_w_

tcfg = cfg.data.test
tcfg.test_mode = True
tcfg.ann_file = cfg.data_root + f"nuscenes_infos_temporal_{a.split}.pkl"
ds = build_dataset(tcfg)
scenes = json.load(open(os.path.join(cfg.data_root, "v1.0-trainval", "scene.json")))
scene_name = {s["token"]: s["name"] for s in scenes}
names = sorted({scene_name[i["scene_token"]] for i in ds.data_infos})
si, sn = map(int, a.shard.split("/"))
keep = set(names[si::sn][: a.limit_scenes or None]) if not a.eval else set(names[: a.limit_scenes or None])
if a.scenes_file:  # the shard split applies to the listed scenes
    keep = set(sorted(set(open(a.scenes_file).read().split()) & set(names))[si::sn])
# keep BEVFormer's own (time-sorted) order; drop scenes of other shards and scenes already done
os.makedirs(a.out, exist_ok=True)
done = {n for n in keep if os.path.exists(f"{a.out}/{n}.pt")} if not a.eval else set()
ds.data_infos = [i for i in ds.data_infos if scene_name[i["scene_token"]] in keep - done]
ds.flag = np.zeros(len(ds.data_infos), dtype=np.uint8)
print("split", a.split, "scenes", len(keep - done), "frames", len(ds.data_infos), flush=True)
loader = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=4, dist=False, shuffle=False,
                          nonshuffler_sampler=cfg.data.nonshuffler_sampler)

model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, a.ckpt, map_location="cpu")
model = MMDataParallel(model.cuda(), device_ids=[0]).eval()

# ego-frame sample points (96 x 64, z = 0), pooled 2x2 afterwards
xs = np.linspace(X0, X1, 2 * NX + 1)[:-1] + (X1 - X0) / (4 * NX)
ys = np.linspace(Y1, Y0, 2 * NY + 1)[:-1] - (Y1 - Y0) / (4 * NY)  # row 0 = left (+y)
gx, gy = np.meshgrid(xs, ys)
P_ego = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)


def bev_to_grid(bev, info):
    """bev (40000, 1, 256) -> (256, 32, 48) on our ego grid."""
    R = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    t = np.asarray(info["lidar2ego_translation"])
    P_l = (P_ego - t) @ R  # ego -> lidar (R maps lidar -> ego)
    u = (P_l[:, 0] - PC[0]) / (PC[3] - PC[0]) * 2 - 1  # x -> bev column
    v = (P_l[:, 1] - PC[1]) / (PC[4] - PC[1]) * 2 - 1  # y -> bev row
    grid = torch.tensor(np.stack([u, v], 1), dtype=torch.float32, device=bev.device).view(1, 2 * NY, 2 * NX, 2)
    bev = bev.reshape(-1, bev.shape[-1]) if bev.shape[0] == BH * BW or bev.shape[1] == BH * BW else bev
    img = bev.T.reshape(1, -1, BH, BW).float()  # queries are row-major over (y, x)
    out = F.grid_sample(img, grid, mode="bilinear", align_corners=False)
    return F.avg_pool2d(out, 2)[0].half().cpu()


results, buf, boxes = [], {}, {}
with torch.no_grad():
    for i, data in enumerate(loader):
        info = ds.data_infos[i]
        if a.zero_canbus:
            for m_ in data["img_metas"][0].data[0]:
                m_["can_bus"][:] = 0
        if a.no_prev:
            model.module.video_test_mode = False
        res = model(return_loss=False, rescale=True, **data)
        results.extend(res)
        if a.boxes_out:  # detected boxes in the LIDAR_TOP frame: x, y, z(bottom), w, l, h, yaw, vx, vy; score; class
            pb = res[0]["pts_bbox"]; b = pb["boxes_3d"].tensor.cpu().numpy(); sco = pb["scores_3d"].cpu().numpy(); lab = pb["labels_3d"].cpu().numpy()
            m = sco >= 0.2
            boxes[info["token"]] = {"scene": scene_name[info["scene_token"]], "boxes": b[m].round(3).tolist(),
                                    "scores": sco[m].round(3).tolist(), "labels": [cfg.class_names[int(l)] for l in lab[m]],
                                    "lidar2ego_t": list(info["lidar2ego_translation"]), "lidar2ego_r": list(info["lidar2ego_rotation"])}
        bev = model.module.prev_frame_info["prev_bev"]
        sc = scene_name[info["scene_token"]]
        b = buf.setdefault(sc, {"bev": [], "sample_tokens": []})
        b["bev"].append(bev_to_grid(bev, info)); b["sample_tokens"].append(info["token"])
        nxt = ds.data_infos[i + 1]["scene_token"] if i + 1 < len(ds.data_infos) else None
        if nxt != info["scene_token"] and a.no_feats:
            del buf[sc]; print(sc, "boxes", i + 1, "/", len(ds.data_infos), flush=True)
        elif nxt != info["scene_token"]:
            torch.save({"bev": torch.stack(b["bev"]), "sample_tokens": b["sample_tokens"],
                        "x_range": (X0, X1), "y_range": (Y1, Y0)}, f"{a.out}/{sc}.pt")
            del buf[sc]
            print(sc, len(b["sample_tokens"]), "frames", i + 1, "/", len(ds.data_infos), flush=True)
if a.boxes_out:
    json.dump(boxes, open(a.boxes_out, "w"))
    print("BOXES", len(boxes), "frames ->", a.boxes_out, flush=True)
if a.eval:
    print("EVAL", ds.evaluate(results, jsonfile_prefix="../../tmp/bev_eval"), flush=True)
print("BEV_EXTRACT_DONE")
