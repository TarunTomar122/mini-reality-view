"""PLAN-video S1 inputs, path (b): raw-view camera tokens for every tick (mrv/wan_cond.py docstring has the design).

Frozen per-view encoder = the ViT-S/14 backbone of Depth Anything V2-Small (default = mrv/guide.py PRIMARY,
depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf; if the guides are built with the FALLBACK, pass that checkpoint here,
so one encoder pass per view serves depth and tokens at run time). Input: the six nuScenes photos of the tick (data/temporal_manifest.json
rows, CAMS order), resized to 448x252 (bicubic, ImageNet normalisation, as scripts/temporal_encode.py) -> last-layer patch
tokens (layer-normed, CLS dropped), 18 x 32 = 576 UNPOOLED tokens x 384 per view. Hard-break ticks (temporal_encode rule,
taken from the enc file) get zeros.
Rays: per scene, the calibrated cameras (fixed per log, scene's first sample, as scripts/calib.py) give each patch centre's
ray; stored as Pluecker (d, o x d) in the CHASE camera frame (mrv.wan_cond.plucker_chase).

Per drive -> <out>/<scene>.pt {"ticks" (n,), "tokens" fp16 (n, 6, 576, 384), "rays" fp32 (6, 576, 6), "grid" (18, 32),
"encoder", "input_wh" (448, 252)} + <scene>.json sidecar (sha256 of the .pt, counts). Ticks are the enc file's ticks, in order.
Disk: 2.65 MB per tick (~56 GB for the 21.3k train ticks, ~60+ GB with dev); the trainer memory-maps these files.
  python scripts/wan_tokens.py --splits train,dev --out data/wan_tok_v1 --shard i/n     (2080 Ti is fine)
"""
import argparse, hashlib, json, os, time
import numpy as np
import torch
from PIL import Image
from mrv.nusc import CAMS
from mrv.wan_cond import plucker_chase, TOK_ENCODER_ID

p = argparse.ArgumentParser()
p.add_argument("--splits", default="train,dev")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--manifest", default="data/temporal_manifest.json")
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--dataroot", default="data/nuscenes_full")
p.add_argument("--version", default="v1.0-trainval")
p.add_argument("--encoder", default=TOK_ENCODER_ID)
p.add_argument("--out", default="data/wan_tok_v1")
p.add_argument("--ticks_per_batch", type=int, default=8)
p.add_argument("--shard", default="0/1")
a = p.parse_args()
dev = "cuda"
W_IN, H_IN, PS = 448, 252, 14
GH, GW = H_IN // PS, W_IN // PS  # 18, 32
os.makedirs(a.out, exist_ok=True)
L = json.load(open(a.lists)); man = json.load(open(a.manifest))
drives = sorted({d for s in a.splits.split(",") for d in L[s]})
si, sn = map(int, a.shard.split("/"))
todo = [d for k, d in enumerate(drives) if k % sn == si and not os.path.exists(f"{a.out}/{d}.pt")]
print(f"{len(todo)} drives to encode in shard {a.shard}", flush=True)
if not todo:
    raise SystemExit(0)

from transformers import AutoModelForDepthEstimation
from nuscenes.nuscenes import NuScenes
dt = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
model = AutoModelForDepthEstimation.from_pretrained(a.encoder, torch_dtype=dt, local_files_only=True).to(dev).eval()
backbone = model.backbone
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
nusc = NuScenes(a.version, a.dataroot, verbose=False)
by = {s["name"]: s for s in nusc.scene}
from pyquaternion import Quaternion
u = (np.arange(GW) + 0.5) * PS; v = (np.arange(GH) + 0.5) * PS
uu, vv = np.meshgrid(u, v); pix = np.stack([uu.ravel(), vv.ravel(), np.ones(GH * GW)], 1)  # row-major like the tokens


def rays_for(sc):
    s = nusc.get("sample", by[sc]["first_sample_token"]); out = []
    for cam in CAMS:
        cs = nusc.get("calibrated_sensor", nusc.get("sample_data", s["data"][cam])["calibrated_sensor_token"])
        K = np.array(cs["camera_intrinsic"], dtype=np.float64); K[0] *= W_IN / 1600; K[1] *= H_IN / 900
        R = Quaternion(cs["rotation"]).rotation_matrix
        d = (np.linalg.inv(K) @ pix.T).T @ R.T
        out.append(plucker_chase(d / np.linalg.norm(d, axis=1, keepdims=True), np.array(cs["translation"])))
    return torch.tensor(np.stack(out), dtype=torch.float32)  # (6, 576, 6)


@torch.no_grad()
def encode(imgs):
    """list of uint8 HxWx3 (6 * k) -> (6k, 576, 384) fp16"""
    x = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2).float() / 255
    fm = backbone(pixel_values=((x - MEAN) / STD).to(dt)).feature_maps[-1]
    if fm.ndim == 4:  # reshaped variant: (B, C, h, w)
        fm = fm.flatten(2).transpose(1, 2)
    else:  # (B, 1 + N, C): drop CLS
        fm = fm[:, 1:]
    assert fm.shape[1] == GH * GW and fm.shape[2] == 384, fm.shape
    return fm.half().cpu()


for sc in todo:
    t0 = time.time()
    E = torch.load(f"{a.enc}/{sc}.pt", map_location="cpu", weights_only=False)
    rows = man[sc]; ticks = E["ticks"].tolist(); hard = E["hard"].tolist()
    assert [r["tick"] for r in rows] == ticks, f"{sc}: manifest ticks != enc ticks"
    toks = torch.zeros(len(ticks), 6, GH * GW, 384, dtype=torch.float16)
    live = [i for i in range(len(ticks)) if not hard[i]]
    for b in range(0, len(live), a.ticks_per_batch):
        idx = live[b:b + a.ticks_per_batch]
        ims = [np.asarray(Image.open(f"{a.dataroot}/{f}").convert("RGB").resize((W_IN, H_IN), Image.BICUBIC)) for i in idx for f in rows[i]["files"]]
        toks[idx] = encode(ims).view(len(idx), 6, GH * GW, 384)
    out = {"scene": sc, "ticks": E["ticks"].clone(), "tokens": toks, "rays": rays_for(sc), "grid": (GH, GW), "encoder": a.encoder,
           "input_wh": (W_IN, H_IN), "layer": "last (layer-normed), CLS dropped", "hard": E["hard"].clone()}
    f = f"{a.out}/{sc}.pt"
    torch.save(out, f + ".tmp"); os.replace(f + ".tmp", f)
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    json.dump({"scene": sc, "sha256": h.hexdigest(), "ticks": len(ticks), "encoded": len(live), "encoder": a.encoder}, open(f"{a.out}/{sc}.json", "w"))
    print(sc, len(ticks), "ticks", len(live), "encoded", f"{time.time() - t0:.0f}s", flush=True)
print("WAN_TOKENS_DONE", a.shard)
