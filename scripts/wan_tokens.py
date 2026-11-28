"""PLAN-video S1 inputs, path (b): raw-view camera tokens for every tick, from the guide's OWN depth pass (Astra item 1).

Shared pass: the guide's depth network (the checkpoint recorded in the guide provenance, no fallback; weights sha checked
against the guide's) on the guide's exact preprocessing (mrv.guide.load_views at depth_res_hw = 294x518, JPEG draft +
bilinear, ImageNet normalisation, fp16) is run ONCE per tick with output_hidden_states=True; tokens = backbone.layernorm(last
hidden state) without CLS (21 x 37 = 777 x 384 per view). The same forward's predicted_depth is compared with
mrv.guide.DepthNet.raw on the first batch of each drive (recorded: "shared_pass_max_abs_depth"), so live inference needs one
encoder pass per view for depth AND tokens. A drive is marked complete only with a non-empty depth-weight hash equal to the
guide's and a finite shared-depth difference <= 1e-3 m (S1b item 1). (Offline, the guide builder and this builder still run
the network twice; the comparison is what binds the two passes.)
Rays: per tick and camera from the guide file's K [n,6,3,3] (1600x900 px) and cam2tick [n,6,4,4] (each camera at its own capture
pose, moved into the tick frame), patch centres of the 294x518 grid, Pluecker (d, o x d) in the chase-camera frame
(mrv.guide.chase_cam_to_ego with the guide's chase_dz). scripts/wan_token_check.py verifies them on a pinned tick.
Guide intake: mrv.wan_cond.check_guide (complete sidecar, camera-only, underlay, every tick built, ticks / hard == enc) and
one guide version for every drive of the run.

Per drive -> <out>/<scene>.pt {"ticks", "hard", "tokens" fp16 [n,6,777,384] (zeros on hard ticks), "rays" fp32 [n,6,777,6],
"grid" (21, 37), "guide_version", "encoder", ...} + <scene>.json sidecar {sha256 of the .pt, guide / enc / manifest hashes,
weights sha, code sha, guide_version, complete}. An existing drive is skipped ONLY if its sidecar and every upstream hash still
match (else error; --rebuild re-encodes it). Disk: 3.7 MB per tick (~78 GB for the 21.3k train ticks).
  python scripts/wan_tokens.py --splits train,dev --out data/wan_tok_v1 --shard i/n
"""
import argparse, hashlib, json, math, os, time
import numpy as np
import torch
from mrv import guide as GD
from mrv import wan_cond as WC
from mrv import wan_plan as WP

p = argparse.ArgumentParser()
p.add_argument("--splits", default="train,dev")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--manifest", default="data/temporal_manifest.json")
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--guides", default="data/guides_v1")
p.add_argument("--dataroot", default="data/nuscenes_full")
p.add_argument("--out", default="data/wan_tok_v1")
p.add_argument("--ticks_per_batch", type=int, default=8)
p.add_argument("--rebuild", action="store_true")
p.add_argument("--shard", default="0/1")
a = p.parse_args()
dev = "cuda"
os.makedirs(a.out, exist_ok=True)
CODE_SHA = WC.code_sha(__file__, "mrv/wan_cond.py", "mrv/wan_plan.py", "mrv/guide.py")
man = json.load(open(a.manifest))
drives = WP.check_split_membership(GD.drive_list(a.splits, a.lists), json.load(open(a.lists)))  # explicit ids / files too; sealed refused
SHARED_TOL = 1e-3  # metres: same weights, same input, same dtype -> the shared pass must reproduce the guide's depth call
si, sn = map(int, a.shard.split("/"))
mine = drives[si::sn]


def upstream(sc):
    E_ = torch.load(f"{a.enc}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True)
    return {"guide_sha": WC.file_sha(f"{a.guides}/{sc}.pt"), "enc_sha": WC.file_sha(f"{a.enc}/{sc}.pt"),
            "manifest_sha": hashlib.sha256(json.dumps([r["files"] for r in man[sc]]).encode()).hexdigest(), "code_sha": CODE_SHA}, E_


todo, VERSION = [], None
for sc in mine:
    side = json.load(open(f"{a.guides}/{sc}.json"))
    Gm = torch.load(f"{a.guides}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True)
    up, E = upstream(sc)
    VERSION = WC.check_guide(f"{a.guides}/{sc}.pt", Gm, side, E["ticks"].tolist(), E["hard"].tolist(), sc, VERSION)
    f = f"{a.out}/{sc}.pt"
    if os.path.exists(f) and not a.rebuild:
        if not os.path.exists(f"{a.out}/{sc}.json"):
            raise SystemExit(f"{f} exists without a sidecar: --rebuild")
        sj = json.load(open(f"{a.out}/{sc}.json"))
        bad = {k: (sj.get(k), v) for k, v in {**up, "guide_version": VERSION}.items() if sj.get(k) != v}
        if not sj.get("complete") or not sj.get("shared_pass_ok") or not sj.get("weights_sha") or WC.file_sha(f) != sj.get("sha256") or bad:
            raise SystemExit(f"{f}: stale or corrupt token cache ({list(bad)[:4]}); rerun with --rebuild")
        print("valid, skip", sc, flush=True); continue
    todo.append(sc)
print(f"{len(todo)} drives to encode in shard {a.shard}; guide version {VERSION}", flush=True)
if not todo:
    raise SystemExit(0)
if tuple(VERSION["depth_res_hw"]) != WC.DEPTH_HW:
    raise SystemExit(f"guide depth resolution {VERSION['depth_res_hw']} != token grid source {WC.DEPTH_HW}")
DN = GD.DepthNet(VERSION["depth_model_id"], device=dev, dtype=torch.float16, allow_fallback=False)
if not VERSION["depth_weights_sha"] or not DN.info.get("weights_sha") or DN.info["weights_sha"] != VERSION["depth_weights_sha"]:
    raise SystemExit(f"depth weights must be hashed and equal: loaded {DN.info.get('weights_sha')!r}, guide {VERSION['depth_weights_sha']!r}")
model, backbone = DN.model, DN.model.backbone
H, W = WC.DEPTH_HW
CH = GD.chase_cam_to_ego(float(VERSION["chase_dz"] or 0.0))


@torch.no_grad()
def encode(views):
    """uint8 [6k, H, W, 3] -> tokens [6k, 777, 384] fp16, predicted depth [6k, H', W']"""
    x = torch.as_tensor(views).to(dev).permute(0, 3, 1, 2).contiguous()
    y = (x.float() / 255 - DN.mean) / DN.std
    out = model(pixel_values=y.to(DN.dtype), output_hidden_states=True)
    hs = out.hidden_states[-1]
    tok = backbone.layernorm(hs)[:, 1:]
    assert tok.shape[1] == WC.TOK_P and tok.shape[2] == WC.TOK_DIM, tok.shape
    return tok.half().cpu(), x, out.predicted_depth


for sc in todo:
    t0 = time.time()
    up, E = upstream(sc)
    Gm = torch.load(f"{a.guides}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True)
    rows = man[sc]; ticks = E["ticks"].tolist(); hard = E["hard"].tolist(); n = len(ticks)
    assert [r["tick"] for r in rows] == ticks, f"{sc}: manifest ticks != enc ticks"
    toks = torch.zeros(n, 6, WC.TOK_P, WC.TOK_DIM, dtype=torch.float16)
    rays = torch.from_numpy(np.stack([WC.token_rays(Gm["K"][i].numpy(), Gm["cam2tick"][i].numpy(), CH) for i in range(n)]))
    live = [i for i in range(n) if not hard[i]]; shared = None
    for b in range(0, len(live), a.ticks_per_batch):
        idx = live[b:b + a.ticks_per_batch]
        views = np.concatenate([GD.load_views(a.dataroot, rows[i]["files"], (W, H)) for i in idx])
        tk, x, pd = encode(views)
        toks[idx] = tk.view(len(idx), 6, WC.TOK_P, WC.TOK_DIM)
        if shared is None:  # the guide's own depth call on the same input must equal this pass's depth output
            ref = DN.raw(x)
            pdx = pd[:, 0] if pd.dim() == 4 else pd
            if pdx.shape[-2:] != ref.shape[-2:]:
                pdx = torch.nn.functional.interpolate(pdx[:, None].float(), size=ref.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            shared = float((pdx.float() - ref).abs().max())
    if shared is None or not math.isfinite(shared) or shared > SHARED_TOL:  # nothing is written for a failing drive
        raise SystemExit(f"{sc}: shared-pass depth check failed ({shared!r} vs tolerance {SHARED_TOL} m): tokens not from the guide's depth pass")
    out = {"scene": sc, "ticks": E["ticks"].clone(), "hard": E["hard"].clone(), "tokens": toks, "rays": rays, "grid": WC.TOK_GRID,
           "encoder": VERSION["depth_model_id"], "input_hw": WC.DEPTH_HW, "guide_version": VERSION,
           "tokens_from": "depth forward (output_hidden_states), backbone.layernorm(last), CLS dropped", "shared_pass_max_abs_depth": shared}
    f = f"{a.out}/{sc}.pt"
    torch.save(out, f + ".tmp"); os.replace(f + ".tmp", f)
    json.dump({"scene": sc, "sha256": WC.file_sha(f), **up, "guide_version": VERSION, "weights_sha": DN.info["weights_sha"],
               "ticks": n, "encoded": len(live), "shared_pass_max_abs_depth": shared, "shared_pass_tol_m": SHARED_TOL,
               "shared_pass_ok": True, "complete": True}, open(f"{a.out}/{sc}.json.tmp", "w"))
    os.replace(f"{a.out}/{sc}.json.tmp", f"{a.out}/{sc}.json")
    print(sc, n, "ticks", len(live), "encoded", f"shared-pass depth diff {shared:.3g}", f"{time.time() - t0:.0f}s", flush=True)
print("WAN_TOKENS_DONE", a.shard)
