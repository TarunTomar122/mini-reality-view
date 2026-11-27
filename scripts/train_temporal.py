"""Temporal arms A / B / C (council/PLAN-temporal.md, signed v5).

  python scripts/train_temporal.py --arm A --init <E-winner ckpt> --out runs/tmpA
  python scripts/train_temporal.py --arm B --init <E-winner ckpt> --out runs/tmpB
  python scripts/train_temporal.py --arm C --init runs/tmpB/ckpt_1000.pt --cache data/rollcache --out runs/tmpC

Batch 16 = 12 dense rows (a 10 Hz tick of a train drive, with its predecessor tick in the same sequence) + 4 keyframe
singles (data/lat512_tpv + data/enc_full, all train drives except the 6 dev drives). The draws (rows, timesteps, noise,
camera dropout, history noise) come from fixed seeded streams, so A, B and C see the same targets.
  A: no history input at all (control).
  B: a zero-initialised conv beside the frozen conv_in reads 5 channels: the previous tick's key latent corrupted as
     sqrt(1 - s^2) z + s eps (s ~ U[0, 0.7]) plus a constant plane holding s. Singles and 10% of dense rows get no
     history (zeros, plane 1). Both CFG passes see the same history at inference.
  C: B forked at 1,000 updates; the 6 dense rows per batch drawn from the 128 preselected clips (all arms draw them,
     so targets are identical) use the model's own cached frame (scripts/temporal_ccache.py; corrupted the same way)
     instead of the previous key. Caches from snapshots at 1,000 (B), 2,500 and 4,000 (C); C stops at 2,500 / 4,000
     for the cache to be rebuilt and is then resumed with --cache pointing at the new folder.
LoRA + adapter LR 5e-5 (500 warm-up), history conv 1e-4; 10% camera dropout; style 2; epsilon loss; bf16.
"""
import argparse, glob, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig
from mrv.sd import CameraAdapter, LORA_TARGETS

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True, choices=["A", "B", "C"])
p.add_argument("--init", required=True)
p.add_argument("--out", required=True)
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--singles", default="data/lat512_tpv_lit", help="keyframe singles: lit keys (same lighting as the dense 10 Hz keys)")
p.add_argument("--tokens", default="data/enc_full")
p.add_argument("--rays", default="data/calib_rays.pt")
p.add_argument("--cache", default="", help="arm C: folder of cached own rollouts (<scene>.pt {ticks, lat})")
p.add_argument("--cache_step", type=int, default=-1, help="arm C: the snapshot step the cache must come from (asserted)")
p.add_argument("--pool_list", default="", help="long bet: file of drives whose WHOLE-DRIVE own rollouts form the own-history pool; the 6 'own' rows per batch are drawn from ALL dense rows of these drives (for every arm, so targets stay identical); replaces the 128-clip pool")
p.add_argument("--cache_snapshot", default="", help="arm C: the snapshot checkpoint path the cache must come from (asserted)")
p.add_argument("--cache_root", default="", help="live own cache (council/drift-s2): folder of own_<N>/ subfolders, each complete when own_<N>/DONE exists; every 200 updates the newest complete one with N <= step is validated and swapped in; training waits if the cache in use is more than --max_stale updates old")
p.add_argument("--max_stale", type=int, default=4000)
p.add_argument("--refresh_every", type=int, default=0, help="live cache: after saving ckpt_<N> at multiples of this, submit the refresh rollouts for own_<N> (slurm/own_refresh.sh)")
p.add_argument("--true_s_min", type=float, default=0.0, help="drift lever 2 (council/drift-r3): arm C rows 6-11 (true previous key) use s' = true_s_min + s * (0.7 - true_s_min) / 0.7 from the same draw")
p.add_argument("--true_drop", type=float, default=0.0, help="drift lever 2: extra history dropout on arm C rows 6-11, from a separate seeded stream (A / B draws unchanged)")
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--save_at", default="1000,2000,2500,4000,6000")
p.add_argument("--clips", default="data/temporal_cclips.json", help="the 128 preselected eight-tick train clips (arm C's own-rollout rows); made once, seeded")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev, AMP = "cuda", torch.bfloat16
SD = "sd-legacy/stable-diffusion-v1-5"
L = json.load(open(a.lists)); TRAIN = set(L["train"]); DEV = set(L["dev"])
RAYS = torch.load(a.rays)

# ---------- data ----------
dense = []  # (scene, i): rows i whose predecessor i-1 is the previous tick in the same sequence, both usable
E = {}
for f in sorted(glob.glob(f"{a.enc}/*.pt")):
    sc = os.path.basename(f)[:-3]
    if sc not in TRAIN:
        continue
    e = torch.load(f); E[sc] = e
    ok, seq, ticks = e["ok"], e["seq"], e["ticks"]
    for i in range(1, len(ticks)):
        if ok[i] and ok[i - 1] and seq[i] == seq[i - 1] and ticks[i] == ticks[i - 1] + 1:
            dense.append((sc, i))
# 128 preselected eight-tick clips (ticks st..st+7 consecutive and usable). Half of every batch's dense rows (6 of 12)
# come from these clips' 7 history rows, for ALL arms, so A / B / C see the same targets and C can replace exactly
# those rows' history with its own cached frames (plan: "for half of the history rows use the cached own frame").
DSET = set(dense)
if not os.path.exists(a.clips):
    starts = [(sc, i - 1) for (sc, i) in dense if all((sc, i + k) in DSET for k in range(7))]
    rr = random.Random(12345); rr.shuffle(starts); clips = []
    for sc, st in starts:  # no overlapping clips
        if all(not (c[0] == sc and abs(c[1] - st) < 8) for c in clips):
            clips.append((sc, st))
        if len(clips) == 128:
            break
    tmp = a.clips + ".tmp"; json.dump({"clips": [[sc, int(st), int(E[sc]["ticks"][st])] for sc, st in clips]}, open(tmp, "w")); os.replace(tmp, a.clips)
CLIPS = [(c[0], c[1]) for c in json.load(open(a.clips))["clips"]]
clip_rows = [(sc, st + k) for sc, st in CLIPS for k in range(1, 8)]
assert all(r_ in DSET for r_ in clip_rows), "clip rows must be dense rows"
POOL = None
if a.pool_list:  # long bet: whole-drive pool
    POOL = sorted(set(open(a.pool_list).read().split()))
    if not set(POOL) <= set(E):
        raise SystemExit(f"pool drives not in the train set: {sorted(set(POOL) - set(E))}")
    clip_rows = [(sc, i) for (sc, i) in dense if sc in set(POOL)]
    print(f"own-history pool: {len(POOL)} drives, {len(clip_rows)} dense rows (whole drives)", flush=True)
singles = []  # keyframe rows: (tokens, lat, scene)
from nuscenes.utils.splits import train as NTRAIN
for f in sorted(glob.glob(f"{a.singles}/*.pt")):
    sc = os.path.basename(f)[:-3]
    if sc not in set(NTRAIN) or sc in DEV or not os.path.exists(f"{a.tokens}/{sc}.pt"):
        continue
    k = torch.load(f); e = torch.load(f"{a.tokens}/{sc}.pt")
    pos = {t: i for i, t in enumerate(e["sample_tokens"])}
    for j, t in enumerate(k["sample_tokens"]):
        singles.append((e["tokens"][pos[t]], k["lat"][j], sc))
def load_cache(folder, want_step, want_snap):
    """validated whole-drive own cache -> {scene: {tick: latent}}; raises SystemExit on any mismatch"""
    cache, cf, steps_, snaps_ = {}, glob.glob(f"{folder}/*.pt"), set(), set()
    if POOL is not None:  # whole-drive rollout files are named <scene>_s<seed>.pt
        cf = [f for f in cf if os.path.basename(f).rsplit("_s", 1)[0] in set(POOL)]
        got_ = [os.path.basename(f).rsplit("_s", 1)[0] for f in cf]
        if set(got_) != set(POOL) or len(got_) != len(set(got_)):
            raise SystemExit(f"cache {folder} must hold exactly one rollout per pool drive (missing {sorted(set(POOL) - set(got_))})")
    else:
        clip_sc = {c_[0] for c_ in json.load(open(a.clips))["clips"]}
        if {os.path.basename(f)[:-3] for f in cf} != clip_sc:
            raise SystemExit(f"cache {folder} does not cover all clip drives")
    for f in cf:
        c = torch.load(f); sc = os.path.basename(f)[:-3] if POOL is None else os.path.basename(f).rsplit("_s", 1)[0]
        steps_.add(c.get("snapshot_step", -1)); snaps_.add(c.get("snapshot"))
        if c.get("method", "hist") != "hist":
            raise SystemExit(f"cache file {f} made with method {c.get('method')}")
        cache[sc] = {int(t): z for t, z in zip(c["ticks"].tolist(), c["lat"])}
    if steps_ != {want_step}:
        raise SystemExit(f"cache {folder} made from snapshot steps {steps_}, expected {want_step}")
    if not want_snap or snaps_ != {want_snap}:
        raise SystemExit(f"cache snapshot {snaps_} != expected {want_snap}")
    if POOL is not None:  # every pool row's predecessor must have an own frame (non-hard ticks are rendered)
        for sc_, i_ in clip_rows:
            if int(E[sc_]["ticks"][i_ - 1]) not in cache[sc_]:
                raise SystemExit(f"cache {folder} misses {sc_} tick {int(E[sc_]['ticks'][i_ - 1])}")
    else:
        for sc_, st_, t0_ in json.load(open(a.clips))["clips"]:  # every tick of every clip must have an own frame
            for q_ in range(8):
                if int(E[sc_]["ticks"][st_ + q_]) not in cache[sc_]:
                    raise SystemExit(f"cache {folder} misses {sc_} tick {int(E[sc_]['ticks'][st_ + q_])}")
    return cache


CACHE, CACHE_STEP = {}, -1
if a.arm == "C":
    CACHE = load_cache(a.cache, a.cache_step, a.cache_snapshot); CACHE_STEP = a.cache_step
print(f"arm {a.arm}: dense rows {len(dense)} from {len(E)} drives, singles {len(singles)}, cached own frames {sum(len(v) for v in CACHE.values())}", flush=True)

# ---------- model ----------
unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet")
unet.requires_grad_(False)
ck0 = torch.load(a.init, map_location="cpu")
r = ck0["args"]["rank"]
unet.add_adapter(LoraConfig(r=r, lora_alpha=r, init_lora_weights="gaussian", target_modules=LORA_TARGETS))
unet.load_state_dict(ck0["lora"], strict=False)
adapter = CameraAdapter(n_styles=ck0["args"].get("n_styles", 3))
adapter.load_state_dict(ck0["adapter"])


class HistConvIn(nn.Module):
    """frozen conv_in + zero-initialised conv on [history latent (4), noise-level plane (1)]"""
    def __init__(s, base):
        super().__init__(); s.base = base; s.hist = nn.Conv2d(5, base.out_channels, 3, padding=1)
        nn.init.zeros_(s.hist.weight); nn.init.zeros_(s.hist.bias); s.h = None
    def forward(s, x):
        y = s.base(x)
        return y if s.h is None else y + s.hist(s.h.to(y.dtype))


hist = None
if a.arm in ("B", "C"):
    unet.conv_in = HistConvIn(unet.conv_in); hist = unet.conv_in.hist
    if "hist" in ck0:
        hist.load_state_dict(ck0["hist"]); print("history conv loaded from init", flush=True)
    hist.requires_grad_(True)
for n_, p_ in unet.named_parameters():
    if "lora" in n_:
        p_.requires_grad_(True)
unet.to(dev); adapter.to(dev)
for p_ in unet.parameters():
    if not p_.requires_grad:
        p_.data = p_.data.to(AMP)
main_params = [p_ for n_, p_ in unet.named_parameters() if p_.requires_grad and ".hist." not in n_] + list(adapter.parameters())
groups = [{"params": main_params, "lr": 5e-5, "base": 5e-5}]
if hist is not None:
    groups.append({"params": list(hist.parameters()), "lr": 1e-4, "base": 1e-4})
opt = torch.optim.AdamW(groups, weight_decay=0.0)
start = 0
if a.arm == "C" and "opt" in ck0 and "hist" in ck0:  # fork: keep B's optimiser state and step count
    opt.load_state_dict(ck0["opt"]); start = ck0["step"]; print("forked from B at", start, flush=True)  # must succeed (no silent restart at 0)
if a.arm == "C" and start == 0 and not os.path.exists(f"{a.out}/ckpt.pt"):
    raise SystemExit("arm C must fork from B's checkpoint with optimiser state (--init runs/tmpB/ckpt_1000.pt)")
ck = f"{a.out}/ckpt.pt"
if os.path.exists(ck):
    s_ = torch.load(ck, map_location="cpu")
    adapter.load_state_dict(s_["adapter"]); unet.load_state_dict(s_["lora"], strict=False)
    if hist is not None:
        hist.load_state_dict(s_["hist"])
    opt.load_state_dict(s_["opt"]); start = s_["step"]; print("resumed at", start, flush=True)
sched = DDPMScheduler.from_pretrained(SD, subfolder="scheduler"); acp = sched.alphas_cumprod.to(dev)

# ---------- identical draws ----------
G = torch.Generator().manual_seed(a.seed)
def draw(step):
    g = torch.Generator().manual_seed(a.seed * 100003 + step)  # same rows/noise for every arm at every step
    dc = torch.randint(0, len(clip_rows), (6,), generator=g); da = torch.randint(0, len(dense), (6,), generator=g)
    rows = [clip_rows[k] for k in dc.tolist()] + [dense[k] for k in da.tolist()]  # rows 0-5: clip rows (C: own history)
    si = torch.randint(0, len(singles), (4,), generator=g)
    t = torch.randint(0, 1000, (16,), generator=g); eps = torch.randn(16, 4, 32, 64, generator=g)
    drop = torch.rand(16, generator=g) < 0.1
    s = torch.rand(12, generator=g) * 0.7; heps = torch.randn(12, 4, 32, 64, generator=g)
    nohist = torch.rand(12, generator=g) < 0.1
    return rows, si, t, eps, drop, s, heps, nohist


N_OWN = [0, 0]
N_L2 = [0]  # lever 2: true-key rows with history dropped


def batch(step):
    rows, si, t, eps, drop, s, heps, nohist = draw(step)
    if a.arm == "C" and (a.true_s_min > 0 or a.true_drop > 0):  # lever 2: harder history on the true-key rows only
        s = s.clone(); nohist = nohist.clone()
        s[6:12] = a.true_s_min + s[6:12] * (0.7 - a.true_s_min) / 0.7
        g2 = torch.Generator().manual_seed(a.seed * 7919 + step + 10 ** 9)
        nohist[6:12] = nohist[6:12] | (torch.rand(6, generator=g2) < a.true_drop)
        N_L2[0] += int(nohist[6:12].sum())
    tok, lat, ray, H = [], [], [], []
    for j, (sc, i) in enumerate(rows):
        e = E[sc]
        tok.append(e["tokens"][i]); lat.append(e["lat"][i]); ray.append(RAYS[sc])
        prev = e["lat"][i - 1].float()
        if a.arm == "C" and j < 6:
            own_ = CACHE.get(sc, {}).get(int(e["ticks"][i - 1]))
            assert own_ is not None, f"arm C: no cached own frame for {sc} tick {int(e['ticks'][i - 1])}"
            prev = own_.float(); N_OWN[0] += 1
        N_OWN[1] += 1
        sj = float(s[j])
        h = torch.cat([(1 - sj ** 2) ** 0.5 * prev + sj * heps[j], torch.full((1, 32, 64), sj)])
        H.append(torch.cat([torch.zeros(4, 32, 64), torch.ones(1, 32, 64)]) if nohist[j] else h)
    for k in si.tolist():
        tk, lt, sc = singles[k]; tok.append(tk); lat.append(lt); ray.append(RAYS[sc])
        H.append(torch.cat([torch.zeros(4, 32, 64), torch.ones(1, 32, 64)]))
    f = lambda xs: torch.stack([x.float() for x in xs]).to(dev)
    return f(tok), f(lat), f(ray), f(H), t.to(dev), eps.to(dev), drop.to(dev)


def save(step_done, path):
    d = {"adapter": adapter.state_dict(), "lora": {k: v for k, v in unet.state_dict().items() if "lora" in k},
         "opt": opt.state_dict(), "step": step_done, "args": {**vars(a), "rank": r, "n_styles": ck0["args"].get("n_styles", 3)}}
    if hist is not None:
        d["hist"] = hist.state_dict()
    torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)


style = torch.full((16,), 2, dtype=torch.long, device=dev)
unet.train(); adapter.train(); t0 = time.time()
log = open(f"{a.out}/log.jsonl", "a")
SAVE = set(int(x) for x in a.save_at.split(",") if x)
def live_caches():
    """complete live caches: [(N, folder)] with own_<N>/DONE, sorted by N"""
    out = []
    for d_ in glob.glob(f"{a.cache_root}/own_*"):
        try:
            n_ = int(os.path.basename(d_)[4:])
        except ValueError:
            continue
        if os.path.exists(f"{d_}/DONE"):
            out.append((n_, d_))
    return sorted(out)


BAD = set()


def maybe_reload(step):
    global CACHE, CACHE_STEP
    while True:
        cand = [(n_, d_) for n_, d_ in live_caches() if CACHE_STEP < n_ <= step and d_ not in BAD]
        if cand:
            n_, d_ = cand[-1]
            try:  # load into a new dict, validate, then swap; a bad cache is logged and skipped (the old one stays)
                snap_ = json.load(open(f"{d_}/DONE"))["snapshot"]
                t_ = time.time(); new_ = load_cache(d_, n_, snap_)
                CACHE, CACHE_STEP = new_, n_
                print(f"live cache: step {step} now uses {d_} (snapshot {snap_}, age {step - n_}), loaded in {time.time() - t_:.0f} s", flush=True)
            except (SystemExit, Exception) as e_:
                BAD.add(d_); print(f"live cache: REJECTED {d_}: {e_}", flush=True)
                continue
        if step - CACHE_STEP <= a.max_stale:
            return
        print(f"live cache: waiting at step {step}: cache {CACHE_STEP} is {step - CACHE_STEP} > {a.max_stale} updates old", flush=True)
        time.sleep(60)


def submit_refresh(n_):
    """after ckpt_<n_> is saved: submit the 12 pool rollout shards for own_<n_> plus the DONE writer (slurm/own_refresh.sh)"""
    import subprocess
    r_ = subprocess.run(["bash", "slurm/own_refresh.sh", f"{a.out}/ckpt_{n_}.pt", f"{a.cache_root}/own_{n_}"], capture_output=True, text=True)
    print(f"refresh own_{n_}: rc {r_.returncode} {r_.stdout.strip()[-300:]} {r_.stderr.strip()[-300:]}", flush=True)


for step in range(start, a.steps):
    if a.cache_root and a.arm == "C" and step % 200 == 0:
        maybe_reload(step)
    tok, x0, ray, H, t, eps, drop = batch(step)
    ab = acp[t].view(-1, 1, 1, 1); xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    for g_ in opt.param_groups:
        g_["lr"] = g_["base"] * min(1, (step + 1) / 500)
    if hist is not None:
        unet.conv_in.h = H
    with torch.autocast("cuda", dtype=AMP):
        ctx = adapter(tok, ray, style, drop)
        pred = unet(xt, t, encoder_hidden_states=ctx).sample
    loss = ((pred.float() - eps) ** 2).mean()
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_([q for g_ in opt.param_groups for q in g_["params"]], 1.0)
    opt.step()
    if step % 100 == 0:
        rec = {"step": step, "loss": round(loss.item(), 4), "gn": round(gn.item(), 3), "min": round((time.time() - t0) / 60, 1),
               "own_share": round(N_OWN[0] / max(1, N_OWN[1]), 3), "cache_step": CACHE_STEP, "stale": step - CACHE_STEP if CACHE_STEP >= 0 else None,
               "true_drop_share": round(N_L2[0] / max(1, 6 * (step - start + 1)), 3)}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    if (step + 1) % 1000 == 0 or step + 1 == a.steps:
        save(step + 1, ck)
    if step + 1 in SAVE:
        save(step + 1, f"{a.out}/ckpt_{step + 1}.pt")
    if a.refresh_every and a.cache_root and (step + 1) % a.refresh_every == 0 and step + 1 < a.steps:
        if step + 1 not in SAVE:
            save(step + 1, f"{a.out}/ckpt_{step + 1}.pt")
        submit_refresh(step + 1)
print("TEMPORAL_TRAIN_DONE", a.arm)
