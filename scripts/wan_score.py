"""F3 (council/s1-timing-r1-CONVERGED.md item 4): signed RGB scoring OFF the slice, on 2080 Ti jobs (slurm/wan_score.sh), under its
own ledger runs/wan_score_ledger.jsonl (charged by the allocation's outer beat, kind score_<panel> / score_parity) and reported
separately from the 24 H200-3g slice-hour S1 budget. The scorer itself is unchanged (scripts/score_rgb.py -> the signed
scripts/temporal_metrics.py); this wrapper only enforces where and on what it runs.

  --parity [--shard i/n]   A@20k replay through the RGB path on THIS GPU (score_rgb.py --from_lat; unsharded also --parity_ref:
                           every per-run field must equal the signed metrics). Output <wroot>/score2080/m_A20k_rgbpath[_<i>of<n>].json
                           + .rec.json (device, wall, ticks, rc). The unsharded wall / ticks is the MEASURED 2080 Ti scoring rate.
  --parity_check --shards n  (CPU) sharded == unsharded on the same inputs (same runs, same accepted ticks, same metrics, same
                           input shas), every part on one 2080 Ti device type with the same scorer / adapter / SD VAE shas, the
                           unsharded replay PASS -> <wroot>/score_parity_2080.json {pass, replay_pass, sharded_equal, device,
                           binding (mrv.wan_plan.score_binding), inputs_sha, score_tick_s}. MODE=train and the panel decision
                           require it (mrv.wan_plan.score_parity_ok).
  --panel NAME --what W [--shard i/n]  score one scoring of a rendered panel (W = hybrid | guide | tokens | none | swap_full |
                           swap_postdonor | swap_postrecv) from <wroot>/<NAME>_rgb_manifest.json (the slice's sha manifest; every
                           input must match it), on the parity record's device and code only; output
                           <wroot>/m_<NAME>_<W>_dev.shard<i>of<n>.json. Cap: 1.25 x the measured scoring GPU-h of one panel
                           (mrv.wan_plan.score_cap_h at the parity record's rate); refused at start when the score ledger for
                           this panel has reached it, terminated (exit 10, SCORE_CAP_EXCEEDED) when it is reached while running.
  python scripts/wan_score.py --panel s1_12k --what hybrid --shard 0/3
"""
import argparse, json, os, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrv import wan_plan as WP

p = argparse.ArgumentParser()
p.add_argument("--parity", action="store_true"); p.add_argument("--parity_check", action="store_true")
p.add_argument("--panel", default=""); p.add_argument("--what", default="")
p.add_argument("--shard", default="0/1"); p.add_argument("--shards", type=int, default=3)
p.add_argument("--wroot", default="runs/wroll")
p.add_argument("--a_metrics", default="runs/troll/m_A20k_indep_dev.json"); p.add_argument("--a_roll", default="runs/troll/A20k_indep_dev")
p.add_argument("--ledger", default="runs/wan_score_ledger.jsonl")
p.add_argument("--poll_s", type=float, default=30.0)
a = p.parse_args()
PY = sys.executable
PREC = f"{a.wroot}/score_parity_2080.json"
PDIR = f"{a.wroot}/score2080"
WHAT = {"hybrid": [], "guide": [], "tokens": [], "none": [], "swap_full": ["--lag_ref", a.a_roll], "swap_postdonor": ["--post_swap_only"],
        "swap_postrecv": ["--post_swap_only", "--as_receiver"]}


def run_capped(cmd, kind, cap_h):
    """score_rgb.py in a subprocess; the score ledger (this kind) is polled and the run terminated when it reaches cap_h"""
    L = WP.Ledger(a.ledger, kind, write=not os.environ.get("WAN_LEDGER_OUTER")).start()
    print("RUN", " ".join(cmd), flush=True)
    t0 = time.time()
    with tempfile.TemporaryFile("w+") as fo:
        pr = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, text=True)
        while pr.poll() is None:
            try:
                pr.wait(timeout=a.poll_s)
            except subprocess.TimeoutExpired:
                if cap_h is not None and L.by_kind().get(kind, 0.0) >= cap_h:
                    pr.terminate()
                    try:
                        pr.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        pr.kill()
                    L.stop("budget")
                    print(f"SCORE_CAP_EXCEEDED {kind}: {L.by_kind().get(kind, 0.0):.3f} h >= cap {cap_h:.3f} h", flush=True)
                    sys.exit(10)
        fo.seek(0); out = fo.read()
    L.stop()
    print(out[-4000:], flush=True)
    return pr.returncode, out, round(time.time() - t0, 1)


def device():
    import torch
    return torch.cuda.get_device_name()


if a.parity:  # ------------------------------------------------------------------------------------------- 2080 Ti replay
    dv = device()
    if WP.SCORE_DEVICE not in dv:
        raise SystemExit(f"--parity must run on a {WP.SCORE_DEVICE} (got {dv!r}): slurm/wan_score.sh parity")
    os.makedirs(PDIR, exist_ok=True)
    si, sn = map(int, a.shard.split("/"))
    out = f"{PDIR}/m_A20k_rgbpath.json" if sn == 1 else f"{PDIR}/m_A20k_rgbpath_{si}of{sn}.json"
    cmd = [PY, "scripts/score_rgb.py", "--from_lat", a.a_roll, "--out", out, "--shard", a.shard, "--tag", f"par{si}of{sn}"]
    if sn == 1:
        cmd += ["--parity_ref", a.a_metrics]
    rc, log, wall = run_capped(cmd, "score_parity", None)
    b = json.load(open(out)).get("binding", {}) if os.path.exists(out) else {}
    rec = {"out": out, "shard": a.shard, "rc": rc, "device": dv, "wall_s": wall, "ticks": b.get("ticks"),
           "replay_pass": (sn == 1 and rc == 0 and "RGB_PATH_PARITY PASS" in log), "log": log[-600:], "job": os.environ.get("SLURM_JOB_ID"),
           "when": time.time()}
    json.dump(rec, open(out + ".rec.json", "w"), indent=1)
    print("WAN_SCORE_PARITY_PART", json.dumps({k: v for k, v in rec.items() if k != "log"}))
    raise SystemExit(0 if rc == 0 else 1)

if a.parity_check:  # ------------------------------------------------------------------------------ CPU: sharded == unsharded
    n = a.shards
    uo = f"{PDIR}/m_A20k_rgbpath.json"
    parts = [f"{PDIR}/m_A20k_rgbpath_{i}of{n}.json" for i in range(n)]
    miss = [f for f in [uo] + parts if not (os.path.exists(f) and os.path.exists(f + ".rec.json"))]
    if miss:
        raise SystemExit(f"parity parts missing: {miss}")
    U, UR = json.load(open(uo)), json.load(open(uo + ".rec.json"))
    S = [json.load(open(f)) for f in parts]; SR = [json.load(open(f + ".rec.json")) for f in parts]
    why = []
    if not UR.get("replay_pass"):
        why.append("unsharded A@20k replay not PASS")
    sharded_equal, why_s = WP.compare_sharded(U, S)
    why += why_s
    devs = {j["binding"]["device"] for j in [U] + S}
    if len(devs) != 1 or WP.SCORE_DEVICE not in next(iter(devs)):
        why.append(f"devices {sorted(devs)} (need one {WP.SCORE_DEVICE} type)")
    for k in ("score_rgb", "temporal_metrics", "sd_vae"):
        if len({j["binding"][k] for j in [U] + S}) != 1:
            why.append(f"{k} differs across parts")
    BIND = WP.score_binding(a.a_metrics, a.a_roll)
    for k in ("score_rgb", "temporal_metrics", "sd_vae"):
        if U["binding"][k] != BIND[k]:
            why.append(f"{k} changed since the parity parts were scored")
    rec = {"pass": not why, "replay_pass": bool(UR.get("replay_pass")), "sharded_equal": sharded_equal, "why": why,
           "device": next(iter(devs)) if len(devs) == 1 else sorted(devs), "binding": BIND, "inputs_sha": U["binding"]["inputs_sha"],
           "score_tick_s": round(UR["wall_s"] / max(1, UR["ticks"] or 1), 4), "ticks": UR["ticks"], "unsharded_wall_s": UR["wall_s"],
           "shard_wall_s": [r["wall_s"] for r in SR], "shards": n, "runs": len(U["drives"]),
           "rate_note": "from the A@20k replay (includes the real SD-VAE decode of every tick: an upper bound for RGB scoring)",
           "when": time.time()}
    json.dump(rec, open(PREC, "w"), indent=1)
    print("WAN_SCORE_PARITY", "PASS" if rec["pass"] else "FAIL", json.dumps({k: v for k, v in rec.items() if k != "binding"}))
    raise SystemExit(0 if rec["pass"] else 1)

# ------------------------------------------------------------------------------------------------------------ panel scoring
if not (a.panel and a.what in WHAT):
    raise SystemExit(f"give --parity, --parity_check, or --panel NAME --what one of {sorted(WHAT)}")
ok_p, why_p = WP.score_parity_ok(PREC, WP.score_binding(a.a_metrics, a.a_roll))
if not ok_p:
    raise SystemExit(f"no bound 2080 Ti scorer-parity PASS ({PREC}: {why_p}): run slurm/wan_score.sh parity first")
PR = json.load(open(PREC))
dv = device()
if dv != PR["device"]:
    raise SystemExit(f"scoring device {dv!r} != the parity record's {PR['device']!r}")
MAN = f"{a.wroot}/{a.panel}_rgb_manifest.json"
if not os.path.exists(MAN):
    raise SystemExit(f"{MAN} missing: render the panel on the slice first (scripts/wan_panel.py --phase render)")
M = json.load(open(MAN))
side = MAN + ".sha256"
if not os.path.exists(side) or open(side).read().split()[0] != WP._sha_file(MAN):
    raise SystemExit(f"{MAN} does not match {side}: the RGB manifest changed after rendering")
src = M["dirs"]["swap" if a.what.startswith("swap") else a.what]
DTX = int(M["dev_ticks_x_seeds"])
CAP = WP.score_cap_h(DTX, float(PR["score_tick_s"]))
KIND = f"score_{a.panel}"
spent = WP.Ledger(a.ledger, KIND, write=False).by_kind().get(KIND, 0.0)
print(f"score ledger {KIND}: {spent:.3f} h of cap {CAP:.3f} h (1.25 x {WP.panel_score_h(DTX, float(PR['score_tick_s'])):.3f} h measured "
      f"at {PR['score_tick_s']} s / tick on {PR['device']})", flush=True)
if spent >= CAP:
    print(f"SCORE_CAP_EXCEEDED {KIND}: {spent:.3f} h >= cap {CAP:.3f} h: refused", flush=True)
    sys.exit(10)
si, sn = map(int, a.shard.split("/"))
out = f"{a.wroot}/m_{a.panel}_{a.what}_dev.shard{si}of{sn}.json"
cmd = [PY, "scripts/score_rgb.py", "--rgb", src, "--rgb_manifest", MAN, "--shard", a.shard, "--out", out,
       "--tag", f"{a.what}{si}of{sn}"] + WHAT[a.what]
rc, log, wall = run_capped(cmd, KIND, CAP)
rec = {"panel": a.panel, "what": a.what, "shard": a.shard, "out": out, "rc": rc, "wall_s": wall, "device": dv, "cap_h": CAP,
       "score_ledger_h": WP.Ledger(a.ledger, KIND, write=False).by_kind().get(KIND, 0.0), "job": os.environ.get("SLURM_JOB_ID"), "when": time.time()}
with open(f"{a.wroot}/{a.panel}_score_runs.jsonl", "a") as f:
    f.write("\n" + json.dumps(rec) + "\n")
print("WAN_SCORE", json.dumps(rec))
raise SystemExit(0 if rc == 0 else 1)
