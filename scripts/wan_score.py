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
WHAT = {w: (["--post_swap_only"] if f[0] else []) + (["--as_receiver"] if f[1] else []) + (["--lag_ref", a.a_roll] if f[2] else [])
        for w, f in WP.SCORE_FLAGS.items()}  # one flag table with the panel's intake check (Astra t1 item 13)


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
    # Astra t1 item 12: compare the bindings RECORDED AT EXECUTION (receipts rc 0, scorer / adapter / wrapper / every metric
    # weight, the A@20k reference files and signed metrics actually used) with the current files, never stamp current values
    BIND = WP.score_binding(a.a_metrics, a.a_roll)
    import glob as _g
    AR = {os.path.basename(f): WP._sha_file(f) for f in sorted(_g.glob(f"{a.a_roll.rstrip('/')}/*_s[0-9]*.pt"))}
    ok_pp, why_pp = WP.parity_parts_ok([(uo, U, UR)] + list(zip(parts, S, SR)), BIND, AR, WP._sha_file(a.a_metrics))
    why += why_pp
    rec = {"pass": not why, "replay_pass": bool(UR.get("replay_pass")), "sharded_equal": sharded_equal, "parts_bound": ok_pp, "why": why,
           "device": next(iter(devs)) if len(devs) == 1 else sorted(devs), "binding": BIND, "inputs_sha": U["binding"]["inputs_sha"],
           "score_tick_s": round(UR["wall_s"] / max(1, UR["ticks"] or 1), 4), "ticks": UR["ticks"], "unsharded_wall_s": UR["wall_s"],
           "shard_wall_s": [r["wall_s"] for r in SR], "shard_rc": [r.get("rc") for r in SR], "shards": n, "runs": len(U["drives"]),
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
RATE = float(PR["score_tick_s"])
CAP = WP.score_cap_h(DTX, RATE)
KIND = f"score_{a.panel}"
JID = os.environ.get("SLURM_JOB_ID") or f"local{os.getpid()}"
si, sn = map(int, a.shard.split("/"))
mine = WP.shard_by_drive([f for f in M["runs"] if os.path.dirname(f) == os.path.normpath(src)], a.shard)
OWN = sum(M["runs"][f]["ticks"] for f in mine) * RATE / 3600  # post-swap scorings priced at full length (conservative)
RESV = f"{a.wroot}/{a.panel}_score_reservations.jsonl"
LOCK = f"{a.wroot}/{a.panel}_score.lock"


def _lock():
    """cross-node lock: mkdir is atomic on Lustre; a lock older than 300 s is stale (holder died)"""
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK); return
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(LOCK) > 300:
                    os.rmdir(LOCK); continue
            except OSError:
                pass
            if time.time() - t0 > 600:
                raise SystemExit(f"could not take {LOCK}")
            time.sleep(1.0)


def _resv():
    out = []
    if os.path.exists(RESV):
        for line in open(RESV):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                out.append(json.loads(line))
    return out


# Astra t1 item 14: aggregate headroom across ACTIVE workers, checked and reserved under one lock, before any scoring
_lock()
try:
    jobs = WP.Ledger.read(a.ledger)
    spent = sum(WP.Ledger.charge_h(j) for j in jobs.values() if j["kind"] == KIND)
    others = WP.active_reserved_h(_resv(), jobs, KIND, exclude=JID)
    ok_h, tot_h = WP.score_headroom(spent, others, OWN, CAP)
    print(f"score ledger {KIND}: spent {spent:.3f} h + other live workers {others:.3f} h + this shard {OWN:.3f} h = {tot_h:.3f} h of cap "
          f"{CAP:.3f} h (1.25 x {WP.panel_score_h(DTX, RATE):.3f} h at {RATE} s / tick on {PR['device']})", flush=True)
    if not ok_h:
        print(f"SCORE_CAP_EXCEEDED {KIND}: refused before scoring", flush=True)
        sys.exit(10)
    with open(RESV, "a") as f:
        f.write("\n" + json.dumps({"job": JID, "est_h": OWN, "what": a.what, "shard": a.shard, "t": time.time()}) + "\n")
finally:
    try:
        os.rmdir(LOCK)
    except OSError:
        pass
out = f"{a.wroot}/m_{a.panel}_{a.what}_dev.shard{si}of{sn}.json"
cmd = [PY, "scripts/score_rgb.py", "--rgb", src, "--rgb_manifest", MAN, "--shard", a.shard, "--out", out,
       "--tag", f"{a.what}{si}of{sn}"] + WHAT[a.what]
rc, log, wall = run_capped(cmd, KIND, CAP)
final = sum(WP.Ledger.charge_h(j) for j in WP.Ledger.read(a.ledger).values() if j["kind"] == KIND)
over = final > CAP
if over and os.path.exists(out):  # finished, but the panel's scoring went over its cap: the output is not admissible
    os.replace(out, out + ".overcap")
rec = {"panel": a.panel, "what": a.what, "shard": a.shard, "out": out, "rc": rc, "over_cap": over, "wall_s": wall, "device": dv, "cap_h": CAP,
       "score_ledger_h": round(final, 4), "own_est_h": round(OWN, 4), "job": JID, "when": time.time()}
with open(f"{a.wroot}/{a.panel}_score_runs.jsonl", "a") as f:
    f.write("\n" + json.dumps(rec) + "\n")
print("WAN_SCORE", json.dumps(rec))
if over:
    print(f"SCORE_CAP_EXCEEDED {KIND}: {final:.3f} h > cap {CAP:.3f} h after this shard; output set aside", flush=True)
    sys.exit(10)
raise SystemExit(0 if rc == 0 else 1)
