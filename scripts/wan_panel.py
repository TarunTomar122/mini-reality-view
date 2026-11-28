"""PLAN-video S1 dev panel (Astra items 9, 10, 12; S1b items 7-11): one checkpoint -> full six-drive x two-seed panel through the
SIGNED comparator, then the S1 decision.

1. RGB-path parity (S1b 7): the signed A@20k SD-latent rollouts are decoded with the REAL SD VAE into RGB (score_rgb.py
   --from_lat, the scorer's own arithmetic) and scored through the RGB entry; every per-run field must equal the signed metrics.
   The cached pass is reused only if its binding still holds: sha of the signed metrics file, of every reference rollout file,
   of scripts/score_rgb.py (adapter), scripts/temporal_metrics.py (scorer) and of the SD VAE files. Any change -> replay.
2. Production check (S1b 11): the checkpoint must be a production one (global batch 8, not timing / overfit / test).
3. Four conditioning modes (hybrid, guide, tokens, none; same seeds = FIXED noise and windows), each at FULL length (every run's
   tick count == the drive's; --max_ticks refused): wan_rollout.py -> score_rgb.py -> signed temporal_compare.py with
   candidate FIRST, A@20k SECOND, A's k=3 count list, --floor 0 --cold_ref B; each panel complete (6 drives, 12 runs).
4. Camera swap on PAIRED evidence (S1b 8-9): hybrid --swap; per run: windows / noise == the unswapped hybrid run of the same drive
   and seed (from both files) and no donor truncation; post-swap car F1 vs donor keys and vs receiver keys on the SAME frames
   (score_rgb --post_swap_only common-frame selection; equal nonzero placement counts); lag from the full run vs the donor's
   signed A@20k cold reference (decoded with the real SD VAE), exclusions reported, non-recoveries infinite, every drive needs
   lag-eligible evidence -> mrv.wan_plan.donor_following (blocks on any missing / mismatched evidence).
5. Decision (mrv.wan_plan.decide_s1): < 12k updates unresolved; 'check' informational; '24h' happens ONCE (marker
   <run dir>/S1_24H_DECISION.json): pass / one 12 h extension only for car F1 in [0.4156, 0.4456) with every other gate passed /
   stop; 'final' only after a 24 h EXTEND marker, bar 0.4456. Gates need the production checkpoint and all four mode panels.
Budget (S1b 10, Astra 08:30): the panel is charged by the allocation's outer ledger beat (slurm/wan_panel.sh); it refuses to
start if ledger + the MEASURED panel cost (timing.json forecast_final.panel_h, or --panel_estimate_h) > --budget_hours, and the
deadline is enforced while every step runs (ledger polled every --poll_s s; the running step is terminated at the cap).
Swap donors (Astra / Neon 08:30): scripts/wan_swap_preflight.py plans and checks all six assignments on CPU first (full mode:
donor at least as long as the receiver; shared mode only if none qualifies: receiver cut to the donor length and compared with an
unswapped receiver run of that length, rendered here). Any blocked or missing swap evidence makes the decision INVALID and no
24 h marker is written; STOP needs complete evidence. Both 20 % drop marginals are recorded with every comparison.
Writes <wroot>/<name>_panel.json. Dev only.
  python scripts/wan_panel.py --ckpt runs/wan_s1/ckpt_12000.pt --name s1_12k --stage check
"""
import argparse, glob, json, os, subprocess, sys, time
from mrv import wan_plan as WP

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=""); p.add_argument("--name", default="")
p.add_argument("--parity_only", action="store_true", help="only (re)establish the bound A@20k RGB-path parity record (MODE=train needs it)")
p.add_argument("--stage", default="check", choices=["check", "24h", "final"])
p.add_argument("--wroot", default="runs/wroll")
p.add_argument("--a_metrics", default="runs/troll/m_A20k_indep_dev.json")
p.add_argument("--a_roll", default="runs/troll/A20k_indep_dev")
p.add_argument("--count_list", default="runs/count/count_A20k_dev_k3.json")
p.add_argument("--lists", default="data/temporal_lists.json"); p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--ledger", default="runs/wan_s1_ledger.jsonl")
p.add_argument("--budget_hours", type=float, default=24.0, help="36 after the one signed extension")
p.add_argument("--panel_estimate_h", type=float, default=None, help="override of the measured full-panel cost (hours)")
p.add_argument("--timing_json", default="", help="timing.json with forecast_final.panel_h (default: newest runs/wan_timing_*/timing.json)")
p.add_argument("--poll_s", type=float, default=30.0, help="ledger deadline check interval while a step runs")
p.add_argument("--swap_at", type=int, default=50)
p.add_argument("--rollout_args", default="", help="extra flags for wan_rollout.py, e.g. '--steps 30 --cfg 1.0' (no --max_ticks)")
a = p.parse_args()
if not a.parity_only and not (a.ckpt and a.name):
    raise SystemExit("--ckpt and --name are required (or --parity_only)")
PY = sys.executable
PARITY_RECORD = f"{a.wroot}/a20k_rgbpath_parity.json"
MODES = ["hybrid", "guide", "tokens", "none"]
roll_extra = a.rollout_args.split() if a.rollout_args else []
if any(x.startswith("--max_ticks") or x in ("--swap", "--split", "--cond", "--seeds", "--drive_list") for x in roll_extra):
    raise SystemExit("--rollout_args may not change length, split, mode, seeds or drives (full-length 6 x 2 panels only)")
LEDGER = WP.Ledger(a.ledger, "panel").start()
os.makedirs(a.wroot, exist_ok=True)
T0 = time.time(); timing = {}


def fail(msg, note="failed"):
    LEDGER.stop(note); raise SystemExit(f"PANEL_FAIL {msg}")


def cap():
    h = LEDGER.hours()
    if h >= a.budget_hours:
        fail(f"budget: S1 ledger {h:.2f} h >= {a.budget_hours} h", "budget")


def run(cmd, what):
    """one panel step; the S1 deadline is enforced WHILE it runs (Astra 08:30): the ledger is polled every --poll_s seconds and
    the step is terminated (TERM, then KILL) when the cap is reached"""
    cap()
    t = time.time(); print("RUN", " ".join(cmd), flush=True)
    import tempfile
    with tempfile.TemporaryFile("w+") as fo, tempfile.TemporaryFile("w+") as fe:
        pr = subprocess.Popen(cmd, stdout=fo, stderr=fe, text=True)
        while pr.poll() is None:
            try:
                pr.wait(timeout=a.poll_s)
            except subprocess.TimeoutExpired:
                if LEDGER.hours() >= a.budget_hours:
                    pr.terminate()
                    try:
                        pr.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        pr.kill()
                    fail(f"budget reached during {what}: S1 ledger {LEDGER.hours():.2f} h >= {a.budget_hours} h", "budget")
        fo.seek(0); fe.seek(0); out, err = fo.read(), fe.read()
    timing[what] = round(time.time() - t, 1)
    if pr.returncode:
        print(out[-3000:], err[-3000:], flush=True)
        fail(f"{what} (exit {pr.returncode})")
    return out


def tcompare(m):
    out = run([PY, "scripts/temporal_compare.py", "--compare", m, a.a_metrics, "--count_list", a.count_list, "--floor", "0", "--cold_ref", "B"], f"compare {m}")
    d = json.loads(out.split("TCOMPARE ", 1)[1].splitlines()[0])
    if not (d["drives"] == 6 and d["runs"] == 12 and not d["unpaired_drives"]):
        fail(f"incomplete panel for {m}: {d['drives']} drives, {d['runs']} runs, unpaired {d['unpaired_drives']}", "incomplete")
    if d.get("cold_ref") != "B" or d.get("A") != m or d.get("B") != a.a_metrics:
        fail(f"compare for {m} not run as specified (A {d.get('A')}, B {d.get('B')}, cold_ref {d.get('cold_ref')})", "cold_ref")
    return d


import torch


def ensure_parity():
    """(re)establish the A@20k replay parity through the real-SD-to-RGB path, bound to references / adapter / scorer / decoder"""
    BIND = WP.parity_binding(a.a_metrics, a.a_roll)
    if not WP.parity_ok(PARITY_RECORD, BIND):
        out = run([PY, "scripts/score_rgb.py", "--from_lat", a.a_roll, "--parity_ref", a.a_metrics, "--out", f"{a.wroot}/m_A20k_rgbpath.json"], "a20k_parity")
        ok = "RGB_PATH_PARITY PASS" in out
        json.dump({"pass": ok, "binding": BIND, "log": out[-500:], "when": time.time()}, open(PARITY_RECORD, "w"), indent=1)
        if not ok:
            fail("A@20k replay through the real-SD-to-RGB path differs from the signed metrics", "parity")
    return BIND


if a.parity_only:
    B_ = ensure_parity(); print("A20K_PARITY PASS", json.dumps(B_)); LEDGER.stop(); raise SystemExit(0)

# 0. swap preflight FIRST (Astra S1c item 3): all six real assignments must pass, with the rollout's actual window size, before
#    any GPU work or budget is spent
WIN = 5
for i_, x_ in enumerate(roll_extra):
    if x_ == "--window_latents" and i_ + 1 < len(roll_extra):
        WIN = int(roll_extra[i_ + 1])
    elif x_.startswith("--window_latents="):
        WIN = int(x_.split("=", 1)[1])
DJ = f"{a.wroot}/{a.name}_swap_donors.json"
run([PY, "scripts/wan_swap_preflight.py", "--lists", a.lists, "--enc", a.enc, "--swap_at", str(a.swap_at), "--window_latents", str(WIN), "--out", DJ],
    "swap preflight")
PLAN = json.load(open(DJ)); DON = PLAN["donors"]
if len(DON) != 6 or not PLAN.get("ok") or any(v is None for v in DON.values()):
    fail(f"swap preflight: need six qualifying assignments, got {DON}")

# 1. budget: the measured panel cost + one extra unswapped same-length rollout (2 seeds) per shared-mode donor (Astra S1c item 4)
used = LEDGER.hours()
FC = None
if a.panel_estimate_h is None:  # the TIMED reserve (wan_forecast.py), never a zero default
    tj = a.timing_json or (sorted(glob.glob("runs/wan_timing_*/timing.json"), key=os.path.getmtime) or [""])[-1]
    try:
        FC = json.load(open(tj))["forecast_final"]
        # base panel cost (without shared references) + the shared references of THIS run's actual donor plan
        EST = WP.panel_with_shared_h(float(FC.get("panel_base_h", FC["panel_h"])), DON, float(FC["roll_tick_s"]), float(FC["roll_start_s"]))
    except Exception as e:  # noqa: BLE001
        fail(f"no measured panel cost ({tj!r}: {type(e).__name__}); run MODE=timing or pass --panel_estimate_h")
else:
    EST = a.panel_estimate_h
if used + EST > a.budget_hours:
    fail(f"budget: ledger {used:.2f} h + measured panel cost {EST:.2f} h > {a.budget_hours} h", "budget")
CK = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
step = int(CK["step"])
PROD = WP.production_ok({"timing": CK.get("timing"), "recipe": CK.get("recipe"), "args": CK.get("args")})
MARK = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "S1_24H_DECISION.json")
prior = json.load(open(MARK))["verdict"] if os.path.exists(MARK) else None
if a.stage == "24h" and prior is not None:
    fail(f"a 24 h decision already exists ({MARK}: {prior}); the 24 h read happens once")
if a.stage == "final" and not (prior or "").startswith("EXTEND"):
    fail(f"final read without a qualifying 24 h EXTEND verdict (marker: {prior})")

# 2. RGB-path parity, bound to references, adapter, scorer and decoder
BIND = ensure_parity()
DROPM = {k: (CK.get("recipe") or {}).get(k) for k in ("p_drop_both", "p_drop_guide_only", "p_drop_tokens_only")}
if all(v is not None for v in DROPM.values()):
    DROPM.update(guide_dropped=DROPM["p_drop_both"] + DROPM["p_drop_guide_only"], tokens_dropped=DROPM["p_drop_both"] + DROPM["p_drop_tokens_only"])
res = {"ckpt": a.ckpt, "name": a.name, "stage": a.stage, "step": step, "production": PROD, "parity_binding": BIND, "modes": {},
       "panel_estimate_h": EST, "train_dropout_marginals": DROPM, "window_latents": WIN, "swap_preflight": DJ}

# 2-3. the four modes at fixed noise, full length
dev = sorted(json.load(open(a.lists))["dev"])
n_full = {d: len(torch.load(f"{a.enc}/{d}.pt", map_location="cpu", weights_only=False, mmap=True)["ticks"]) for d in dev}
mode_why, CMP = [], None
for cond in MODES:
    rd, m = f"{a.wroot}/{a.name}_{cond}_dev", f"{a.wroot}/m_{a.name}_{cond}_dev.json"
    run([PY, "scripts/wan_rollout.py", "--ckpt", a.ckpt, "--split", "dev", "--seeds", "0,1", "--cond", cond, "--out", rd, "--ledger", ""] + roll_extra, f"rollout {cond}")
    for d in dev:
        for sd in (0, 1):
            R = torch.load(f"{rd}/{d}_s{sd}.pt", map_location="cpu", weights_only=False, mmap=True)
            if len(R["ticks"]) != n_full[d] or R.get("max_ticks") or R.get("cond") != cond or R.get("ckpt_sha") is None:
                mode_why.append(f"{cond}: {d}_s{sd} not a full-length {cond} run ({len(R['ticks'])}/{n_full[d]} ticks)")
    run([PY, "scripts/score_rgb.py", "--rgb", rd, "--out", m], f"score {cond}")
    d_ = tcompare(m)
    res["modes"][cond] = {"metrics": m, "car_f1": d_["car_f1"], "road_iou": d_["road_iou"], "ID": d_["ID"], "W": d_["W"],
                          "ghost_ge5": d_.get("ghost_ge5"), "recall": d_.get("recall"), "flips": d_.get("flips"),
                          "train_dropout_marginals": DROPM}  # recorded with every comparison (Astra 08:30)
    if cond == "hybrid":
        CMP = d_
MODES_OK = {"ok": not mode_why and set(res["modes"]) == set(MODES), "why": mode_why}

# 4. camera swap on paired evidence (donor plan from step 0)
rd, hyb = f"{a.wroot}/{a.name}_swap_dev", f"{a.wroot}/{a.name}_hybrid_dev"
run([PY, "scripts/wan_rollout.py", "--ckpt", a.ckpt, "--split", "dev", "--seeds", "0,1", "--cond", "hybrid", "--swap", "--swap_at", str(a.swap_at),
     "--donors", DJ, "--out", rd, "--ledger", ""] + roll_extra, "rollout swap")
REF = {}  # receiver -> folder of its unswapped reference run (full-length hybrid, or a same-length run for shared mode)
for r_, dp in DON.items():
    REF[r_] = hyb
    if dp["mode"] == "shared":
        sd_ = f"{a.wroot}/{a.name}_hybrid_shared_{r_}"; os.makedirs(sd_, exist_ok=True)
        open(f"{sd_}/drive.txt", "w").write(r_ + "\n")
        run([PY, "scripts/wan_rollout.py", "--ckpt", a.ckpt, "--split", "dev", "--seeds", "0,1", "--cond", "hybrid", "--drive_list", f"{sd_}/drive.txt",
             "--max_ticks", str(dp["n"]), "--out", sd_, "--ledger", ""] + roll_extra, f"rollout shared reference {r_}")
        REF[r_] = sd_
mf, md, mr = (f"{a.wroot}/m_{a.name}_swap_{k}.json" for k in ("full", "postdonor", "postrecv"))
run([PY, "scripts/score_rgb.py", "--rgb", rd, "--lag_ref", a.a_roll, "--out", mf], "score swap full")
run([PY, "scripts/score_rgb.py", "--rgb", rd, "--post_swap_only", "--out", md], "score swap donor")
run([PY, "scripts/score_rgb.py", "--rgb", rd, "--post_swap_only", "--as_receiver", "--out", mr], "score swap receiver")
full, dn, rc = (json.load(open(f))["drives"] for f in (mf, md, mr))
runs = {}
for d in dev:
    for sd in (0, 1):
        rid = f"{d}_s{sd}"
        if rid not in full or rid not in rc:
            continue
        S = torch.load(f"{rd}/{rid}.pt", map_location="cpu", weights_only=False, mmap=True)
        H = torch.load(f"{REF[d]}/{rid}.pt", map_location="cpu", weights_only=False, mmap=True)
        did = f"{S['other']}_s{sd}x{d}"  # donor-keyed post-swap file of THIS receiver
        if did not in dn or S["other"] != DON[d]["donor"]:
            continue
        wm = (S.get("windows") == H.get("windows") and S.get("windows_match_unswapped") is True and not S.get("truncated_by_donor")
              and S.get("ckpt_sha") == H.get("ckpt_sha"))
        q = full[rid]
        runs[rid] = {"donor_f1": dn[did]["car_f1"], "receiver_f1": rc[rid]["car_f1"], "donor_frames": len(dn[did]["placement_ticks"]),
                     "receiver_frames": len(rc[rid]["placement_ticks"]), "windows_match": wm, "donor": S["other"], "mode": DON[d]["mode"],
                     "ref_ticks": len(H["ticks"]),
                     "lag": {"excluded": bool(q.get("lag_excluded")), "recovered": bool(q.get("lag_recovered")), "start": q.get("lag_start"),
                             "eligible_windows": q.get("lag_eligible_windows")}}
SWAP = WP.donor_following(runs)
res["swap"] = {**SWAP, "runs": runs, "donor_plan": DON, "train_dropout_marginals": DROPM}

# 5. decision
res["decision"] = WP.decide_s1(CMP, step, a.stage, SWAP, PROD, MODES_OK, prior)
res["timing_s"] = timing; res["panel_hours"] = round((time.time() - T0) / 3600, 3)
res["ledger_hours"] = round(LEDGER.hours(), 3); res["ledger_by_kind"] = LEDGER.by_kind()
json.dump(res, open(f"{a.wroot}/{a.name}_panel.json", "w"), indent=1, default=str)
if LEDGER.hours() > a.budget_hours:  # Astra S1c item 4: the last step may have finished between deadline polls
    res["decision"]["verdict"] = f"INVALID (S1 budget exceeded at publication: {LEDGER.hours():.2f} h > {a.budget_hours} h; no marker)"
    json.dump(res, open(f"{a.wroot}/{a.name}_panel.json", "w"), indent=1, default=str)
if a.stage == "24h" and step >= WP.FLOOR_UPDATES and not res["decision"]["verdict"].startswith(("INVALID", "unresolved")):
    json.dump({"verdict": res["decision"]["verdict"], "panel": f"{a.wroot}/{a.name}_panel.json", "step": step, "ckpt": a.ckpt},
              open(MARK, "w"), indent=1)
print("WAN_PANEL", json.dumps({"step": step, "stage": a.stage, "verdict": res["decision"]["verdict"], "gates": res["decision"]["gates"],
                               "swap_blocked": SWAP["blocked"], "modes": {k: v["car_f1"].get("A") for k, v in res["modes"].items()}}, default=str), flush=True)
LEDGER.stop()
