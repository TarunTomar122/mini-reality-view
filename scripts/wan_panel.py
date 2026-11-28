"""PLAN-video S1 dev panel (Astra items 9, 10, 12; S1b items 7-11; council/s1-timing-r1-CONVERGED.md item 4 = F3): one checkpoint
-> full six-drive x two-seed panel through the SIGNED comparator, then the S1 decision. F3 splits it into three steps; the
panel itself is unchanged (6 drives x 2 seeds, every eligible tick, four modes, swap, shared-mode references, six valid donors;
every signed bar unchanged):

--phase render (H200 slice, charged to the S1 ledger; slurm/wan_panel.sh <ckpt> <name> <stage>):
   0. swap preflight (CPU, scripts/wan_swap_preflight.py): six valid assignments at the rollout's window size, else no GPU work;
   1. budget: refuses if ledger + the MEASURED render-only panel cost (timing.json forecast_final: panel_h_total priced at the
      load-once rollout rate, panel_cost_basis "render_only_load_once") > --budget_hours; the deadline is enforced while it runs;
   2. production check (S1b 11) and the 24 h / final marker rules;
   3. ONE scripts/wan_rollout.py --plan process renders hybrid / guide / tokens / none at fixed noise, the swap run and every
      shared-mode reference with Wan loaded ONCE (per-run state / RNG reset); each mode run must be full length;
   4. writes <wroot>/<name>_rgb_manifest.json (sha256 + RGB sha + ticks of every rendered file) + .sha256 and <name>_render.json;
      the slice is released. Scoring is NOT done on the slice.
scoring (2080 Ti jobs, slurm/wan_score.sh panel <name> <what> <i/n>; own ledger runs/wan_score_ledger.jsonl with cap 1.25 x the
   measured scoring GPU-h of one panel, reported separately): scripts/wan_score.py -> m_<name>_<what>_dev.shard<i>of<n>.json.
--phase decide (CPU, no slice time): requires the bound 2080 Ti scorer-parity PASS (A@20k replay + sharded == unsharded;
   mrv.wan_plan.score_parity_ok); reads ONLY score shards whose binding names the RGB manifest this panel rendered (sha) and the
   parity record's device / scorer / adapter / SD VAE; every run present; merges the shards per scoring, then: signed
   temporal_compare.py per mode (candidate FIRST, A@20k SECOND, A's k=3 count list, --floor 0 --cold_ref B; complete 6 x 12);
   camera swap on PAIRED evidence (windows / noise == the unswapped reference, equal nonzero common frames, lag vs the donor's
   signed A@20k cold reference, every drive lag-eligible -> mrv.wan_plan.donor_following); decision mrv.wan_plan.decide_s1
   (< 12k unresolved; 'check' informational; '24h' ONCE, marker <run dir>/S1_24H_DECISION.json; 'final' only after an EXTEND
   marker, bar 0.4456). Any blocked or missing evidence -> INVALID, never STOP. Writes <wroot>/<name>_panel.json.
--rehearsal (slice, part of MODE=timing): one receiver run rendered by the load-once loop (after a different run in the same
   process) must be byte-equal (RGB and SD latents) to a fresh-load render; also measures the load-once rollout rate and the one
   model load -> <out>/rehearsal.json (scripts/wan_forecast.py prices the panel from it; re-time rule).
  python scripts/wan_panel.py --phase render --ckpt runs/wan_s1/ckpt_12000.pt --name s1_12k --stage 24h
  python scripts/wan_panel.py --phase decide --ckpt runs/wan_s1/ckpt_12000.pt --name s1_12k --stage 24h
"""
import argparse, glob, json, os, subprocess, sys, time
from mrv import wan_plan as WP

p = argparse.ArgumentParser()
p.add_argument("--phase", default="", choices=["", "render", "decide"])
p.add_argument("--rehearsal", action="store_true", help="MODE=timing: load-once vs fresh-load byte-equality + load-once rates")
p.add_argument("--out", default="", help="--rehearsal output folder")
p.add_argument("--ckpt", default=""); p.add_argument("--name", default="")
p.add_argument("--stage", default="check", choices=["check", "24h", "final"])
p.add_argument("--wroot", default="runs/wroll")
p.add_argument("--a_metrics", default="runs/troll/m_A20k_indep_dev.json")
p.add_argument("--a_roll", default="runs/troll/A20k_indep_dev")
p.add_argument("--count_list", default="runs/count/count_A20k_dev_k3.json")
p.add_argument("--lists", default="data/temporal_lists.json"); p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--ledger", default="runs/wan_s1_ledger.jsonl")
p.add_argument("--score_ledger", default="runs/wan_score_ledger.jsonl")
p.add_argument("--budget_hours", type=float, default=24.0, help="36 after the one signed extension")
p.add_argument("--panel_estimate_h", type=float, default=None, help="override of the measured render-only panel cost (hours)")
p.add_argument("--timing_json", default="", help="timing.json with forecast_final (default: newest runs/wan_timing_*/timing.json)")
p.add_argument("--poll_s", type=float, default=30.0, help="ledger deadline check interval while a step runs")
p.add_argument("--swap_at", type=int, default=50)
p.add_argument("--rollout_args", default="", help="extra flags for wan_rollout.py, e.g. '--steps 30 --cfg 1.0' (no --max_ticks)")
a = p.parse_args()
if not a.rehearsal and not (a.phase and a.ckpt and a.name):
    raise SystemExit("give --phase render|decide with --ckpt and --name (scoring: slurm/wan_score.sh), or --rehearsal --ckpt --out")
if a.rehearsal and not (a.ckpt and a.out):
    raise SystemExit("--rehearsal needs --ckpt and --out")
PY = sys.executable
PREC = f"{a.wroot}/score_parity_2080.json"
MODES = ["hybrid", "guide", "tokens", "none"]
SCORINGS = MODES + ["swap_full", "swap_postdonor", "swap_postrecv"]
# Astra S1d item 3: the rollout options are parsed ONCE (the rollout's own option set, abbreviations resolved the same way); the
# canonical list carries --window_latents explicitly and goes to every wan_rollout.py run, WIN to the preflight
WIN, roll_extra = WP.parse_rollout_extra(a.rollout_args.split() if a.rollout_args else [])
# the slice ledger is charged by the allocation's outer beat (render / rehearsal); decide runs on CPU and only READS it
LEDGER = WP.Ledger(a.ledger, "panel", write=False if a.phase == "decide" else None).start()
os.makedirs(a.wroot, exist_ok=True)
T0 = time.time(); timing = {}


def fail(msg, note="failed"):
    LEDGER.stop(note); raise SystemExit(f"PANEL_FAIL {msg}")


def cap():
    if a.phase == "decide":
        return
    h = LEDGER.hours()
    if h >= a.budget_hours:
        fail(f"budget: S1 ledger {h:.2f} h >= {a.budget_hours} h", "budget")


def run(cmd, what):
    """one step; on the slice the S1 deadline is enforced WHILE it runs (Astra 08:30): the ledger is polled every --poll_s seconds
    and the step is terminated (TERM, then KILL) when the cap is reached"""
    cap()
    t = time.time(); print("RUN", " ".join(cmd), flush=True)
    import tempfile
    with tempfile.TemporaryFile("w+") as fo, tempfile.TemporaryFile("w+") as fe:
        pr = subprocess.Popen(cmd, stdout=fo, stderr=fe, text=True)
        while pr.poll() is None:
            try:
                pr.wait(timeout=a.poll_s)
            except subprocess.TimeoutExpired:
                if a.phase != "decide" and LEDGER.hours() >= a.budget_hours:
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


import torch


def roll_cmd(rd, cond, seeds="0,1", extra=()):
    return ["--ckpt", a.ckpt, "--split", "dev", "--seeds", seeds, "--cond", cond, "--out", rd, "--ledger", ""] + list(extra) + roll_extra


def plan_run(runs, out, what):
    """ONE wan_rollout.py process renders every run (Wan loaded once); returns its plan_timing.json"""
    os.makedirs(out, exist_ok=True)
    pj = f"{out}/plan.json"
    json.dump({"runs": runs}, open(pj, "w"), indent=1)
    run([PY, "scripts/wan_rollout.py", "--ckpt", a.ckpt, "--plan", pj, "--out", out, "--ledger", ""] + roll_extra, what)
    return json.load(open(f"{out}/plan_timing.json"))


def load_run(f):
    return torch.load(f, map_location="cpu", weights_only=False, mmap=True)


# ================================================================================================ rehearsal (MODE=timing)
if a.rehearsal:
    dev = sorted(json.load(open(a.lists))["dev"])
    R = dev[0]
    os.makedirs(a.out, exist_ok=True)
    open(f"{a.out}/drive.txt", "w").write(R + "\n")
    dl = ["--drive_list", f"{a.out}/drive.txt", "--allow_inert"]
    # load-once loop: a DIFFERENT run first (tokens-only mode), then the receiver's hybrid run, in one process
    PT = plan_run([roll_cmd(f"{a.out}/loop_tokens", "tokens", "0", dl), roll_cmd(f"{a.out}/loop_hybrid", "hybrid", "0", dl)],
                  f"{a.out}/loop", "rehearsal load-once loop")
    t = time.time()
    run([PY, "scripts/wan_rollout.py"] + roll_cmd(f"{a.out}/fresh_hybrid", "hybrid", "0", dl), "rehearsal fresh load")
    fresh_wall = time.time() - t
    A_, B_ = load_run(f"{a.out}/loop_hybrid/{R}_s0.pt"), load_run(f"{a.out}/fresh_hybrid/{R}_s0.pt")
    why = []
    if A_["rgb_sha"] != B_["rgb_sha"] or not torch.equal(A_["rgb"], B_["rgb"]):
        why.append("RGB differs between the load-once loop and a fresh load")
    if ("lat" in A_) != ("lat" in B_) or ("lat" in A_ and not torch.equal(A_["lat"], B_["lat"])):
        why.append("SD latents differ")
    for k in ("ticks", "ok", "hard", "restart"):
        if not torch.equal(A_[k], B_[k]):
            why.append(f"{k} differs")
    if A_.get("windows") != B_.get("windows"):
        why.append("windows differ")
    runs_ = [load_run(f"{a.out}/loop_{c}/{R}_s0.pt") for c in ("tokens", "hybrid")]
    ticks = sum(len(r_["ticks"]) for r_ in runs_)
    inner = sum(sum(r_["timing"][k] for k in ("encode_s", "sample_s", "decode_s", "sd_s")) for r_ in runs_)
    wall_runs = sum(r_["wall_s"] for r_ in PT["runs"])
    rec = {"pass": not why, "why": "; ".join(why) or "byte-equal", "receiver": R, "rgb_sha_loop": A_["rgb_sha"], "rgb_sha_fresh": B_["rgb_sha"],
           "ticks": ticks, "load_s": PT["load_s"], "roll_tick_inner_s": round(inner / ticks, 4), "roll_tick_wall_s": round(wall_runs / ticks, 4),
           "roll_tick_s": round(max(inner, wall_runs) / ticks, 4), "window_latents": int(A_["window_latents"]), "fresh_process_s": round(fresh_wall, 1),
           "ckpt_sha": A_["ckpt_sha"], "plan_timing": PT, "timing_s": timing}
    json.dump(rec, open(f"{a.out}/rehearsal.json", "w"), indent=1)
    print("WAN_PANEL_REHEARSAL", json.dumps({k: v for k, v in rec.items() if k != "plan_timing"}), flush=True)
    LEDGER.stop(); raise SystemExit(0 if rec["pass"] else 1)

DJ = f"{a.wroot}/{a.name}_swap_donors.json"
MANP = f"{a.wroot}/{a.name}_rgb_manifest.json"
RENDER = f"{a.wroot}/{a.name}_render.json"
MARK = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "S1_24H_DECISION.json")
prior = json.load(open(MARK))["verdict"] if os.path.exists(MARK) else None
if a.stage == "24h" and prior is not None:
    fail(f"a 24 h decision already exists ({MARK}: {prior}); the 24 h read happens once")
if a.stage == "final" and not (prior or "").startswith("EXTEND"):
    fail(f"final read without a qualifying 24 h EXTEND verdict (marker: {prior})")
dev = sorted(json.load(open(a.lists))["dev"])

# ================================================================================================ render (slice)
if a.phase == "render":
    # 0. swap preflight FIRST (Astra S1c item 3), before any GPU work or budget is spent
    run([PY, "scripts/wan_swap_preflight.py", "--lists", a.lists, "--enc", a.enc, "--swap_at", str(a.swap_at), "--window_latents", str(WIN), "--out", DJ],
        "swap preflight")
    PLAN = json.load(open(DJ)); DON = PLAN["donors"]
    if len(DON) != 6 or not PLAN.get("ok") or any(v is None for v in DON.values()):
        fail(f"swap preflight: need six qualifying assignments, got {DON}")
    # 1. budget: the measured RENDER-ONLY panel cost (load-once rate) + shared references of this run's actual donor plan
    used = LEDGER.hours()
    if a.panel_estimate_h is None:
        tj = a.timing_json or (sorted(glob.glob("runs/wan_timing_*/timing.json"), key=os.path.getmtime) or [""])[-1]
        try:
            FC = json.load(open(tj))["forecast_final"]
            if FC.get("panel_cost_basis") != "render_only_load_once":
                raise ValueError("legacy forecast (scoring priced on the slice): re-run MODE=timing")
            if (FC.get("panel_settings") or {}) != {"window_latents": WIN, "swap_at": a.swap_at}:
                raise ValueError(f"forecast priced {FC.get('panel_settings')}, panel uses window {WIN} / swap {a.swap_at}")
            if FC.get("rollout_window_latents") != WIN:  # Astra S1g 4b
                raise ValueError(f"forecast rates measured at rollout window {FC.get('rollout_window_latents')}, panel uses window {WIN}")
            EST = WP.panel_with_shared_h(float(FC["panel_base_h"]), DON, float(FC["roll_tick_s"]), float(FC.get("shared_ref_start_s", 0.0)))
        except Exception as e:  # noqa: BLE001
            fail(f"no measured panel cost for these settings ({tj!r}: {type(e).__name__}: {e}); rerun scripts/wan_forecast.py with "
                 f"--window_latents {WIN} --swap_at {a.swap_at}, or pass --panel_estimate_h")
    else:
        EST = a.panel_estimate_h
    if used + EST > a.budget_hours:
        fail(f"budget: ledger {used:.2f} h + measured render-only panel cost {EST:.2f} h > {a.budget_hours} h", "budget")
    CK = load_run(a.ckpt)
    step = int(CK["step"])
    PROD = WP.production_ok({"timing": CK.get("timing"), "recipe": CK.get("recipe"), "args": CK.get("args")})
    # 3. every run of the panel in ONE process (Wan loaded once)
    DIRS = {c: f"{a.wroot}/{a.name}_{c}_dev" for c in MODES}
    DIRS["swap"] = f"{a.wroot}/{a.name}_swap_dev"
    runs = [roll_cmd(DIRS[c], c) for c in MODES]
    runs.append(roll_cmd(DIRS["swap"], "hybrid", extra=["--swap", "--swap_at", str(a.swap_at), "--donors", DJ]))
    SHARED = {}
    for r_, dp in DON.items():
        if dp["mode"] == "shared":
            sd_ = f"{a.wroot}/{a.name}_hybrid_shared_{r_}"; os.makedirs(sd_, exist_ok=True)
            open(f"{sd_}/drive.txt", "w").write(r_ + "\n")
            runs.append(roll_cmd(sd_, "hybrid", extra=["--drive_list", f"{sd_}/drive.txt", "--max_ticks", str(dp["n"])]))
            SHARED[r_] = sd_
    PT = plan_run(runs, f"{a.wroot}/{a.name}_plan", "render panel (load once)")
    n_full = {d: len(torch.load(f"{a.enc}/{d}.pt", map_location="cpu", weights_only=False, mmap=True)["ticks"]) for d in dev}
    mode_why = []
    for cond in MODES:
        for d in dev:
            for sd in (0, 1):
                R = load_run(f"{DIRS[cond]}/{d}_s{sd}.pt")
                if len(R["ticks"]) != n_full[d] or R.get("max_ticks") or R.get("cond") != cond or R.get("ckpt_sha") is None:
                    mode_why.append(f"{cond}: {d}_s{sd} not a full-length {cond} run ({len(R['ticks'])}/{n_full[d]} ticks)")
    # 4. the RGB + sha manifest written on the slice: scoring reads only these bytes
    entries = {}
    for dd in list(DIRS.values()) + list(SHARED.values()):
        for f in sorted(glob.glob(f"{dd}/*_s[0-9]*.pt")):
            R = load_run(f)
            entries[os.path.normpath(f)] = {"sha256": WP._sha_file(f), "rgb_sha": R.get("rgb_sha"), "ticks": len(R["ticks"]),
                                            "size": os.path.getsize(f), "ckpt_sha": R.get("ckpt_sha")}
    MAN = {"name": a.name, "stage": a.stage, "ckpt": a.ckpt, "step": step, "dirs": {**DIRS, "shared": SHARED}, "runs": entries,
           "dev_ticks_x_seeds": 2 * sum(n_full.values()), "window_latents": WIN, "swap_at": a.swap_at, "donors": DJ,
           "plan_timing": PT, "when": time.time()}
    json.dump(MAN, open(MANP, "w"), indent=1, sort_keys=True)
    msha = WP._sha_file(MANP)
    open(MANP + ".sha256", "w").write(f"{msha}  {os.path.basename(MANP)}\n")
    over = LEDGER.hours() > a.budget_hours  # the last step may have finished between deadline polls
    rec = {"ckpt": a.ckpt, "name": a.name, "stage": a.stage, "step": step, "production": PROD, "mode_why": mode_why, "panel_estimate_h": EST,
           "rgb_manifest": MANP, "rgb_manifest_sha": msha, "window_latents": WIN, "swap_preflight": DJ, "timing_s": timing,
           "render_slice_h": round((time.time() - T0) / 3600, 3), "ledger_hours": round(LEDGER.hours(), 3), "budget_exceeded": over,
           "plan_load_s": PT["load_s"]}
    json.dump(rec, open(RENDER, "w"), indent=1, default=str)
    print("WAN_PANEL_RENDERED", json.dumps({k: rec[k] for k in ("name", "step", "rgb_manifest_sha", "render_slice_h", "budget_exceeded")}), flush=True)
    print(f"next: score on 2080 Ti jobs (slurm/wan_score.sh panel {a.name} <what> <i/n> for {SCORINGS}), then "
          f"scripts/wan_panel.py --phase decide --ckpt {a.ckpt} --name {a.name} --stage {a.stage}", flush=True)
    LEDGER.stop(); raise SystemExit(0)

# ================================================================================================ decide (CPU)
if not os.path.exists(RENDER) or not os.path.exists(MANP):
    fail(f"no render record for {a.name}: run --phase render on the slice first")
RD = json.load(open(RENDER))
MSHA = WP._sha_file(MANP)
if RD["rgb_manifest_sha"] != MSHA or open(MANP + ".sha256").read().split()[0] != MSHA:
    fail(f"{MANP} changed after rendering")
if RD.get("budget_exceeded"):
    fail("the render finished over the S1 budget: no verdict")
M = json.load(open(MANP))
if os.path.abspath(M["ckpt"]) != os.path.abspath(a.ckpt) or RD["stage"] != a.stage:
    fail(f"render record is for {M['ckpt']} / {RD['stage']}, decide asked for {a.ckpt} / {a.stage}")
ok_p, why_p = WP.score_parity_ok(PREC, WP.score_binding(a.a_metrics, a.a_roll))
if not ok_p:
    fail(f"no bound 2080 Ti scorer-parity PASS ({PREC}: {why_p})", "parity")
PR = json.load(open(PREC))
step, PROD, mode_why = RD["step"], RD["production"], RD["mode_why"]
DIRS = M["dirs"]
DON = json.load(open(M["donors"]))["donors"]


def runs_in(d):
    return sorted(os.path.basename(f)[:-3] for f in M["runs"] if os.path.dirname(f) == os.path.normpath(d))


def merged_scores(w):
    """merge the 2080 Ti shards of one scoring; only shards bound to THIS render's RGB manifest and the parity record's device /
    scorer / adapter / SD VAE count; every rendered run must be scored exactly once"""
    fs = sorted(glob.glob(f"{a.wroot}/m_{a.name}_{w}_dev.shard*of*.json"))
    ns = {int(f.rsplit("of", 1)[1][:-5]) for f in fs}
    if len(ns) != 1:
        fail(f"scoring {w}: shard files {fs} (need one complete i/n set)", "incomplete")
    n = ns.pop()
    want = [f"{a.wroot}/m_{a.name}_{w}_dev.shard{i}of{n}.json" for i in range(n)]
    if sorted(want) != fs:
        fail(f"scoring {w}: shards {fs} != {want}", "incomplete")
    out, why_m = WP.merge_score_shards([(f, json.load(open(f))) for f in want], MSHA, PR)
    if why_m:
        fail(f"scoring {w}: {why_m[:3]}", "incomplete")
    out["roll"] = f"merged:{w}:{MSHA[:16]}"
    src = runs_in(DIRS["swap" if w.startswith("swap") else w])
    exp = len(src)
    if (w == "swap_postdonor" and len(out["drives"]) != exp) or (w != "swap_postdonor" and sorted(out["drives"]) != src):
        fail(f"scoring {w}: {len(out['drives'])} runs scored, {exp} rendered", "incomplete")
    out["binding"] = {"merged_from": want, "rgb_manifest_sha": MSHA, "device": PR["device"]}
    mf = f"{a.wroot}/m_{a.name}_{w}_dev.json"
    json.dump(out, open(mf, "w"))
    return mf


def tcompare(m):
    out = run([PY, "scripts/temporal_compare.py", "--compare", m, a.a_metrics, "--count_list", a.count_list, "--floor", "0", "--cold_ref", "B"], f"compare {m}")
    d = json.loads(out.split("TCOMPARE ", 1)[1].splitlines()[0])
    if not (d["drives"] == 6 and d["runs"] == 12 and not d["unpaired_drives"]):
        fail(f"incomplete panel for {m}: {d['drives']} drives, {d['runs']} runs, unpaired {d['unpaired_drives']}", "incomplete")
    if d.get("cold_ref") != "B" or d.get("A") != m or d.get("B") != a.a_metrics:
        fail(f"compare for {m} not run as specified (A {d.get('A')}, B {d.get('B')}, cold_ref {d.get('cold_ref')})", "cold_ref")
    return d


def checked(f):
    """open a rendered file only if its bytes are the manifest's"""
    e = M["runs"].get(os.path.normpath(f))
    if e is None or WP._sha_file(f) != e["sha256"]:
        fail(f"{f} is not the file the slice rendered (sha manifest)", "incomplete")
    return load_run(f)


CK = load_run(a.ckpt)
DROPM = {k: (CK.get("recipe") or {}).get(k) for k in ("p_drop_both", "p_drop_guide_only", "p_drop_tokens_only")}
if all(v is not None for v in DROPM.values()):
    DROPM.update(guide_dropped=DROPM["p_drop_both"] + DROPM["p_drop_guide_only"], tokens_dropped=DROPM["p_drop_both"] + DROPM["p_drop_tokens_only"])
res = {"ckpt": a.ckpt, "name": a.name, "stage": a.stage, "step": step, "production": PROD, "parity_record": PREC, "parity_device": PR["device"],
       "modes": {}, "panel_estimate_h": RD["panel_estimate_h"], "train_dropout_marginals": DROPM, "window_latents": M["window_latents"],
       "swap_preflight": M["donors"], "rgb_manifest": MANP, "rgb_manifest_sha": MSHA, "render": RENDER}
CMP = None
for cond in MODES:
    m = merged_scores(cond)
    d_ = tcompare(m)
    res["modes"][cond] = {"metrics": m, "car_f1": d_["car_f1"], "road_iou": d_["road_iou"], "ID": d_["ID"], "W": d_["W"],
                          "ghost_ge5": d_.get("ghost_ge5"), "recall": d_.get("recall"), "flips": d_.get("flips"),
                          "train_dropout_marginals": DROPM}  # recorded with every comparison (Astra 08:30)
    if cond == "hybrid":
        CMP = d_
MODES_OK = {"ok": not mode_why and set(res["modes"]) == set(MODES), "why": mode_why}
# camera swap on paired evidence
rd, hyb = DIRS["swap"], DIRS["hybrid"]
REF = {r_: (DIRS["shared"].get(r_) or hyb) for r_ in DON}
mf, md, mr = (merged_scores(w) for w in ("swap_full", "swap_postdonor", "swap_postrecv"))
full, dn, rc = (json.load(open(f))["drives"] for f in (mf, md, mr))
runs = {}
for d in dev:
    for sd in (0, 1):
        rid = f"{d}_s{sd}"
        if rid not in full or rid not in rc:
            continue
        S = checked(f"{rd}/{rid}.pt")
        H = checked(f"{REF[d]}/{rid}.pt")
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
# decision
res["decision"] = WP.decide_s1(CMP, step, a.stage, SWAP, PROD, MODES_OK, prior)
SL = WP.Ledger(a.score_ledger, f"score_{a.name}", write=False)
DTX = int(M["dev_ticks_x_seeds"])
res["scoring"] = {"ledger": a.score_ledger, "gpu_h": round(SL.by_kind().get(f"score_{a.name}", 0.0), 3),
                  "cap_h": round(WP.score_cap_h(DTX, float(PR["score_tick_s"])), 3), "device": PR["device"],
                  "note": "2080 Ti scoring hours: own ledger and cap, reported separately from the S1 slice budget"}
res["timing_s"] = timing; res["render_slice_h"] = RD["render_slice_h"]
res["ledger_hours"] = round(LEDGER.hours(), 3); res["ledger_by_kind"] = LEDGER.by_kind()
if LEDGER.hours() > a.budget_hours:  # Astra S1c item 4
    res["decision"]["verdict"] = f"INVALID (S1 budget exceeded at publication: {LEDGER.hours():.2f} h > {a.budget_hours} h; no marker)"
json.dump(res, open(f"{a.wroot}/{a.name}_panel.json", "w"), indent=1, default=str)
if a.stage == "24h" and step >= WP.FLOOR_UPDATES and not res["decision"]["verdict"].startswith(("INVALID", "unresolved")):
    json.dump({"verdict": res["decision"]["verdict"], "panel": f"{a.wroot}/{a.name}_panel.json", "step": step, "ckpt": a.ckpt},
              open(MARK, "w"), indent=1)
print("WAN_PANEL", json.dumps({"step": step, "stage": a.stage, "verdict": res["decision"]["verdict"], "gates": res["decision"]["gates"],
                               "swap_blocked": SWAP["blocked"], "modes": {k: v["car_f1"].get("A") for k, v in res["modes"].items()},
                               "scoring_gpu_h": res["scoring"]["gpu_h"], "scoring_cap_h": res["scoring"]["cap_h"]}, default=str), flush=True)
LEDGER.stop()
