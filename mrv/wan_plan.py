"""PLAN-video S1: torch-free planning rules shared by the cache builder, trainer, rollout and panel (tested by
scripts/wan_tests.py with plain python3). Astra review codereview-wan-s1-astra.md items 3, 10, 11, 12.

Streams vs eligibility (item 3). scripts/temporal_manifest.py starts a new `seq` at OFFSET-ONLY rejects (a camera 45-100 ms off:
ok = False, hard = False); those ticks still have keys, cameras (nearest bundle) and guides. So the codec stream (Wan VAE,
rollout windows) runs THROUGH them: a segment breaks only at a hard tick (camera > 100 ms off or a repeated bundle) or a tick-id
gap. Eligibility is separate: a training clip needs every covered tick ok (loss on all its latents); the scorer scores ok and not
hard ticks (unchanged signed rule).
"""
import json, math, os, statistics, threading, time

PASS_CAR_F1 = 0.4456          # A@20k dev car F1 (signed)
EXT_LOW = PASS_CAR_F1 - 0.03  # extension band [0.4156, 0.4456)
FLOOR_UPDATES = 12000         # A1: below this the S1 verdict is unresolved
REL_BAR = -0.094              # ID and W each > 9.4 % better than A


# ------------------------------------------------------------------------------------------------------------ splits
SEALED = ("decision", "final")


def check_split_membership(drives, lists, allowed=("train", "dev"), unseal=False):
    """every RESOLVED scene id (explicit names and list files included) must belong to an allowed split; sealed splits
    (decision / final) need unseal; a scene in no list is refused (Astra S1b item 2; mrv.guide.drive_list accepts any
    scene-* name or file). Returns the drives unchanged or raises SystemExit."""
    member = {}
    for sp, v in lists.items():
        if isinstance(v, list):
            for d in v:
                member.setdefault(d, set()).add(sp)
    bad = {}
    for d in drives:
        sps = member.get(d, set())
        if not sps:
            bad[d] = "in no split list"
        elif sps & set(SEALED) and not unseal:
            bad[d] = f"sealed ({sorted(sps & set(SEALED))})"
        elif not sps & (set(allowed) | (set(SEALED) if unseal else set())):
            bad[d] = f"split {sorted(sps)} not allowed here ({list(allowed)})"
    if bad:
        raise SystemExit(f"drive split check failed for {len(bad)} drives: {dict(list(bad.items())[:5])}")
    return list(drives)


# ------------------------------------------------------------------------------------------------------------ segments
def segments(ticks, hard):
    """maximal runs of non-hard ticks with consecutive tick ids (seq changes at soft rejects do NOT break a run)"""
    out, cur = [], []
    for i, (t, h) in enumerate(zip(ticks, hard)):
        if h:
            if cur:
                out.append(cur)
            cur = []
            continue
        if cur and int(t) != int(ticks[cur[-1]]) + 1:
            out.append(cur); cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def stream_specs(segs, phases=4, clip_latents=5):
    """one codec stream per (segment, phase): frames seg[phase : phase + 4L - 3], L = 1 + (len - 1) // 4 (no padding);
    kept only if it holds at least one mid-stream clip (L >= 1 + clip_latents)"""
    out = []
    for si, seg in enumerate(segs):
        for ph in range(phases):
            fr = seg[ph:]
            if not fr:
                continue
            L = 1 + (len(fr) - 1) // 4
            if L < 1 + clip_latents:
                continue
            out.append({"seg": si, "phase": ph, "pos0": fr[0], "n_frames": 4 * L - 3, "L": L})
    return out


def mid_candidates(streams, ok, hard, clip_latents=5):
    """(stream index, j0) with j0 >= 1 and every covered tick (stream frames 4 j0 - 3 .. 4 (j0 + NL - 1)) ok and not hard"""
    out = []
    for k, st in enumerate(streams):
        p0, L = st["pos0"], st["L"]
        for j0 in range(1, L - clip_latents + 1):
            fr = range(p0 + 4 * j0 - 3, p0 + 4 * j0 + 4 * clip_latents - 3)
            if all(ok[i] and not hard[i] for i in fr):
                out.append((k, j0))
    return out


def start_candidates(segs, ok, hard, frames=17, stride=4):
    """enc index s of a true-start clip: s .. s + frames - 1 inside one segment, all ok; s % stride == 0"""
    out = []
    for seg in segs:
        pos = {i: q for q, i in enumerate(seg)}
        for i in seg:
            if i % stride or pos[i] + frames > len(seg):
                continue
            if all(ok[j] and not hard[j] for j in seg[pos[i]:pos[i] + frames]):
                out.append(i)
    return out


def latent_spans(L, first_single=True):
    """frame offsets of each latent: stream / clip start [0]*4 then 4j-3..4j; mid-stream window: 4 frames each"""
    if first_single:
        return [[0, 0, 0, 0]] + [[4 * j - 3 + k for k in range(4)] for j in range(1, L)]
    return [[4 * j + k for k in range(4)] for j in range(L)]


def token_slots(span, mode):
    if mode == "last":
        return [span[-1]]
    if mode == "ends":
        return [span[0], span[-1]]
    return list(span)


def plan_windows(L, W):
    """windows over L stream latents of W latents (W <= 0: one window); the tail window overlaps so every window has W latents
    when L >= W; returns [(w0, w1, new0)] with new0 = first latent this window contributes"""
    W = L if W <= 0 else W
    starts = list(range(0, L, W))
    if L >= W and starts[-1] + W > L:
        starts[-1] = L - W
    out, done = [], 0
    for w0 in starts:
        w1 = min(w0 + W, L)
        out.append((w0, w1, max(done, w0))); done = w1
    return out


def window_first_frames(L, W):
    """stream frame offset of the first NEW tick of every window (latent 0 -> frame 0, latent j -> frame 4j - 3)"""
    return [0 if n0 == 0 else 4 * n0 - 3 for _, _, n0 in plan_windows(L, W)]


def snap_swap(segs, swap_at, W):
    """first window start (enc position) at or after swap_at, so the swapped run keeps the unswapped run's window borders"""
    best = None
    for seg in segs:
        L = 1 + -(-(len(seg) - 1) // 4)
        for f in window_first_frames(L, W):
            if f < len(seg) and seg[f] >= swap_at and (best is None or seg[f] < best):
                best = seg[f]
    return best


# ------------------------------------------------------------------------------------------------------------ decisions
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def production_ok(ck_meta, expected_batch=8):
    """S1b item 11: only a PRODUCTION checkpoint can carry a verdict: global batch 8, not a timing / overfit / test run.
    ck_meta: {"timing", "recipe": {...}, "args": {...}}"""
    why = []
    if ck_meta.get("timing"):
        why.append("timing / test checkpoint")
    rc = ck_meta.get("recipe") or {}
    if rc.get("batch") != expected_batch:
        why.append(f"global batch {rc.get('batch')} != {expected_batch}")
    if rc.get("overfit"):
        why.append("overfit run")
    kind = (ck_meta.get("args") or {}).get("kind") or "train"
    if kind != "train":
        why.append(f"kind {kind}")
    return {"ok": not why, "why": why}


def decide_s1(cmp, step, stage, swap, prod=None, modes=None, prior_24h=None):
    """PLAN-video S1 gates (signed text + Astra items 12 / S1b 11). cmp = TCOMPARE dict of the HYBRID mode (candidate A, signed
    A@20k B, A's k=3 count list, --floor 0 --cold_ref B). stage: 'check' (informational), '24h' (the ONE first read: pass /
    one 12 h extension / stop), 'final' (only after a 24 h EXTEND verdict; pass bar unchanged).
    swap: donor_following() result (or bool / None). prod: production_ok() result. modes: {"ok": bool, "why": [...]} = all four
    conditioning modes rendered at full length with complete 6 x 2 panels. prior_24h: verdict string of an earlier 24 h read."""
    g = {"complete": cmp.get("drives") == 6 and cmp.get("runs") == 12 and not cmp.get("unpaired_drives") and cmp.get("cold_ref") == "B"}
    vals = {}
    for k in ("car_f1", "road_iou"):
        vals[k] = (cmp.get(k, {}).get("A"), cmp.get(k, {}).get("B"))
    for k in ("ID", "W"):
        vals[k] = (cmp.get(k, {}).get("rel"), cmp.get(k, {}).get("can_pass"))
    finite = all(_num(v) for k in ("car_f1", "road_iou") for v in vals[k]) and all(_num(vals[k][0]) for k in ("ID", "W"))
    g["finite"] = finite
    g["car_f1"] = finite and vals["car_f1"][0] >= PASS_CAR_F1
    g["road_iou"] = finite and vals["road_iou"][0] >= vals["road_iou"][1]
    g["ID"] = finite and vals["ID"][1] is True and vals["ID"][0] < REL_BAR
    g["W"] = finite and vals["W"][1] is True and vals["W"][0] < REL_BAR
    g["swap"] = (swap.get("ok") is True) if isinstance(swap, dict) else (swap is True)
    g["production"] = bool(prod and prod.get("ok"))
    g["modes"] = bool(modes and modes.get("ok"))
    others = all(g[k] for k in ("complete", "finite", "road_iou", "ID", "W", "swap", "production", "modes"))
    all_pass = others and g["car_f1"]
    in_band = finite and EXT_LOW <= vals["car_f1"][0] < PASS_CAR_F1
    if stage not in ("check", "24h", "final"):
        raise ValueError(stage)
    if step < FLOOR_UPDATES:
        verdict = "unresolved (below the 12k-update floor; A1)"
    elif stage == "check":
        verdict = "informational check (no verdict)"
    elif not g["production"] or not g["modes"]:
        verdict = "INVALID (not a production checkpoint with batch 8, or the four full-length mode panels are incomplete)"
    elif stage == "24h":
        if prior_24h is not None:
            verdict = "INVALID (the 24 h read happens once; one already exists)"
        else:
            verdict = "PASS -> S2" if all_pass else ("EXTEND 12 h (car F1 in [A-0.03, A), every other gate passed)" if (others and in_band)
                                                     else "STOP -> re-council")
    else:  # final
        if not (isinstance(prior_24h, str) and prior_24h.startswith("EXTEND")):
            verdict = "INVALID (a final read needs the qualifying 24 h EXTEND verdict)"
        else:
            verdict = "PASS -> S2" if all_pass else "STOP -> re-council (the pass bar stays car F1 >= 0.4456)"
    return {"step": step, "stage": stage, "gates": g, "values": vals, "all_pass": all_pass, "extension_band": in_band, "verdict": verdict,
            "production_why": (prod or {}).get("why"), "modes_why": (modes or {}).get("why")}


def donor_following(runs, margin=0.10, drives=6, seeds=("0", "1")):
    """camera-swap check on PAIRED evidence (S1b items 8-9; rule pre-registered here, council to confirm). runs: run id
    "<receiver>_s<seed>" -> {"donor_f1", "receiver_f1": post-swap car F1 vs the donor / receiver keys on the SAME frames,
    "donor_frames", "receiver_frames": placement frames scored, "windows_match": swap run's windows / noise == the unswapped
    receiver's, "lag": {"excluded": bool, "recovered": bool, "start": int | None}}.
    Blocks (ok False) unless: exactly drives x seeds run ids; every run's windows match; every run scored the same nonzero
    frame count for both keys; every drive has >= 1 lag-eligible run. Then ok = mean over drives (seeds pooled) donor F1 >=
    receiver F1 + margin AND the median recovery START over eligible runs <= 1 tick (non-recoveries = infinity)."""
    why = []
    rids = sorted(runs)
    drv = sorted({r.rsplit("_s", 1)[0] for r in rids})
    if len(drv) != drives or any({r.rsplit("_s", 1)[1] for r in rids if r.rsplit("_s", 1)[0] == d} != set(seeds) for d in drv) or len(rids) != drives * len(seeds):
        why.append(f"run ids: {len(rids)} runs over {len(drv)} drives (need {drives} x seeds {list(seeds)})")
    bad_w = [r for r in rids if runs[r].get("windows_match") is not True]
    if bad_w:
        why.append(f"windows / noise differ from the unswapped receiver on {bad_w}")
    bad_f = [r for r in rids if not (runs[r].get("donor_frames") and runs[r].get("donor_frames") == runs[r].get("receiver_frames"))]
    if bad_f:
        why.append(f"no common nonzero placement frames on {bad_f}")
    no_lag = [d for d in drv if not any(not runs[r]["lag"].get("excluded") for r in rids if r.rsplit("_s", 1)[0] == d)]
    if no_lag:
        why.append(f"no lag-eligible run on drives {no_lag}")
    excluded = [r for r in rids if runs[r]["lag"].get("excluded")]
    lags = [(runs[r]["lag"]["start"] if runs[r]["lag"].get("recovered") else float("inf")) for r in rids if not runs[r]["lag"].get("excluded")]
    per = {d: (statistics.mean(runs[r]["donor_f1"] for r in rids if r.rsplit("_s", 1)[0] == d),
               statistics.mean(runs[r]["receiver_f1"] for r in rids if r.rsplit("_s", 1)[0] == d)) for d in drv} if rids else {}
    dm = statistics.mean(v[0] for v in per.values()) if per else float("nan")
    rm = statistics.mean(v[1] for v in per.values()) if per else float("nan")
    lag = statistics.median(lags) if lags else float("inf")
    rule = bool(per) and dm >= rm + margin and lag <= 1
    return {"ok": (not why) and rule, "blocked": why, "donor_f1": dm, "receiver_f1": rm, "margin": margin, "median_lag_start": lag,
            "lag_values": lags, "lag_excluded_runs": excluded, "per_drive": per}


# ------------------------------------------------------------------------------------------------------------ forecast
def forecast_s1(t):
    """updates S1 training can still buy (S1b item 5). t: dict with
      budget_h, used_h (ledger, all kinds), done_kinds (set of ledger kinds already charged, e.g. {'timing', 'overfit'}),
      wall_s (production-loop seconds / update incl. prefetch wait), startup_s (one trainer start), probe_s, val_every,
      save_s, save_every, n_numbered (numbered saves in the planned run), job_h (wall limit of one training job),
      panel_h (measured cost of ONE full panel), n_panels, overfit_updates (2000), resume_updates (16 + 8 + 8),
      overfit_verdict_h (sampling + decode + scorer of the 8 clips).
    Pending overfit / resume test are charged at the same per-update cost plus their starts unless already in done_kinds."""
    per_upd = t["wall_s"] + t["probe_s"] / max(1, t["val_every"]) + t["save_s"] / max(1, t["save_every"])
    pend = {}
    if "overfit" not in t["done_kinds"]:
        pend["overfit_h"] = (t["startup_s"] + t["overfit_updates"] * t["wall_s"]) / 3600 + t["overfit_verdict_h"]
    if "resume_test" not in t["done_kinds"]:
        pend["resume_test_h"] = (3 * t["startup_s"] + t["resume_updates"] * t["wall_s"]) / 3600
    pend["panels_h"] = t["n_panels"] * t["panel_h"]
    avail = t["budget_h"] - t["used_h"] - sum(pend.values()) - t["n_numbered"] * t["save_s"] / 3600
    jobs = max(1, math.ceil(max(avail, 0) / t["job_h"]))
    avail_train = avail - jobs * t["startup_s"] / 3600
    ups = int(max(0.0, avail_train) * 3600 / per_upd)
    return {"per_update_s": round(per_upd, 3), "pending": {k: round(v, 3) for k, v in pend.items()}, "available_h": round(avail, 3),
            "training_jobs": jobs, "updates": ups, "floor_12k_reachable": ups >= FLOOR_UPDATES,
            "max_s_per_update_for_12k": round(max(0.0, avail_train) * 3600 / FLOOR_UPDATES, 3)}


def panel_cost_h(dev_ticks_x_seeds, roll_tick_s, roll_start_s, score_tick_s, score_start_s, n_modes=4):
    """one full panel: n_modes + 1 (swap) rollouts and n_modes + 3 scorings (swap full / post donor / post receiver) of every
    dev tick x 2 seeds, plus the one-time A@20k replay scoring (counted in every panel, conservative), at measured rates"""
    rolls = (n_modes + 1) * (roll_start_s + dev_ticks_x_seeds * roll_tick_s)
    scores = (n_modes + 3 + 1) * (score_start_s + dev_ticks_x_seeds * score_tick_s)
    return (rolls + scores) / 3600


# ------------------------------------------------------------------------------------------------------------- budget
class Ledger:
    """S1 slice-hour ledger (Astra items 11 / S1b 10). ONE jsonl file of heartbeats {"job", "kind", "t0", "t"[, "note"]}.
    Whole allocation: slurm/wan_teacher.sh and slurm/wan_panel.sh start ONE outer heartbeat for the entire allocation
    (python -m mrv.wan_plan beat ...; exports WAN_LEDGER_OUTER=1), so every process inside, including the timing scorer and
    the shell between steps, is charged exactly once; inner Ledger objects then only READ. A job's charge: finished (end
    note) -> last beat - t0; still beating (last beat within 2 intervals) -> now - t0; dead without an end note (preempted /
    killed) -> last beat - t0 + one beat interval (the last preemption interval, an upper bound)."""

    def __init__(s, path, kind, beat_s=60.0, write=None):
        s.path, s.kind, s.beat_s = path, kind, beat_s
        s.t0 = time.time()
        s.write = (not os.environ.get("WAN_LEDGER_OUTER")) if write is None else write
        s.job = f"{'alloc' if kind.startswith('alloc') else 'proc'}-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}-{int(s.t0)}"
        s._stop = threading.Event()
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def beat(s, note=None):
        if not s.path or not s.write:
            return
        rec = {"job": s.job, "kind": s.kind, "t0": s.t0, "t": time.time(), "beat_s": s.beat_s}
        if note:
            rec["note"] = note
        with open(s.path, "a") as f:  # leading newline: a torn last line (killed writer) never swallows this beat
            f.write("\n" + json.dumps(rec) + "\n")

    def start(s):
        s.beat("start")
        def run():
            while not s._stop.wait(s.beat_s):
                s.beat()
        threading.Thread(target=run, daemon=True).start()
        return s

    def stop(s, note="end"):
        s._stop.set(); s.beat(note)

    @staticmethod
    def read(path):
        jobs = {}
        if not path or not os.path.exists(path):
            return jobs
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line after a kill
            j = jobs.setdefault(r["job"], {"kind": r["kind"], "t0": r["t0"], "t": r["t0"], "ended": False, "beat_s": r.get("beat_s", 60.0)})
            j["t"] = max(j["t"], r["t"])
            j["ended"] = j["ended"] or r.get("note") in ("end", "failed", "parity", "incomplete", "cold_ref", "budget")
        return jobs

    @staticmethod
    def charge_h(j, now=None):
        now = time.time() if now is None else now
        if j["ended"]:
            return (j["t"] - j["t0"]) / 3600
        if now - j["t"] <= 2 * j["beat_s"]:
            return (now - j["t0"]) / 3600  # still beating
        return (j["t"] - j["t0"] + j["beat_s"]) / 3600  # dead without an end note: charge the last interval too

    def hours(s, live=True):
        jobs = Ledger.read(s.path)
        tot = sum(Ledger.charge_h(j) for j in jobs.values())
        if live and s.write and s.job not in jobs:
            tot += (time.time() - s.t0) / 3600
        return tot

    def by_kind(s):
        out = {}
        for j in Ledger.read(s.path).values():
            out[j["kind"]] = out.get(j["kind"], 0.0) + Ledger.charge_h(j)
        return out


def _beat_main(argv):
    """python -m mrv.wan_plan beat <ledger> <kind>: heartbeat for a whole allocation until SIGTERM / SIGINT (writes 'end')"""
    import signal
    L = Ledger(argv[0], argv[1], write=True).start()
    done = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.set())
    done.wait()
    L.stop("end")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4 and sys.argv[1] == "beat":
        _beat_main(sys.argv[2:4])
    else:
        raise SystemExit("usage: python -m mrv.wan_plan beat <ledger.jsonl> <kind>")
