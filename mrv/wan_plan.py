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


def decide_s1(cmp, step, stage, swap_ok):
    """PLAN-video S1 gates (signed text + Astra item 12). cmp = TCOMPARE dict (candidate A, signed A@20k B, A's k=3 count list,
    --floor 0 --cold_ref B). stage: 'check' (informational), '24h' (the first read: pass / one 12 h extension / stop),
    'final' (after the extension: pass bar unchanged). swap_ok: donor-following result (bool) or None (not run)."""
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
    g["swap"] = swap_ok is True
    others = all(g[k] for k in ("complete", "finite", "road_iou", "ID", "W", "swap"))
    all_pass = others and g["car_f1"]
    in_band = finite and EXT_LOW <= vals["car_f1"][0] < PASS_CAR_F1
    if step < FLOOR_UPDATES:
        verdict = "unresolved (below the 12k-update floor; A1)"
    elif stage == "check":
        verdict = "informational check (no verdict)"
    elif stage == "24h":
        verdict = "PASS -> S2" if all_pass else ("EXTEND 12 h (car F1 in [A-0.03, A), every other gate passed)" if (others and in_band)
                                                 else "STOP -> re-council")
    elif stage == "final":
        verdict = "PASS -> S2" if all_pass else "STOP -> re-council (the pass bar stays car F1 >= 0.4456)"
    else:
        raise ValueError(stage)
    return {"step": step, "stage": stage, "gates": g, "values": vals, "all_pass": all_pass, "extension_band": in_band, "verdict": verdict}


def donor_following(post_donor_f1, post_receiver_f1, lag_starts, margin=0.10):
    """camera-swap check (pre-registered here; council to confirm): after the swap the render must follow the DONOR cameras:
    mean (over drives, seeds pooled) post-swap car F1 vs the donor keys >= vs the receiver keys + margin, and the median
    recovery START <= 1 tick (the signed interim lag rule; inf = not recovered)."""
    if not post_donor_f1 or set(post_donor_f1) != set(post_receiver_f1):
        return {"ok": False, "why": "missing / unpaired post-swap scores"}
    d = statistics.mean(post_donor_f1.values()); r = statistics.mean(post_receiver_f1.values())
    lag = statistics.median(lag_starts) if lag_starts else float("inf")
    ok = d >= r + margin and lag <= 1
    return {"ok": ok, "donor_f1": d, "receiver_f1": r, "margin": margin, "median_lag_start": lag}


# ------------------------------------------------------------------------------------------------------------- budget
class Ledger:
    """S1 slice-hour ledger (Astra item 11): every S1 GPU job (timing, overfit, resume test, training, panels) appends
    heartbeats {"job", "kind", "t0", "t"} to ONE jsonl file; a job's charge = last beat - t0, so work lost to a preemption is
    still debited (at most one beat interval unaccounted). hours() sums all jobs."""

    def __init__(s, path, kind, beat_s=60.0):
        s.path, s.kind, s.beat_s = path, kind, beat_s
        s.t0 = time.time()
        s.job = f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}-{int(s.t0)}"
        s._stop = threading.Event()
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def beat(s, note=None):
        if not s.path:
            return
        rec = {"job": s.job, "kind": s.kind, "t0": s.t0, "t": time.time()}
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
            j = jobs.setdefault(r["job"], {"kind": r["kind"], "t0": r["t0"], "t": r["t0"]})
            j["t"] = max(j["t"], r["t"])
        return jobs

    def hours(s, live=True):
        jobs = Ledger.read(s.path)
        tot = sum((j["t"] - j["t0"]) for k, j in jobs.items() if k != s.job) / 3600
        return tot + ((time.time() - s.t0) / 3600 if live else 0.0)

    def by_kind(s):
        out = {}
        for k, j in Ledger.read(s.path).items():
            out[j["kind"]] = out.get(j["kind"], 0.0) + (j["t"] - j["t0"]) / 3600
        return out
