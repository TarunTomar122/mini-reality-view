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


# ------------------------------------------------------------------------------------------------------------ dropout
def drop_masks(u, p_drop_both, p_drop_guide_only, p_drop_tokens_only):
    """JOINT input dropout (Fable 07:30 c1): one uniform u per clip picks exactly one of: both inputs dropped (the unconditional
    branch CFG / S2 distillation need), guide dropped only, tokens dropped only, nothing dropped. Returns (gdrop, tdrop) lists.
    Marginals: guide p_drop_both + p_drop_guide_only, tokens p_drop_both + p_drop_tokens_only."""
    pb, pg, pt = p_drop_both, p_drop_guide_only, p_drop_tokens_only
    if min(pb, pg, pt) < 0 or pb + pg + pt > 1:
        raise ValueError("drop shares must be >= 0 and sum to <= 1")
    g, t = [], []
    for x in u:
        both = x < pb
        go = pb <= x < pb + pg
        to = pb + pg <= x < pb + pg + pt
        g.append(bool(both or go)); t.append(bool(both or to))
    return g, t


NONE_MARGIN = 0.02  # hybrid samples must place cars better than the no-inputs control by more than this (Fable 08:30)


def overfit_status(loss_ok, f1_ok, rt_ok, f1_hybrid, f1_none, hist, prior_not_reached, falling_window=3, none_margin=NONE_MARGIN):
    """pre-registered overfit reading (Fable 07:30, tightened 08:30). In order:
      FAIL if the VAE round trip found no cars (the placement check cannot run);
      FAIL if hybrid car F1 <= no-inputs car F1 + none_margin, whatever the loss says (the direct test that inputs are used);
      PASS if both declared gates hold;
      NOT_REACHED (rerun longer / lr 3e-4) if the loss gate missed while placement passed or the fixed-noise eval loss was still
        falling (each of the last `falling_window` evals below the one before) -- allowed ONCE per gate key: with an earlier
        NOT_REACHED record for the same code / recipe (prior_not_reached >= 1) it is FAIL;
      FAIL otherwise."""
    if not rt_ok:
        return "FAIL (round trip found no cars: placement check invalid)"
    if not (f1_hybrid > f1_none + none_margin):
        return f"FAIL (hybrid car F1 {f1_hybrid:.3f} not above the no-inputs control {f1_none:.3f} + {none_margin}: inputs unused)"
    if loss_ok and f1_ok:
        return "PASS"
    vals = [v for _, v in hist]
    falling = len(vals) > falling_window and all(vals[-k] < vals[-k - 1] for k in range(1, falling_window + 1))
    if not loss_ok and (f1_ok or falling):
        if prior_not_reached >= 1:
            return "FAIL (second NOT_REACHED for the same code / recipe)"
        return "NOT_REACHED (loss ratio missed with " + ("placement passing" if f1_ok else "loss still falling") + "): rerun longer / lr 3e-4"
    return "FAIL"


# ------------------------------------------------------------------------------------------------------------ gate registry
GATE_RECIPE_KEYS = ("rank", "tok_ticks", "use_tokens", "shift", "sigma_dist", "lr", "warmup", "prompt_embed", "wan", "batch")


def gate_key(code_sha, recipe, base_sha, vae_sha, guide_version):
    """identity a gate verdict is bound to: trainer code, model recipe, frozen base, VAE, guide version. Data lists / caches /
    dropout / steps are excluded on purpose (the gates may run on the 48 S0c drives with dropout off)."""
    blob = json.dumps({"code_sha": code_sha, "recipe": {k: recipe.get(k) for k in GATE_RECIPE_KEYS}, "base_sha": base_sha,
                       "vae_sha": vae_sha, "guide_version": guide_version}, sort_keys=True, default=str)
    import hashlib
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def gate_records(path):
    out = []
    if path and os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def gate_append(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write("\n" + json.dumps({**rec, "time": time.time()}, default=str) + "\n")


def gates_ok(path, key):
    """MODE=train / extend precondition: a PASS overfit record AND a PASS resume_test record for this gate key"""
    recs = [r for r in gate_records(path) if r.get("gate_key") == key]
    have = {k: any(r.get("kind") == k and r.get("status") == "PASS" for r in recs) for k in ("overfit", "resume_test")}
    return all(have.values()), have


def prior_not_reached(path, key):
    return sum(1 for r in gate_records(path) if r.get("gate_key") == key and r.get("kind") == "overfit"
               and str(r.get("status", "")).startswith("NOT_REACHED"))


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
    # S1b / Astra 08:30: a verdict (PASS / EXTEND / STOP) needs COMPLETE evidence; missing or blocked swap evidence, an
    # incomplete or non-finite panel, a non-production checkpoint or missing mode panels -> INVALID (no 24 h marker), never STOP
    swap_evidence = isinstance(swap, dict) and not swap.get("blocked") and _num(swap.get("donor_f1")) and _num(swap.get("receiver_f1"))
    g["swap_evidence"] = bool(swap_evidence)
    # Astra S1c item 1: ID / W must be ELIGIBLE (temporal_compare can_pass: every counting drive paired with a shared transition);
    # can_pass False is incomplete evidence -> INVALID, never a threshold miss
    g["eligible"] = vals["ID"][1] is True and vals["W"][1] is True
    evidence = all(g[k] for k in ("complete", "finite", "production", "modes", "swap_evidence", "eligible"))
    others = evidence and all(g[k] for k in ("road_iou", "ID", "W", "swap"))
    all_pass = others and g["car_f1"]
    in_band = finite and EXT_LOW <= vals["car_f1"][0] < PASS_CAR_F1
    if stage not in ("check", "24h", "final"):
        raise ValueError(stage)
    missing = [k for k in ("complete", "finite", "production", "modes", "swap_evidence", "eligible") if not g[k]]
    if step < FLOOR_UPDATES:
        verdict = "unresolved (below the 12k-update floor; A1)"
    elif stage == "check":
        verdict = "informational check (no verdict)"
    elif not evidence:
        verdict = f"INVALID (incomplete evidence: {missing}; no verdict, no 24 h marker)"
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


# ------------------------------------------------------------------------------------------------------------ swap donors
def swap_segments(rt, rh, dt, dh, swap_pos):
    """the swap rollout's segmentation (scripts/wan_rollout.py): receiver ticks / hard before the swap, donor hard ticks and donor
    tick gaps after it (the swap itself is no break)"""
    n = len(rt); segs, cur = [], []
    hards = [rh[i] if i < swap_pos else dh[i] for i in range(n)]
    srct = [rt[i] if i < swap_pos else dt[i] for i in range(n)]
    for i in range(n):
        if hards[i]:
            if cur: segs.append(cur)
            cur = []; continue
        if cur and (rt[i] != rt[cur[-1]] + 1 or (i > swap_pos and cur[-1] >= swap_pos and srct[i] != srct[cur[-1]] + 1)):
            segs.append(cur); cur = []
        cur.append(i)
    if cur: segs.append(cur)
    return segs


def swap_preflight(recv, donor, W=5, swap_at=50, has_key=None):
    """CPU preflight of one receiver / donor assignment (Astra / Neon 08:30; scorer-exact per Astra S1c item 2).
    recv / donor: {"name", "ticks", "ok", "hard"}. Checks:
      full_length       : donor at least as long as the receiver (no truncation);
      windows_identical : the swapped run's segments / windows / noise keys == the unswapped receiver's (scripts/wan_rollout.py);
      common_frames     : post-swap positions the two post-swap scores share (scripts/score_rgb.py --post_swap_only): receiver
                          AND donor tick % 5 == 0, accepted for BOTH drives (ok and not hard), both keys present;
      lag_windows       : windows the signed scorer can use (temporal_metrics.py lag block, exactly): i in [swap, first_reset - 4)
                          with first_reset = the first restart after the swap; all 5 positions accepted with a donor key,
                          consecutive donor ticks, no restart inside (the swap position itself is a restart: a window start).
    Lag eligibility also needs a nonzero cold car F1 on those ticks, which only scoring shows."""
    hk = has_key or (lambda sc, t: True)
    rt, rh, ro = recv["ticks"], recv["hard"], recv["ok"]
    n = len(rt)
    res = {"receiver": recv["name"], "donor": donor["name"], "len_receiver": n, "len_donor": len(donor["ticks"])}
    res["full_length"] = len(donor["ticks"]) >= n and donor["name"] != recv["name"]
    if not res["full_length"]:
        res["ok"] = False; return res
    dt, dh, do = donor["ticks"][:n], donor["hard"][:n], donor["ok"][:n]
    segs0 = segments(rt, rh)
    sp = snap_swap(segs0, swap_at, W)
    res["swap_pos"] = sp
    if sp is None:
        res.update(windows_identical=False, ok=False); return res
    segs = swap_segments(rt, rh, dt, dh, sp)
    win = lambda sg_: [[rt[g_[0]], w0, w1, n0] for g_ in sg_ for w0, w1, n0 in plan_windows(1 + -(-(len(g_) - 1) // 4), W)]
    res["windows_identical"] = segs == segs0 and win(segs) == win(segs0)
    src_ok = [ro[i] if i < sp else do[i] for i in range(n)]       # the rollout's ok / hard = the source drive's
    src_hard = [rh[i] if i < sp else dh[i] for i in range(n)]
    acc = lambda i: src_ok[i] and not src_hard[i]
    res["common_frames"] = sum(1 for i in range(sp, n) if rt[i] % 5 == 0 and dt[i] % 5 == 0 and acc(i) and ro[i] and not rh[i]
                               and hk(recv["name"], rt[i]) and hk(donor["name"], dt[i]))
    rst = [bool(h) for h in src_hard]  # the rollout's restart flags: hard ticks + every window's first new tick
    for g_ in segs:
        L = 1 + -(-(len(g_) - 1) // 4)
        for f in window_first_frames(L, W):
            if f < len(g_):
                rst[g_[f]] = True
    first_reset = next((q for q in range(sp + 1, n) if rst[q]), n)
    kseg = {j for j in range(n) if acc(j) and hk(donor["name"], dt[j])}  # positions the scorer has key masks for
    lag_w = 0
    for i in range(sp, min(n, first_reset) - 4):
        w = [j for j in range(i, i + 5) if j in kseg]
        if len(w) < 5 or any(dt[w[q + 1]] != dt[w[q]] + 1 for q in range(4)) or any(rst[j] for j in w):
            continue
        lag_w += 1
    res["lag_windows"] = lag_w; res["first_reset"] = first_reset
    res["ok"] = bool(res["windows_identical"] and res["common_frames"] > 0 and lag_w > 0)
    return res


def plan_donors(drives, W=5, swap_at=50, has_key=None):
    """donor per dev receiver (Astra / Neon 08:30). Preference: (1) FULL mode: the conventional pairing (the drive half the list
    away, scripts/causal_rollout.py) if it passes swap_preflight, else the qualifying donor closest in length that is at least as
    long; (2) SHARED mode, only if no full-length donor qualifies: the longest donor whose length n passes the preflight with the
    receiver cut to n; the swap is then compared with an UNSWAPPED receiver run of the same length n (scripts/wan_panel.py renders
    it). Donors may repeat (score_rgb names donor-keyed files per receiver). drives: {name: {"ticks", "ok", "hard"}}.
    Returns {"donors": {recv: {"donor", "mode", "n"} | None}, "checks": {...}, "ok": bool}."""
    names = list(drives)
    out, checks = {}, {}
    cut = lambda D, n: {k: v[:n] for k, v in D.items()}
    for i, r in enumerate(names):
        R = {"name": r, **drives[r]}
        conv = names[(i + len(names) // 2) % len(names)]
        cands = [conv] + sorted((d for d in names if d not in (r, conv)), key=lambda d: (len(drives[d]["ticks"]) - len(R["ticks"]), d))
        chosen = None
        for d in cands:
            c = swap_preflight(R, {"name": d, **drives[d]}, W, swap_at, has_key)
            if c["ok"]:
                chosen = {"donor": d, "mode": "full", "n": None}; checks[r] = c; break
            checks.setdefault(r, c)
        if chosen is None:
            for d in sorted((d for d in names if d != r), key=lambda d: (-len(drives[d]["ticks"]), d)):
                n = len(drives[d]["ticks"])
                if n >= len(R["ticks"]):
                    continue
                c = swap_preflight({"name": r, **cut(drives[r], n)}, {"name": d, **drives[d]}, W, swap_at, has_key)
                if c["ok"]:
                    chosen = {"donor": d, "mode": "shared", "n": n}; checks[r] = {**c, "mode": "shared"}; break
        out[r] = chosen
    return {"donors": out, "checks": checks, "ok": all(v is not None for v in out.values())}


# ------------------------------------------------------------------------------------------------------------ replay parity
def _sha_file(f):
    import hashlib
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _sha_files(fs):
    import hashlib
    return hashlib.sha256("".join(os.path.basename(f) + _sha_file(f) for f in sorted(fs)).encode()).hexdigest()


def parity_binding(a_metrics, a_roll):
    """what the A@20k RGB-path replay parity is bound to (Astra S1b 7): the signed metrics file, every reference rollout file,
    the RGB adapter (scripts/score_rgb.py), the signed scorer (scripts/temporal_metrics.py) and the SD VAE files"""
    import glob
    from huggingface_hub import snapshot_download
    sd_dir = snapshot_download("stabilityai/sd-vae-ft-mse", local_files_only=True)
    return {"a_metrics": _sha_file(a_metrics), "a_roll": _sha_files(glob.glob(f"{a_roll}/*_s[0-9]*.pt")),
            "score_rgb": _sha_file("scripts/score_rgb.py"), "temporal_metrics": _sha_file("scripts/temporal_metrics.py"),
            "sd_vae": _sha_files([f for f in glob.glob(f"{sd_dir}/**", recursive=True) if os.path.isfile(f)])}


def parity_ok(record_path, binding):
    """True iff the recorded replay parity PASSED under exactly this binding"""
    if not os.path.exists(record_path):
        return False
    r = json.load(open(record_path))
    return bool(r.get("pass")) and r.get("binding") == binding


# ------------------------------------------------------------------------------------------------------------ forecast
def forecast_s1(t):
    """updates S1 training can still buy (S1b item 5). t: dict with
      budget_h, used_h (ledger, all kinds), done_kinds (set of ledger kinds already charged, e.g. {'timing', 'overfit'}),
      wall_s (production-loop seconds / update incl. prefetch wait), startup_s (one trainer start), probe_s, val_every,
      save_s, save_every, n_numbered (numbered saves in the planned run), job_h (wall limit of one training job),
      panel_h (measured cost of ONE full panel), n_panels, overfit_updates (2000), resume_updates (16 + 8 + 8),
      overfit_verdict_h (sampling of the 8 clips in all THREE modes + decodes + scorer), overfit_evals_h (the recurring
      fixed-noise evals every 100 updates).
    Pending overfit / resume test are charged at the same per-update cost plus their starts unless done_kinds says they have a
    PASS record (a failed gate is not finished: it will run again)."""
    per_upd = t["wall_s"] + t["probe_s"] / max(1, t["val_every"]) + t["save_s"] / max(1, t["save_every"])
    pend = {}
    if "overfit" not in t["done_kinds"]:
        pend["overfit_h"] = (t["startup_s"] + t["overfit_updates"] * t["wall_s"]) / 3600 + t["overfit_verdict_h"] + t.get("overfit_evals_h", 0.0)
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
def panel_with_shared_h(panel_h, donors, roll_tick_s, roll_start_s):
    """measured panel cost + one extra UNSWAPPED same-length hybrid rollout (2 seeds) per shared-mode receiver (Astra S1c item 4)"""
    extra = sum(roll_start_s + 2 * int(v["n"]) * roll_tick_s for v in donors.values() if v and v.get("mode") == "shared")
    return panel_h + extra / 3600


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
