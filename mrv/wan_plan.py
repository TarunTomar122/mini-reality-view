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
GATE_RECIPE_KEYS = ("rank", "tok_ticks", "use_tokens", "shift", "sigma_dist", "lr", "warmup", "prompt_embed", "wan", "batch",
                    "micro", "grad_ckpt")  # F5 (CONVERGED item 5): micro / grad_ckpt explicit and recorded


def gate_key(code_sha, recipe, base_sha, vae_sha, guide_version, manifest_sha=None):
    """identity a gate verdict is bound to: trainer code, model recipe (incl. micro / grad_ckpt), frozen base, VAE, guide version
    and the data manifest sha (CONVERGED item 3: the verified manifest of train + dev). Lists / dropout / steps are excluded."""
    blob = json.dumps({"code_sha": code_sha, "recipe": {k: recipe.get(k) for k in GATE_RECIPE_KEYS}, "base_sha": base_sha,
                       "vae_sha": vae_sha, "guide_version": guide_version, "manifest_sha": manifest_sha}, sort_keys=True, default=str)
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


SCORE_DEVICE = "2080 Ti"  # CONVERGED item 4: signed scoring moves off the slice to the 2080 Ti queue (where A@20k was scored)


METRIC_HF = ("stabilityai/sd-vae-ft-mse", "facebook/mask2former-swin-large-cityscapes-panoptic", "facebook/dinov2-small")
METRIC_HUB = ("raft_small", "alexnet")  # torchvision checkpoints the signed scorer loads (RAFT; LPIPS alex backbone)


def metric_weights():
    """Astra t1 item 12: sha of EVERY weight file the signed scorer (scripts/temporal_metrics.py:148-154) loads: SD VAE,
    Mask2Former, DINOv2 (HF cache), RAFT-small and AlexNet (torch hub cache), the LPIPS alex head (lpips package)"""
    import glob, importlib.util
    out = {}
    try:
        from huggingface_hub import snapshot_download
    except Exception:  # noqa: BLE001
        snapshot_download = None
    for rid in METRIC_HF:
        try:
            d = snapshot_download(rid, local_files_only=True)
            out[rid] = _sha_files([f for f in glob.glob(f"{d}/**", recursive=True) if os.path.isfile(f)])
        except Exception as e:  # noqa: BLE001
            out[rid] = f"missing: {type(e).__name__}"
    th = os.environ.get("TORCH_HOME") or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "torch")
    for pat in METRIC_HUB:
        fs = sorted(glob.glob(os.path.join(th, "hub", "checkpoints", f"{pat}*")))
        out[pat] = _sha_files(fs) if fs else "missing"
    sp = importlib.util.find_spec("lpips")
    lp = os.path.join(os.path.dirname(sp.origin), "weights", "v0.1", "alex.pth") if sp and sp.origin else ""
    out["lpips_alex"] = _sha_file(lp) if lp and os.path.exists(lp) else "missing"
    return out


def weights_resolved(w):
    """Astra t2 item 12: every metric weight entry is a RESOLVED sha256 (64 hex) of files found; a "missing" sentinel (or
    anything else) never matches, even against an equal sentinel"""
    import re
    return isinstance(w, dict) and set(w) == set(METRIC_HF) | set(METRIC_HUB) | {"lpips_alex"} and \
        all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in w.values())


def score_binding(a_metrics, a_roll):
    """the 2080 Ti scorer-parity binding: the A@20k replay binding (references, adapter, signed scorer, SD VAE weights) + the
    scoring wrapper code (scripts/wan_score.py) + every metric weight file (Astra t1 item 12); the device is recorded beside it
    and checked by score_parity_ok"""
    b = parity_binding(a_metrics, a_roll)
    b["wan_score"] = _sha_file("scripts/wan_score.py")
    b["weights"] = metric_weights()
    return b


def score_parity_ok(record_path, binding, device_sub=SCORE_DEVICE):
    """-> (ok, why). The off-slice scorer parity record (scripts/wan_score.py --parity_check) must: exist, have the A@20k replay
    PASS on the device, sharded == unsharded on the same inputs (same accepted ticks, same metrics), every part on ONE device
    whose name contains device_sub, and exactly this binding (references / adapter / scorer / SD VAE weights / wrapper code)."""
    if not record_path or not os.path.exists(record_path):
        return False, "no scorer-parity record"
    r = json.load(open(record_path))
    why = []
    if not r.get("replay_pass"):
        why.append("A@20k replay parity not PASS")
    if not r.get("sharded_equal"):
        why.append("sharded != unsharded")
    if device_sub not in str(r.get("device", "")):
        why.append(f"device {r.get('device')!r} is not a {device_sub}")
    if not weights_resolved((r.get("binding") or {}).get("weights")) or not weights_resolved(binding.get("weights")):
        why.append("metric weights not resolved to file digests (missing weight files)")
    if r.get("binding") != binding:
        diff = sorted(k for k in set(binding) | set(r.get("binding") or {}) if (r.get("binding") or {}).get(k) != binding.get(k))
        why.append(f"binding changed: {diff}")
    if not r.get("pass"):
        why.append("record pass is false")
    return not why, "; ".join(why) or "ok"


def score_binding_matches(part, rec):
    """a scoring output (score_rgb.py "binding") is admissible for a panel decision only if it was produced on the parity record's
    device with the parity record's scorer / adapter / SD VAE weights -> (ok, why)"""
    b = (part or {}).get("binding") or {}
    rb = rec.get("binding") or {}
    why = []
    if b.get("device") != rec.get("device"):
        why.append(f"device {b.get('device')!r} != parity device {rec.get('device')!r}")
    for k in ("score_rgb", "temporal_metrics", "sd_vae", "wan_score", "weights"):
        if b.get(k) is None or b.get(k) != rb.get(k):
            why.append(f"{k} differs from the parity record")
    if not weights_resolved(b.get("weights")):
        why.append("metric weights not resolved to file digests")
    if b.get("bound_before_scoring") is not True:  # Astra T3 item 12: as parity intake
        why.append("binding not recorded before scoring")
    return not why, "; ".join(why) or "ok"


def parity_parts_ok(parts, current, a_roll_inputs, a_metrics_sha):
    """Astra t1 item 12: every parity part (name, score json, receipt) must have succeeded (rc 0) and carry, AS BOUND AT
    EXECUTION, the current scorer / adapter / wrapper / metric weights; the unsharded part must have scored exactly the current
    A@20k reference files against the current signed metrics -> (ok, why list)"""
    why = []
    for name, j, rec in parts:
        b = (j or {}).get("binding") or {}
        if not rec or rec.get("rc") != 0:
            why.append(f"{name}: receipt missing or rc {None if not rec else rec.get('rc')}")
        for k in ("score_rgb", "temporal_metrics", "sd_vae", "wan_score", "weights"):
            if b.get(k) is None or b.get(k) != current.get(k):
                why.append(f"{name}: {k} at execution != current")
        if not weights_resolved(b.get("weights")) or not b.get("bound_before_scoring"):
            why.append(f"{name}: metric weights unresolved or not bound before scoring")
        if b.get("inputs") is None or not set(b["inputs"]) <= set(a_roll_inputs) or any(a_roll_inputs[k] != v for k, v in b["inputs"].items()):
            why.append(f"{name}: scored inputs are not the current A@20k reference files")
    un = [j for name, j, _ in parts if ((j or {}).get("binding") or {}).get("shard") == "0/1"]
    if len(un) != 1:
        why.append("no unique unsharded part")
    else:
        ub = un[0]["binding"]
        if ub.get("inputs") != a_roll_inputs:
            why.append("unsharded part did not score every A@20k reference file")
        if ub.get("parity_ref_sha") != a_metrics_sha:
            why.append("unsharded part compared against another signed metrics file")
    return not why, why


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


def panel_render_h(dev_ticks_x_seeds, roll_tick_s, load_s, n_modes=4, donors=None, enc_fill_tick_s=0.0, drive_ticks=None):
    """CONVERGED items 4 / 7: the SLICE cost of one panel is rendering only, with Wan loaded ONCE per panel process: one load +
    (n_modes + 1 swap) rollouts of every dev tick x 2 seeds + one unswapped same-length reference (2 seeds) per shared-mode donor,
    at the measured load-once rollout rate. Scoring runs on the 2080 Ti under its own ledger (panel_score_h)."""
    # Astra T3 / Neon C: roll_tick_s is the CACHE-HIT per-tick rate; the guide Wan-VAE encoding is filled once per distinct frame
    # sequence per drive (enc_fill_tick_s per tick): once for the ordinary renders (all modes and seeds share the receiver's
    # frames), once for the swap renders (donor frames after the swap, seed-independent), once per shared-mode reference of n ticks
    shared = sum(int(v["n"]) for v in (donors or {}).values() if v and v.get("mode") == "shared")
    if drive_ticks is None:  # legacy aggregate form (no per-receiver lengths): swaps priced at full length (conservative)
        ticks = (n_modes + 1) * dev_ticks_x_seeds + 2 * shared
        fills = 2 * (dev_ticks_x_seeds // 2) + shared  # ordinary + swap sequences per drive (dev ticks each) + shared references
        return (load_s + ticks * roll_tick_s + fills * enc_fill_tick_s) / 3600
    # Astra T3b / Neon C: each swap is rendered at its PLANNED length: the receiver's full length for a full-mode donor, the
    # shared length n for a shared-mode donor (the renderer cuts the swap to n); fills follow the same lengths
    if 2 * sum(drive_ticks.values()) != dev_ticks_x_seeds:
        raise ValueError(f"drive_ticks sum x 2 seeds {2 * sum(drive_ticks.values())} != dev_ticks_x_seeds {dev_ticks_x_seeds}")
    swap_len = {}
    for r, T in drive_ticks.items():
        v = (donors or {}).get(r)
        swap_len[r] = int(v["n"]) if (v and v.get("mode") == "shared") else int(T)
    ticks = n_modes * dev_ticks_x_seeds + 2 * sum(swap_len.values()) + 2 * shared
    fills = sum(int(T) for T in drive_ticks.values()) + sum(swap_len.values()) + shared
    return (load_s + ticks * roll_tick_s + fills * enc_fill_tick_s) / 3600


def panel_score_h(dev_ticks_x_seeds, score_tick_s, n_modes=4):
    """2080 Ti GPU-hours to score one panel: n_modes mode scorings + swap full + swap post-donor + swap post-receiver of every dev
    tick x 2 seeds (post-swap scorings priced at full length: conservative), at the measured 2080 Ti per-tick scoring rate"""
    return (n_modes + 3) * dev_ticks_x_seeds * score_tick_s / 3600


SCORE_CAP_FACTOR = 1.25  # CONVERGED item 4: score ledger cap = 1.25 x measured scoring GPU-h per panel (S0c precedent)


def score_cap_h(dev_ticks_x_seeds, score_tick_s, n_modes=4, factor=SCORE_CAP_FACTOR):
    return factor * panel_score_h(dev_ticks_x_seeds, score_tick_s, n_modes)


def panel_cost_h(dev_ticks_x_seeds, roll_tick_s, roll_start_s, score_tick_s, score_start_s, n_modes=4):
    """one full panel: n_modes + 1 (swap) rollouts and n_modes + 3 scorings (swap full / post donor / post receiver) of every
    dev tick x 2 seeds, plus the one-time A@20k replay scoring (counted in every panel, conservative), at measured rates"""
    rolls = (n_modes + 1) * (roll_start_s + dev_ticks_x_seeds * roll_tick_s)
    scores = (n_modes + 3 + 1) * (score_start_s + dev_ticks_x_seeds * score_tick_s)
    return (rolls + scores) / 3600


# ------------------------------------------------------------------------------------------------------------- off-slice scoring
def shard_by_drive(files, shard):
    """rollout files "<dir>/<scene>_s<seed>.pt" of shard "i/n": the drives (sorted scene names) with index % n == i, all seeds"""
    i, n = map(int, shard.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"bad shard {shard}")
    scene = lambda f: os.path.basename(f)[:-3].rsplit("_s", 1)[0]
    sc = sorted({scene(f) for f in files})
    return [f for f in sorted(files) if sc.index(scene(f)) % n == i]


def _same(x, y):
    return x == y or (isinstance(x, float) and isinstance(y, float) and (math.isclose(x, y, rel_tol=0, abs_tol=1e-9) or (x != x and y != y)))


def compare_sharded(unsharded, shards):
    """sharded == unsharded on the same inputs: runs partitioned (no run twice, none missing), every per-run field equal (same
    accepted ticks, same metrics), same input files / bytes. unsharded / shards: score_rgb.py outputs with "drives" and
    "binding". -> (ok, why list)"""
    why, merged, ins = [], {}, {}
    for j in shards:
        dup = set(merged) & set(j["drives"])
        if dup:
            why.append(f"runs in two shards: {sorted(dup)[:3]}")
        merged.update(j["drives"]); ins.update((j.get("binding") or {}).get("inputs") or {})
    U = unsharded["drives"]
    if set(merged) != set(U):
        why.append(f"shard runs != unsharded runs: {sorted(set(merged) ^ set(U))[:4]}")
    diff = [(r, k) for r in sorted(set(merged) & set(U)) for k in sorted(set(merged[r]) | set(U[r])) if not _same(merged[r].get(k), U[r].get(k))]
    if diff:
        why.append(f"sharded != unsharded on {len(diff)} fields, e.g. {diff[:6]}")
    if ins != ((unsharded.get("binding") or {}).get("inputs") or {}):
        why.append("shard inputs != unsharded inputs (different files or bytes)")
    return not why, why


SCORE_FLAGS = {"hybrid": (False, False, False), "guide": (False, False, False), "tokens": (False, False, False), "none": (False, False, False),
               "swap_full": (False, False, True), "swap_postdonor": (True, False, False), "swap_postrecv": (True, True, False)}
# (post_swap_only, as_receiver, lag_ref) per scoring (Astra t1 item 13)


def merge_score_shards(parts, what, manifest, manifest_sha, parity_rec, receipts, lag_inputs=None):
    """the panel decision's intake (CONVERGED item 4, Astra t1 items 12-14). parts: [(file name, score json, i, n)]. Each shard
    must: be bound to the RGB manifest the slice rendered (sha); have scored exactly its partition (shard i/n by drive) of THIS
    scoring's folder with the manifest's bytes; carry this scoring's flags (post-swap / receiver / lag reference, the lag
    reference's files == lag_inputs); match the parity record's device / scorer / adapter / wrapper / weights; have a successful,
    not over-cap receipt. Shards must cover the folder exactly once. -> (merged {"drives", "cold"}, why list)"""
    why, out, seen = [], {"drives": {}, "cold": None}, set()
    src = os.path.normpath(manifest["dirs"]["swap" if what.startswith("swap") else what])
    files = sorted(f for f in manifest["runs"] if os.path.dirname(f) == src)
    psw, rcv, lag = SCORE_FLAGS[what]
    for name, j, i, n in parts:
        b = j.get("binding") or {}
        if b.get("rgb_manifest_sha") != manifest_sha:
            why.append(f"{name}: scored RGB of another render (manifest {str(b.get('rgb_manifest_sha'))[:12]} != {str(manifest_sha)[:12]})"); continue
        if os.path.normpath(str(b.get("src"))) != src or b.get("shard") != f"{i}/{n}":
            why.append(f"{name}: scored {b.get('src')} shard {b.get('shard')}, expected {src} shard {i}/{n}"); continue
        exp = {os.path.basename(f): manifest["runs"][f]["sha256"] for f in shard_by_drive(files, f"{i}/{n}")}
        if b.get("inputs") != exp:
            why.append(f"{name}: inputs are not exactly this shard's manifest entries"); continue
        if bool(b.get("post_swap_only")) != psw or bool(b.get("as_receiver")) != rcv or bool(b.get("lag_ref")) != lag:
            why.append(f"{name}: scoring flags (post_swap {b.get('post_swap_only')}, receiver {b.get('as_receiver')}, lag {b.get('lag_ref')}) "
                       f"are not those of {what}"); continue
        if lag and b.get("lag_ref_inputs") != lag_inputs:
            why.append(f"{name}: lag reference files differ from the signed A@20k references"); continue
        ok_b, why_b = score_binding_matches(j, parity_rec)
        if not ok_b:
            why.append(f"{name}: {why_b}"); continue
        rc = receipts.get(name)
        if not rc or rc.get("rc") != 0 or rc.get("over_cap"):
            why.append(f"{name}: no successful in-cap receipt ({rc})"); continue
        dup = set(out["drives"]) & set(j["drives"])
        if dup:
            why.append(f"{name}: runs scored twice {sorted(dup)[:3]}")
        out["drives"].update(j["drives"]); seen |= set(exp)
        if j.get("cold") is not None:
            out["cold"] = j["cold"]
    if not why and seen != {os.path.basename(f) for f in files}:
        why.append(f"shards do not cover {src} exactly ({len(seen)} of {len(files)} files)")
    return out, why


def ledger_slurm_id(jid):
    """Slurm job id of a ledger job key "<alloc|proc>-<SLURM_JOB_ID>-<pid>-<t0>" (Neon 11:30 P0: the 2080 Ti heartbeats of
    slurm/wan_score.sh were written as "proc-<job>-..." with kind score_*; both prefixes are one allocation's charge) or None"""
    parts = str(jid).split("-")
    return parts[1] if len(parts) >= 4 and parts[0] in ("alloc", "proc") else None


def active_reserved_h(reservations, jobs, kind, now=None, exclude=None):
    """Astra t1 item 14: hours still reserved by OTHER live scoring workers of this panel: per reservation {job, est_h}, if that
    Slurm job's ledger entry (kind) is still beating, its estimate minus what it has already been charged"""
    now = time.time() if now is None else now
    tot = 0.0
    for r in reservations:
        if exclude is not None and str(r.get("job")) == str(exclude):
            continue
        for jid, j in jobs.items():
            if j.get("kind") == kind and ledger_slurm_id(jid) == str(r.get("job")) and not j.get("ended") and now - j["t"] <= 2 * j.get("beat_s", 60.0):
                # Astra t2 item 14: the reservation holds its estimate from the moment it was MADE (not the allocation start)
                tot += max(0.0, float(r.get("est_h", 0.0)) - (now - float(r.get("t", now))) / 3600)
                break
    return tot


def reconcile_scoring(receipts, jobs, kind, cap_h, now=None):
    """Astra t2 item 14: before publication every accepted shard's receipt job must have a FINISHED charge of this panel's kind in
    the score ledger, no worker of the kind may still be live, and the total charge must be within the cap -> (ok, why, spent)"""
    why = []
    kj = {jid: j for jid, j in jobs.items() if j.get("kind") == kind}
    ids = {ledger_slurm_id(jid) for jid in kj} - {None}
    if not kj:
        why.append(f"no {kind} charges in the score ledger")
    miss = sorted({str(r.get("job")) for r in receipts} - ids)
    if miss:
        why.append(f"receipt jobs without a ledger charge: {miss[:4]}")
    live = sorted(jid for jid, j in kj.items() if not j.get("ended"))
    if live:
        why.append(f"scoring workers still live / unfinished in the ledger: {live[:3]}")
    spent = sum(Ledger.charge_h(j, now) for j in kj.values())
    if spent > cap_h:
        why.append(f"score ledger {spent:.3f} h > cap {cap_h:.3f} h")
    return not why, why, spent


def score_headroom(spent_h, others_reserved_h, own_est_h, cap_h):
    """aggregate headroom across active workers: spent (all charged so far) + other live workers' remaining reservations + this
    worker's estimate must fit under the cap -> (ok, total)"""
    tot = spent_h + others_reserved_h + own_est_h
    return tot <= cap_h, round(tot, 4)


def render_binding_ok(manifest, ckpt_sha256, donors_sha256, swap_dir=None):
    """Astra t1 item 15: the decision uses the checkpoint and donor plan the slice rendered: digests recorded at render must equal
    the files now, and every rendered file must carry the checkpoint's digest -> (ok, why)"""
    why = []
    if manifest.get("ckpt_sha256") != ckpt_sha256:
        why.append(f"checkpoint changed since rendering ({str(manifest.get('ckpt_sha256'))[:12]} -> {str(ckpt_sha256)[:12]})")
    if manifest.get("donors_sha256") != donors_sha256:
        why.append("donor plan file changed since rendering")
    bad = [f for f, e in (manifest.get("runs") or {}).items() if e.get("ckpt_sha") != str(manifest.get("ckpt_sha256"))[:16]
           or e.get("ckpt_sha256") != manifest.get("ckpt_sha256")]
    if bad:  # Astra t2 item 15: the bytes each rollout CONSUMED (its own digest) must be the pinned checkpoint
        why.append(f"{len(bad)} rendered files from another checkpoint, e.g. {bad[:2]}")
    sd = os.path.normpath(swap_dir or (manifest.get("dirs") or {}).get("swap", ""))
    badd = [f for f, e in (manifest.get("runs") or {}).items() if os.path.dirname(f) == sd and e.get("donors_sha256") != manifest.get("donors_sha256")]
    if badd or not any(os.path.dirname(f) == sd for f in (manifest.get("runs") or {})):
        why.append(f"swap renders did not consume the pinned donor plan ({len(badd)} files)")
    if not manifest.get("donor_plan"):
        why.append("no donor plan stored at render")
    return not why, "; ".join(why) or "ok"


# ------------------------------------------------------------------------------------------------------------- F5 / re-time
def matched_updates(rows, ref_key, rtol=0.02, gn_rtol=0.05):
    """scalar part of the F5 matched-update check (loss and grad norm from the SAME params / inputs / noise / seed under each
    setting); the decisive part is delta_check (per-tensor Adam deltas, Astra t1 item 2). -> {"ref", "rows", "all_ok"}"""
    ref = rows[ref_key]; out = {}
    for k, r in rows.items():
        dl = abs(r["loss"] - ref["loss"]) / max(abs(ref["loss"]), 1e-12)
        dg = abs(r["grad_norm"] - ref["grad_norm"]) / max(abs(ref["grad_norm"]), 1e-12)
        fin = _num(r["loss"]) and _num(r["grad_norm"])
        out[k] = {**r, "d_loss_rel": round(dl, 6), "d_gn_rel": round(dg, 6), "ok": bool(fin and dl <= rtol and dg <= gn_rtol)}
    return {"ref": ref_key, "rtol": rtol, "gn_rtol": gn_rtol, "rows": out, "all_ok": all(v["ok"] for v in out.values())}


DELTA_RTOL = 0.05       # per-tensor ||d - d_ref|| / ||d_ref|| after ONE Adam step (bf16 accumulation order; reversal = 2.0)
DELTA_ZERO_ABS = 1e-12  # a tensor whose reference delta is zero must stay (numerically) zero


def delta_check(rows, rtol=DELTA_RTOL, zero_abs=DELTA_ZERO_ABS):
    """Astra t1 item 2: rows {tensor: {"ref": ||d_ref||, "diff": ||d - d_ref||, "norm": ||d||}} for ONE setting against the
    reference, where d = parameter change of ONE Adam step from identical weights, optimizer state, batch and RNG. Every tensor
    must match (relative L2 <= rtol; zero reference -> zero delta); reversed or permuted gradients fail even when the loss and
    the global grad norm agree. -> {"ok", "tensors", "worst", "worst_rel", "failing"}"""
    worst, wrel, bad = None, 0.0, []
    for n, r in rows.items():
        if not (_num(r.get("ref")) and _num(r.get("diff")) and _num(r.get("norm"))):
            bad.append(n); continue
        if r["ref"] <= zero_abs:
            rel = 0.0 if r["norm"] <= max(zero_abs, 1e-9) else float("inf")
        else:
            rel = r["diff"] / r["ref"]
        if rel > wrel:
            worst, wrel = n, rel
        if rel > rtol:
            bad.append(n)
    return {"ok": bool(rows) and not bad, "tensors": len(rows), "worst": worst, "worst_rel": wrel if wrel != float("inf") else "inf",
            "failing": bad[:8], "n_failing": len(bad)}


def select_setting(fits, match):
    """fastest fitting (grad_ckpt, micro) setting whose matched-update rows are ok (scalar AND per-tensor delta check) ->
    {"key", "micro", "grad_ckpt", "gpu_s"} or None"""
    grid = {f"gc{g}_m{m}" for g in SWEEP_GC for m in SWEEP_MICRO}  # Astra t2 item 4: never a cell outside the declared grid
    ok = [k for k in fits if k in grid and int(fits[k].get("grad_ckpt", -1)) in SWEEP_GC and int(fits[k].get("micro", -1)) in SWEEP_MICRO
          and (match.get("rows") or {}).get(k, {}).get("ok")]
    if not ok:
        return None
    k = min(ok, key=lambda x: (fits[x]["gpu_s"], -fits[x]["grad_ckpt"], fits[x]["micro"]))
    return {"key": k, "micro": int(fits[k]["micro"]), "grad_ckpt": int(fits[k]["grad_ckpt"]), "gpu_s": fits[k]["gpu_s"]}


UPDATE_TARGET = 13200       # CONVERGED item 7: the forecast must buy >= floor + 10 %
RETIME_MIN_UPDATES = 200    # Astra: >= 200 consecutive production updates
RETIME_QUEUE_FRAC = 0.05
RETIME_WALL_OVER_GPU = 1.10
RETIME_BLOCK = 32           # Astra t1 item 4: block width is fixed (wider blocks average a failing 32-update block away)
SWEEP_GC = (1, 0)           # declared F5 sweep
SWEEP_MICRO = (1, 2, 4, 8)
GLOBAL_BATCH = 8
N_TRAIN, N_DEV = 144, 6     # Astra t1 item 3: the production lists, exactly
CANON_LISTS = "data/temporal_lists.json"
SENS_STEP = 32              # Astra t1 item 1: positive input sensitivity is required after exactly this many optimizer steps
BUDGET_H, JOB_H = 24.0, 12.0
STARTUP_PARTS = ("pre_trainer_s", "process_to_ready_s", "parity_check_s", "frozen_snapshot_s", "first_update_s", "guard_verify_s",
                 "first_probe_s", "resume_load_s")  # parity_check_s: production's scorer-parity binding I/O per start (Astra t2 item 7)
EVIDENCE_FILES = ("scripts/train_wan_teacher.py", "mrv/wan_cond.py", "mrv/wan_plan.py", "mrv/wan_stage.py", "scripts/wan_forecast.py",
                  "scripts/wan_panel.py", "scripts/wan_rollout.py", "scripts/wan_token_check.py", "slurm/wan_teacher.sh", "mrv/guide.py")


def ids_sha(ids):
    import hashlib
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def check_lists(train, dev, canon, mode):
    """Astra t1 item 3. train / dev as given to the job (duplicates kept), canon = data/temporal_lists.json. mode 'full' (timing,
    train, extend): exactly the N_TRAIN distinct canonical train IDs and the N_DEV distinct canonical dev IDs. mode 'gate'
    (overfit, resume test): any subset, but every ID inside its ORIGINAL canonical split. -> (ok, why)"""
    why = []
    ct, cd = list(canon.get("train", [])), list(canon.get("dev", []))
    if len(set(ct)) != N_TRAIN or len(ct) != N_TRAIN or len(set(cd)) != N_DEV or len(cd) != N_DEV:
        why.append(f"canonical lists have {len(set(ct))}/{len(ct)} train and {len(set(cd))}/{len(cd)} dev IDs (need {N_TRAIN} / {N_DEV} distinct)")
    if len(set(train)) != len(train) or len(set(dev)) != len(dev):
        why.append("duplicate IDs in the job's lists")
    if mode == "full":
        if sorted(set(train)) != sorted(ct) or len(train) != N_TRAIN:
            why.append(f"train list is not the exact {N_TRAIN} canonical train IDs ({len(set(train))} given, {len(set(train) - set(ct))} foreign)")
        if sorted(set(dev)) != sorted(cd) or len(dev) != N_DEV:
            why.append(f"dev list is not the exact {N_DEV} canonical dev IDs")
    elif mode == "gate":
        if not set(train) <= set(ct):
            why.append(f"gate train IDs outside the canonical train split: {sorted(set(train) - set(ct))[:3]}")
        if not set(dev) <= set(cd):
            why.append(f"gate dev IDs outside the canonical dev split: {sorted(set(dev) - set(cd))[:3]}")
    else:
        raise ValueError(mode)
    return not why, "; ".join(why) or "ok"


def check_modes(timing, overfit, resume_test, production):
    """Astra t1 item 10: exactly one run mode (an overlap such as --timing --overfit would skip the re-time gate)"""
    on = [n for n, v in (("--timing", timing), ("--overfit", overfit), ("--resume_test", resume_test), ("--production", production)) if v]
    if len(on) > 1:
        raise SystemExit(f"overlapping run modes {on}: exactly one of --timing / --overfit / --resume_test / --production (or none)")
    return on[0] if on else "train"


OVERFIT_SIGNED = {"clips": 8, "steps": 2000, "ratio": 0.25, "f1_frac": 0.8, "none_margin": 0.02}


def overfit_signed_ok(clips, steps, ratio, f1_frac, none_margin=None):
    """Astra t1 item 10: an overfit verdict may only be registered under the signed gate (8 clips, 2,000 updates, loss ratio
    0.25, car F1 fraction 0.8, none margin 0.02) -> (ok, why)"""
    got = {"clips": clips, "steps": steps, "ratio": ratio, "f1_frac": f1_frac, "none_margin": NONE_MARGIN if none_margin is None else none_margin}
    bad = {k: (got[k], v) for k, v in OVERFIT_SIGNED.items() if got[k] != v}
    return not bad, (f"overfit gate differs from the signed one: {bad}" if bad else "ok")


def init_inert(sens, tol_rel=1e-3):
    """Astra t1 item 1: at the UNTOUCHED init (optimizer step 0) the guide path is exactly inert (zero conv) and the token path
    inert within the check_parity bound (the gated term is exactly 0; GEMM tiling may differ in the last bits)"""
    tr = (sens or {}).get("trained") or {}
    if not tr.get("finite"):
        return False
    return tr.get("guide") == 0.0 and abs(tr.get("tokens", 0.0)) <= tol_rel * max(tr.get("v_abs", 0.0), 1e-12)


def sweep_complete(sweep, gcs=SWEEP_GC, micros=SWEEP_MICRO):
    """Astra t1 item 4: every declared (grad_ckpt, micro) cell is measured, an OOM (with its message), or explicitly skipped
    after an OOM at a SMALLER micro of the same grad_ckpt -> (ok, missing)"""
    miss = []
    extra = sorted(set(sweep or {}) - {f"gc{g}_m{m}" for g in gcs for m in micros})
    if extra:  # Astra t2 item 4: exactly the declared grid, no extra cells
        miss.append(f"cells outside the declared grid: {extra}")
    for gc in gcs:
        for mb in micros:
            c = (sweep or {}).get(f"gc{gc}_m{mb}")
            if c is None:
                miss.append(f"gc{gc}_m{mb}: absent"); continue
            if "gpu_s" in c and _num(c["gpu_s"]):
                continue
            if c.get("oom"):
                continue
            sk = c.get("skipped_after_oom")
            if sk and (sweep.get(sk) or {}).get("oom") and int(sk.split("_m")[1]) < mb and sk.startswith(f"gc{gc}_"):
                continue
            miss.append(f"gc{gc}_m{mb}: no measurement / OOM evidence")
    return not miss, miss


def check_sweep_declared(gcs, micros):
    """Astra t2 item 4: the timing sweep must be EXACTLY grad_ckpt {1, 0} x micro {1, 2, 4, 8}, validated before it runs"""
    if list(gcs) != list(SWEEP_GC) or list(micros) != list(SWEEP_MICRO):
        raise SystemExit(f"timing sweep must be exactly grad_ckpt {list(SWEEP_GC)} x micro {list(SWEEP_MICRO)}: got {list(gcs)} x {list(micros)}")


def blocks_ok(pl, block=RETIME_BLOCK):
    """Astra t1 item 4: blocks of exactly `block` updates, consecutive from the first production update, covering every full
    block of the loop -> (ok, why)"""
    bl = pl.get("blocks") or []; n = pl.get("updates") or 0
    why = []
    if pl.get("block") != block:
        why.append(f"block width {pl.get('block')} != {block}")
    if len(bl) != n // block:
        why.append(f"{len(bl)} blocks for {n} updates (need {n // block})")
    for i, b in enumerate(bl):
        if b.get("updates") != block or b.get("first") != i * block:
            why.append(f"block {i} covers updates {b.get('first')}..+{b.get('updates')} (need {i * block}..+{block})"); break
    return not why, "; ".join(why) or "ok"


def startup_for_forecast(J, prod_train_drives=None):
    """Astra t1 item 7: start-up priced per job start = the SELECTED recipe's measured production start components (allocation
    -> trainer, minus timing-only steps; process -> ready incl. the cold stage-in; frozen snapshot; first update incl. Adam;
    guard verification; first probe; resume checkpoint load). -> (seconds | None, info)"""
    st = J.get("startup") or {}
    comp = st.get("components") or {}
    info = {"components": comp, "from_allocation": comp.get("pre_trainer_s") is not None, "timing_drives": J.get("train_drives"),
            "extrapolated": False, "complete": all(_num(comp.get(k)) for k in STARTUP_PARTS)}
    if not info["complete"]:
        return None, info
    return float(sum(comp[k] for k in STARTUP_PARTS)), info


def evidence_sha(root="."):
    """Astra t1 item 11: identity of every file that produces re-time evidence (trainer, model, rules, staging, forecast, panel /
    rehearsal, rollout, geometry check, slurm wrapper)"""
    import hashlib
    h = hashlib.sha1()
    for f in EVIDENCE_FILES:
        h.update(f.encode() + b"\0" + open(os.path.join(root, f), "rb").read())
    return h.hexdigest()[:16]


ORIGINAL_TIMING_JOB = "3756964"  # the first S1 timing allocation, charged to the ledger (council/s1-timing-r1.md)


def ledger_evidence(jobs, this_job, original=ORIGINAL_TIMING_JOB):
    """Astra t1 item 5 / t2 item 5: the S1 ledger must hold BOTH timing allocations: the charged ORIGINAL timing (3756964, not
    any earlier id) and this re-time -> dict"""
    tj = sorted({j.split("-")[1] for j, v in jobs.items() if v.get("kind") == "alloc_timing" and j.startswith("alloc-")})
    return {"timing_allocations": tj, "this_job": this_job, "original": original, "original_charged": original in tj,
            "this_job_charged": bool(this_job) and str(this_job) in tj,
            "ok": original in tj and bool(this_job) and str(this_job) in tj and str(this_job) != original}


def sampler_valid(AS, this_job=None):
    """Astra t2 item 8: an allocation sampler record counts only with >= 1 VALID job-level read (memory.peak or memory.current
    of a job_<id> cgroup) of this job"""
    return bool(AS) and int(AS.get("n_valid", 0)) > 0 and "job_" in str(AS.get("dir") or "") and (this_job is None or str(AS.get("job")) == str(this_job))


def memory_final(mem, AS, rehearsal_end_t, this_job=None):
    """Astra t2 item 8: the allocation peak is finalised only AFTER the rehearsal: the trainer's peaks + the job-level allocation
    sampler read after the rehearsal ended, over the Slurm request -> {"rule", "includes_rehearsal", "sampler_valid"}"""
    ok_s = sampler_valid(AS, this_job)
    tr = mem.get("rule") or {}
    peaks = [tr.get("peak")] + ([AS.get("peak"), AS.get("max_current")] if ok_s else [])
    r = memory_rule(peaks, mem.get("request_bytes"))
    # Astra T3 item 8: coverage needs a SUCCESSFUL job-level read after the rehearsal ended (a failed read never extends coverage)
    inc = bool(ok_s and rehearsal_end_t is not None and AS.get("t_valid") is not None and float(AS["t_valid"]) >= float(rehearsal_end_t))
    return {"rule": r, "includes_rehearsal": inc, "sampler_valid": ok_s, "pass": bool(r["pass"] is True and inc)}


def memory_rule(peaks, request, frac=None):
    """Astra t1 item 8: peak incl. staged files from the allocation start (job memory.peak, the job-level sampler started by
    slurm at allocation, the in-trainer sampler: the largest) over the SLURM REQUEST (never an ancestor cgroup limit)"""
    from mrv.wan_stage import PEAK_FRAC
    frac = PEAK_FRAC if frac is None else frac
    pk = [p for p in peaks if _num(p)]
    if not pk or not _num(request) or request <= 0:
        return {"pass": None, "peak": max(pk) if pk else None, "request": request, "ratio": None}
    p = max(pk)
    return {"pass": p <= frac * request, "peak": p, "request": request, "ratio": round(p / request, 4)}


def retime_rules(J, fc, canon, score_parity, rehearsal, token_geom=None):
    """CONVERGED item 7 + Astra t1 items 1-9 / 11: every rule with pass / fail and its numbers; all must hold, else no overfit.
    J = timing.json, fc = forecast_final, canon = data/temporal_lists.json, score_parity = (ok, why) of the 2080 Ti record,
    rehearsal = the panel-loop rehearsal record (or None)."""
    R = {}
    st, pl, mem = J.get("stage") or {}, J.get("production_loop") or {}, J.get("memory") or {}
    ok_l = (J.get("train_ids_sha") == ids_sha(canon.get("train", [])) and J.get("dev_ids_sha") == ids_sha(canon.get("dev", []))
            and J.get("train_drives") == N_TRAIN == len(set(canon.get("train", []))) and J.get("dev_drives") == N_DEV == len(set(canon.get("dev", []))))
    R["exact_production_lists"] = {"pass": bool(ok_l), "train_drives": J.get("train_drives"), "dev_drives": J.get("dev_drives"),
                                   "train_ids_sha": J.get("train_ids_sha"), "canonical_train_sha": ids_sha(canon.get("train", []))}
    su, si = startup_for_forecast(J)
    R["cold_allocation_startup_measured"] = {"pass": bool(st.get("cold") and si["complete"] and si["from_allocation"]
                                                          and (fc.get("startup_info") or {}).get("complete")),
                                             "stage_cold": st.get("cold"), "components": si["components"], "startup_s": su}
    R["production_updates_ge_200"] = {"pass": (pl.get("updates") or 0) >= RETIME_MIN_UPDATES, "updates": pl.get("updates")}
    ok_b, why_b = blocks_ok(pl)
    R["blocks_32_consecutive"] = {"pass": ok_b, "why": why_b, "block": pl.get("block"), "blocks": len(pl.get("blocks") or [])}
    bl = pl.get("blocks") or []
    qf = [b.get("queue_frac") for b in bl]; wg = [b.get("wall_over_gpu") for b in bl]
    R["queue_wait_le_5pct_each_block"] = {"pass": len(bl) >= 2 and all(_num(x) and x <= RETIME_QUEUE_FRAC for x in qf), "per_block": qf}
    R["wall_le_1.10_gpu_each_block"] = {"pass": len(bl) >= 2 and all(_num(x) and x <= RETIME_WALL_OVER_GPU for x in wg), "per_block": wg}
    ok_s, miss_s = sweep_complete(J.get("sweep"))
    R["declared_recipe_and_sweep"] = {"pass": bool(ok_s and J.get("batch") == GLOBAL_BATCH and J.get("use_tokens") is True
                                                   and pl.get("micro") == (J.get("selected_setting") or {}).get("micro")
                                                   and pl.get("grad_ckpt") == (J.get("selected_setting") or {}).get("grad_ckpt")),
                                      "sweep_missing": miss_s, "batch": J.get("batch"), "use_tokens": J.get("use_tokens")}
    mu = J.get("matched_update") or {}
    sel = J.get("selected_setting") or {}
    srow = (mu.get("rows") or {}).get(sel.get("key", ""), {})
    R["matched_update_selected"] = {"pass": bool(sel and mu.get("method") == "per_tensor_adam_delta" and srow.get("ok")
                                                 and (srow.get("delta") or {}).get("ok")),
                                    "selected": sel, "method": mu.get("method"), "selected_delta": srow.get("delta")}
    tg = token_geom if token_geom is not None else J.get("token_geometry")
    ii, es = J.get("init_inertness") or {}, J.get("early_sensitivity") or {}
    R["gates_final_recipe"] = {"pass": bool((tg or {}).get("pass") and (tg or {}).get("manifest_verified") and (J.get("init_parity") or {}).get("pass")
                                            and (tg or {}).get("manifest_sha") and (tg or {}).get("manifest_sha") == J.get("manifest_sha")
                                            and ii.get("pass") and ii.get("opt_step") == 0 and es.get("pass") and es.get("opt_step") == SENS_STEP
                                            and es.get("micro") == sel.get("micro") and es.get("grad_ckpt") == sel.get("grad_ckpt")),
                               "token_geometry": (tg or {}).get("pass"), "geometry_inputs_hashed": (tg or {}).get("manifest_verified"),
                               "geometry_manifest_sha": (tg or {}).get("manifest_sha"), "trainer_manifest_sha": J.get("manifest_sha"),
                               "init_parity": (J.get("init_parity") or {}).get("pass"), "init_inert": ii.get("pass"), "init_opt_step": ii.get("opt_step"),
                               "sensitivity": es.get("pass"), "sensitivity_opt_step": es.get("opt_step"), "required_step": SENS_STEP}
    R["staged_hashes_eq_manifest"] = {"pass": bool(st.get("hash_equals_manifest") and not st.get("mismatches")),
                                      "files": st.get("total_files"), "bytes": st.get("total_bytes"), "manifest_sha": J.get("manifest_sha")}
    sv = J.get("staged_vs_lustre") or {}
    R["staged_batches_byte_equal_lustre"] = {"pass": bool(sv.get("all_equal") and (sv.get("batches") or 0) >= 8), "batches": sv.get("batches")}
    mf = mem.get("final") if isinstance(mem.get("final"), dict) else {}
    R["peak_memory_le_85pct_of_request"] = {**(mf.get("rule") or {}), "pass": mf.get("pass") is True and mf.get("includes_rehearsal") is True
                                            and mf.get("sampler_valid") is True, "includes_rehearsal": mf.get("includes_rehearsal"),
                                            "sampler_valid": mf.get("sampler_valid")}
    sp_ok, sp_why = score_parity if score_parity else (False, "not checked")
    R["scorer_parity_off_slice"] = {"pass": bool(sp_ok), "why": sp_why}
    R["six_valid_donors"] = {"pass": bool(fc.get("donor_plan_ok")), "receivers": len(fc.get("donor_plan") or {})}
    rh = rehearsal or {}
    R["panel_rehearsal_byte_equal"] = {"pass": bool(rh.get("pass") and rh.get("fresh_all_ticks") and rh.get("ckpt_sha")
                                                    and rh.get("ckpt_sha") == J.get("ckpt_sha16")
                                                    and rh.get("inputs_manifest_sha") and rh.get("inputs_manifest_sha") == J.get("manifest_sha")),
                                       "inputs_manifest_sha": rh.get("inputs_manifest_sha"),
                                       "why": rh.get("why", "no rehearsal record"), "ckpt_sha": rh.get("ckpt_sha"), "timing_ckpt_sha": J.get("ckpt_sha16"),
                                       "fresh_all_ticks": rh.get("fresh_all_ticks")}
    inp = fc.get("inputs") or {}
    le = fc.get("ledger_evidence") or {}
    ok_f = (fc.get("floor_12k_reachable") and (fc.get("updates") or 0) >= UPDATE_TARGET and fc.get("n_panels") == 1 and inp.get("n_panels") == 1
            and inp.get("budget_h") == BUDGET_H and inp.get("job_h") == JOB_H and fc.get("panel_plan_complete") is True
            and _num(fc.get("panel_h_total")) and fc["panel_h_total"] > 0 and inp.get("panel_h") == fc.get("panel_h_total") and le.get("ok"))
    R["forecast_ge_13200_one_panel"] = {"pass": bool(ok_f), "updates": fc.get("updates"), "target": UPDATE_TARGET, "n_panels": fc.get("n_panels"),
                                        "budget_h": inp.get("budget_h"), "job_h": inp.get("job_h"), "panel_h_total": fc.get("panel_h_total"),
                                        "ledger_evidence": le}
    R["evidence_code_identity"] = {"pass": bool(J.get("evidence_sha") and J.get("evidence_sha") == fc.get("evidence_sha")),
                                   "timing": J.get("evidence_sha"), "forecast": fc.get("evidence_sha")}
    failing = [k for k, v in R.items() if not v["pass"]]
    return {"rules": R, "all_pass": not failing, "failing": failing}


def plan_fresh(pt_runs, expected):
    """Astra t1 item 6: every requested run rendered every requested tick in THIS process (wan_rollout.py skips files that
    already exist: a reused RGB would show up as fewer ticks rendered). pt_runs = plan_timing.json "runs", expected = ticks
    each run must render (same order) -> (ok, why)"""
    got = [int(r.get("ticks_rendered", -1)) for r in pt_runs]
    if len(got) != len(expected):
        return False, f"{len(got)} runs rendered, {len(expected)} requested"
    bad = [(i, g, e) for i, (g, e) in enumerate(zip(got, expected)) if g != e]
    return not bad, (f"runs rendered fewer / other ticks than requested (index, rendered, requested): {bad[:4]}" if bad else "ok")


def dir_empty(d):
    return not os.path.exists(d) or not os.listdir(d)


def refresh_forecast(fc, used_h, done_kinds, start_step=0):
    """Astra t1 item 11: the stored forecast re-run against the ACTUAL ledger now (every job since the timing charged) and the gate
    registry -> (ok, refreshed). Before production: >= UPDATE_TARGET; a production resume: start_step + affordable >= floor."""
    inp = dict(fc.get("inputs") or {})
    if not inp:
        return False, {"why": "forecast has no inputs (legacy)"}
    inp.update(used_h=used_h, done_kinds=set(done_kinds))
    r = forecast_s1(inp)
    need = UPDATE_TARGET if start_step == 0 else FLOOR_UPDATES
    r["start_step"] = start_step; r["need"] = need; r["used_h"] = round(used_h, 3)
    return (start_step + r["updates"]) >= need, r


def retime_gate(tj, micro, grad_ckpt, evidence=None, used_h=None, done_kinds=(), start_step=0, ledger_path=None, ledger_jobs=None):
    """overfit / resume test / train / extend precondition -> (ok, why): the newest re-time passed every rule, selected exactly
    this (micro, grad_ckpt), was produced by the CURRENT evidence code, and its forecast still holds against the actual ledger"""
    sel = (tj or {}).get("selected_setting")
    rp = (tj or {}).get("retime_pass") or {}
    if not sel:
        return False, "timing.json has no selected_setting (legacy timing or no matched setting): re-run MODE=timing"
    if (int(micro), int(grad_ckpt)) != (int(sel["micro"]), int(sel["grad_ckpt"])):
        return False, f"--micro {micro} --grad_ckpt {grad_ckpt} != the re-time selection micro {sel['micro']} grad_ckpt {sel['grad_ckpt']}"
    if not rp.get("all_pass"):
        return False, f"re-time rule failed: {rp.get('failing') or 'no retime_pass block (run scripts/wan_forecast.py)'}"
    if evidence is not None and (tj.get("evidence_sha") != evidence or (tj.get("forecast_final") or {}).get("evidence_sha") != evidence):
        return False, f"evidence code changed since the re-time ({tj.get('evidence_sha')} -> {evidence}): re-run MODE=timing"
    if ledger_path is not None:  # Astra t2 item 11: the refresh reads THE timing's ledger, which must exist and hold both charges
        fl = (tj.get("forecast_final") or {}).get("ledger_path")
        if not fl or os.path.normpath(ledger_path) != os.path.normpath(fl):
            return False, f"--ledger {ledger_path} is not the ledger the re-time was forecast on ({fl})"
        if not os.path.exists(ledger_path):
            return False, f"ledger {ledger_path} missing: spend unreadable (never zero)"
        le = ledger_evidence(ledger_jobs if ledger_jobs is not None else Ledger.read(ledger_path), tj.get("job"))
        if not le["ok"]:
            return False, f"ledger does not hold both timing charges at refresh: {le}"
    if used_h is not None:
        ok, r = refresh_forecast(tj.get("forecast_final") or {}, used_h, done_kinds, start_step)
        if not ok:
            return False, (f"refreshed forecast against the actual ledger ({used_h:.2f} h used): {start_step} + {r.get('updates')} updates < "
                           f"{r.get('need')}: report to the owner")
    return True, "ok"


# ------------------------------------------------------------------------------------------------------------- budget
# rollout option mirror (Astra S1d item 3): the EXACT option set of scripts/wan_rollout.py (tested against its source), so
# argparse resolves abbreviations ("--window=1") the same way the rollout will; the panel parses its --rollout_args ONCE here
ROLLOUT_FLAGS = {"unseal", "swap", "mp4", "mp4_keys", "allow_inert"}
ROLLOUT_TYPES = {"steps": int, "shift": float, "cfg": float, "window_latents": int, "max_ticks": int, "swap_at": int, "sd_lat": int}
ROLLOUT_OPTIONS = ["ckpt", "split", "unseal", "lists", "enc", "guides", "tokens", "keys", "seeds", "drive_list", "steps", "sampler",
                   "shift", "cfg", "cond", "window_latents", "max_ticks", "swap", "swap_at", "donors", "sd_lat", "mp4", "mp4_keys",
                   "dtype", "ledger", "allow_inert", "shard", "manifest", "manifest_sha", "plan", "out"]
ROLLOUT_EXTRA_ALLOWED = {"steps", "sampler", "shift", "cfg", "window_latents", "sd_lat", "mp4", "mp4_keys", "dtype"}


def parse_rollout_extra(extra):
    """--rollout_args of the panel -> (window_latents, canonical arg list). The window size is decided ONCE and the canonical
    list carries it explicitly, so the preflight and every wan_rollout.py call use the same value. Any option outside
    ROLLOUT_EXTRA_ALLOWED (length, split, mode, seeds, drives, swap, donors, output ...), even abbreviated, is refused."""
    import argparse
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=True, exit_on_error=False)
    for o_ in ROLLOUT_OPTIONS:
        if o_ in ROLLOUT_FLAGS:
            ap.add_argument(f"--{o_}", action="store_true", default=None)
        else:
            ap.add_argument(f"--{o_}", type=ROLLOUT_TYPES.get(o_, str), default=None)
    try:
        ns, rest = ap.parse_known_args(list(extra))
    except (argparse.ArgumentError, SystemExit) as e:
        raise SystemExit(f"--rollout_args not understood: {e}")
    if rest:
        raise SystemExit(f"--rollout_args has unknown / ambiguous arguments: {rest}")
    given = {k: v for k, v in vars(ns).items() if v is not None}
    bad = sorted(set(given) - ROLLOUT_EXTRA_ALLOWED)
    if bad:
        raise SystemExit(f"--rollout_args may not set {bad} (full-length 6 x 2 panels; the panel controls these)")
    win = int(given.get("window_latents", 5))
    canon = []
    for k, v in given.items():
        if k == "window_latents":
            continue
        canon += [f"--{k}"] if k in ROLLOUT_FLAGS else [f"--{k}", str(v)]
    return win, canon + ["--window_latents", str(win)]


def train_reserve_h(fc, reserve_panels, window_latents=5, swap_at=50):
    """Astra S1d / S1e item 4: the measured TOTAL panel cost of a COMPLETE donor plan (six valid assignments, shared-mode
    references priced) computed for the SAME panel settings (rollout window size, swap position) the panel will use, times
    reserve_panels; legacy / incomplete forecasts and forecasts for other settings are refused."""
    if not isinstance(fc, dict):
        raise SystemExit("no forecast_final in timing.json: run MODE=timing with the current code")
    ps = fc.get("panel_settings") or {}
    if ps.get("window_latents") != window_latents or ps.get("swap_at") != swap_at:
        raise SystemExit(f"forecast priced the panel for {ps} but the panel uses window_latents={window_latents}, swap_at={swap_at}: "
                         f"rerun scripts/wan_forecast.py with those settings")
    if fc.get("rollout_window_latents") != window_latents:  # Astra S1g 4b: the RATES must have been measured at that width too
        raise SystemExit(f"forecast rates measured at rollout window {fc.get('rollout_window_latents')}, panel uses {window_latents}: "
                         f"rerun MODE=timing with PANEL_WIN={window_latents}")
    plan = fc.get("donor_plan") or {}
    if not fc.get("donor_plan_ok") or len(plan) != 6 or any(not v for v in plan.values()):
        raise SystemExit(f"forecast donor plan incomplete ({len(plan)} receivers, ok={fc.get('donor_plan_ok')}): no valid reserve")
    tot = fc.get("panel_h_total")
    if not _num(tot) or tot <= 0:
        raise SystemExit("forecast has no measured total panel cost (panel_h_total): legacy timing.json, rerun MODE=timing")
    return float(tot) * reserve_panels


def production_reserve_h(fc, reserve_panels, explicit_h=None, window_latents=5, swap_at=50):
    """the reserve a production (train / extend) run keeps (Astra S1e item 4a): at least one measured panel, never below the
    measured total; an explicit --reserve_hours may only RAISE it"""
    if reserve_panels < 1:
        raise SystemExit(f"--reserve_panels {reserve_panels} < 1: production keeps at least one measured panel in reserve")
    measured = train_reserve_h(fc, reserve_panels, window_latents, swap_at)
    if explicit_h is not None and explicit_h < measured:
        raise SystemExit(f"--reserve_hours {explicit_h} is below the measured panel reserve {measured:.2f} h: refused")
    return max(measured, explicit_h or 0.0)


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
        # an allocation-wide outer heartbeat: alloc_* (slice) and score_* (slurm/wan_score.sh, 2080 Ti) kinds (Neon 11:30 P0)
        s.job = f"{'alloc' if kind.startswith(('alloc', 'score_')) else 'proc'}-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}-{int(s.t0)}"
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
