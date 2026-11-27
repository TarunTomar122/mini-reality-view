"""Temporal comparisons (council/PLAN-temporal.md + amendments; split out of temporal_metrics.py so the SIGNED scorer
file, content hash 764dd3ec95db, stays unchanged while comparison rules are added).

  python scripts/temporal_compare.py --make_count_list 'runs/troll/indep_dec_*.json' --k 3 --out council/count_indep_dec.json
  python scripts/temporal_compare.py --compare 'CAND*.json' 'COMP*.json' --count_list council/count_indep_dec.json --floor 20

Same paired drive bootstrap as temporal_metrics.py --compare (signed), plus amendment 3: ID / W are computed only on the
comparator's counting drives (comparator alone matches >= k eligible transitions, seeds pooled; list fixed from the
comparator's file before the candidate's is opened); a counting drive the candidate lacks a seed on, or with no
same-seed shared transition, or fewer counting drives than --floor -> that co-primary cannot pass. A missing late gap on
a contiguous run is a hard failure (a rollout scored without --cold is its own cold reference: gap 0).
"""
import argparse, glob, json, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--compare", nargs=2, default=None)
p.add_argument("--make_count_list", default="")
p.add_argument("--k", type=int, default=3)
p.add_argument("--count_list", default="")
p.add_argument("--floor", type=int, default=None, help="REQUIRED with --count_list: 20 decision, 12 final, 0 dev (amendment 3)")
p.add_argument("--out", default="")
p.add_argument("--cold_ref", default="none", choices=["A", "B", "none"],
               help="which side of --compare IS the cold reference (its late gaps are 0 by definition); every other missing late gap on a contiguous run is a hard failure")
a = p.parse_args()

def load_merged(pat):  # a json, or a glob of shard jsons
    out = {"drives": {}, "roll": None, "cold": None}
    for f in sorted(glob.glob(pat)):
        j_ = json.load(open(f)); out["drives"].update(j_["drives"]); out["roll"] = j_.get("roll"); out["cold"] = j_.get("cold")
    return out


if a.make_count_list:
    C_ = load_merged(a.make_count_list); shas = {q.get("scorer_sha") for q in C_["drives"].values()}
    assert len(shas) == 1, shas
    scs = sorted({r.rsplit("_s", 1)[0] for r in C_["drives"]})
    res = {"comparator": a.make_count_list, "comparator_roll": C_["roll"], "scorer_sha": shas.pop(), "k": a.k, "seeds_pooled": ["0", "1"],
           "drives": scs}
    fin = lambda x: isinstance(x, (int, float)) and np.isfinite(x)
    for name in ("ID", "W"):
        cnt, elig, per_seed = {}, {}, {}
        for s_ in scs:
            rs = [f"{s_}_s{q}" for q in ("0", "1")]
            assert all(r in C_["drives"] for r in rs), f"comparator lacks a seed for {s_}"
            per_seed[s_] = {r[-1]: {"eligible": len(C_["drives"][r]["trans_" + name]),
                                    "matched": sum(1 for t in C_["drives"][r]["trans_" + name] if t[5] and fin(t[4]))} for r in rs}
            cnt[s_] = sum(v["matched"] for v in per_seed[s_].values()); elig[s_] = sum(v["eligible"] for v in per_seed[s_].values())
        res[name] = {"counting": [s_ for s_ in scs if cnt[s_] >= a.k], "comparator_matched": cnt, "eligible": elig, "per_seed": per_seed}
    json.dump(res, open(a.out, "w"), indent=1)
    print("COUNTLIST", {n: len(res[n]["counting"]) for n in ("ID", "W")}, "of", len(scs), "drives, k =", a.k); raise SystemExit

if a.compare:
    def _unused_load_merged(pat):  # a json, or a glob of shard jsons
        out = {"drives": {}, "roll": None, "cold": None}
        for f in sorted(glob.glob(pat)):
            j_ = json.load(open(f)); out["drives"].update(j_["drives"]); out["roll"] = j_.get("roll"); out["cold"] = j_.get("cold")
        return out
    A, B = (load_merged(f) for f in a.compare)
    for nm_, D_ in (("A", A), ("B", B)):
        vs = {(q.get("scorer_sha"), q.get("link")) for q in D_["drives"].values()}
        if len(vs) != 1:
            raise SystemExit(f"{nm_}: mixed scorer versions / link modes {vs}")
    va_, vb_ = ({(q.get("scorer_sha"), q.get("link")) for q in D_["drives"].values()}.pop() for D_ in (A, B))
    if va_ != vb_ or va_[0] is None:
        raise SystemExit(f"scorer version / link mode mismatch or missing: A {va_} B {vb_}")
    runs = sorted(set(A["drives"]) & set(B["drives"]))  # "<scene>_s<seed>"
    seeds_of = lambda D, s_: {r.rsplit("_s", 1)[1] for r in D["drives"] if r.rsplit("_s", 1)[0] == s_}
    all_sc = sorted({r.rsplit("_s", 1)[0] for r in runs})
    # paired seeds: a drive enters only if both methods have the same full seed set for it
    SEEDS = {"0", "1"}  # plan: seeds 0 / 1 on every drive
    all_sc = sorted({r.rsplit("_s", 1)[0] for r in set(A["drives"]) | set(B["drives"])})
    scenes = [s_ for s_ in all_sc if seeds_of(A, s_) == SEEDS and seeds_of(B, s_) == SEEDS]
    unpaired = [s_ for s_ in all_sc if s_ not in scenes]
    if unpaired:
        raise SystemExit(f"incomplete panel: drives without seeds 0 and 1 in both methods: {unpaired} (no drive may silently drop out of any gate)")
    by_scene = {s_: [r for r in runs if r.rsplit("_s", 1)[0] == s_] for s_ in scenes}
    PER = ("recall", "flips", "car_f1", "road_iou", "late_lpips_gap", "late_f1_gap", "early_lpips_gap", "ghosts", "ghost_growth", "ghost_ge5")

    def per_drive(s_):
        """one value per drive (seeds pooled inside the drive): ID / W on SHARED transitions, others averaged over seeds"""
        rs = by_scene[s_]; v = {}
        for name in ("ID", "W"):
            ea, eb, elig = [], [], 0
            for d in rs:
                ta = {tuple(t[:4]): t[4] for t in A["drives"][d]["trans_" + name] if t[5] and np.isfinite(t[4])}
                tb = {tuple(t[:4]): t[4] for t in B["drives"][d]["trans_" + name] if t[5] and np.isfinite(t[4])}
                elig += len({tuple(t[:4]) for t in A["drives"][d]["trans_" + name]} | {tuple(t[:4]) for t in B["drives"][d]["trans_" + name]})
                for k in ta.keys() & tb.keys():
                    ea.append(ta[k]); eb.append(tb[k])
            v[name] = (np.mean(ea), np.mean(eb), len(ea), elig) if ea else (np.nan, np.nan, 0, elig)
        for name in PER:
            # a method compared with its own cold reference: the reference's gap is 0 by definition
            gap = name.endswith("_gap")
            def ga(D, O, d):
                v_ = D["drives"][d].get(name)
                if v_ is not None or not gap:
                    return v_
                if not D["drives"][d].get("contiguous_170"):
                    return None  # no late window for this run
                if (a.cold_ref == "A" and D is A) or (a.cold_ref == "B" and D is B):
                    return 0.0  # declared cold reference: gap 0 by definition
                raise SystemExit(f"{d}: late gap missing although the run is contiguous (cold = {D.get('cold')}) - hard fail")
            va = [ga(A, B, d) for d in rs if ga(A, B, d) is not None]
            vb = [ga(B, A, d) for d in rs if ga(B, A, d) is not None]
            v[name] = (np.mean(va), np.mean(vb), len(va), 0) if va and vb and len(va) == len(vb) else (np.nan, np.nan, 0, 0)
        return v
    PD = {s_: per_drive(s_) for s_ in scenes}
    no_shared = {name: [s_ for s_ in scenes if PD[s_][name][2] == 0 and PD[s_][name][3] > 0] for name in ("ID", "W")}
    CL = json.load(open(a.count_list)) if a.count_list else None
    if CL is not None:
        if a.floor is None:
            raise SystemExit("--floor is required with --count_list (20 decision / 12 final / 0 dev)")
        if CL["k"] != 3:
            raise SystemExit(f"count list made with k = {CL['k']}; amendment 3 (as settled) uses k = 3")
    COUNT = {}
    if CL is not None:
        assert CL["scorer_sha"] == vb_[0], f"count list made with scorer {CL['scorer_sha']}, comparator scored with {vb_[0]}"
        assert CL["comparator_roll"] and os.path.normpath(CL["comparator_roll"]) == os.path.normpath(B["roll"] or ""), \
            f"count list belongs to comparator {CL['comparator_roll']}, not {B['roll']}"
        assert set(CL["drives"]) == set(scenes), "count list drive panel differs from the compared panel"
        for name in ("ID", "W"):
            cnt_ = CL[name]["counting"]
            missing_ = [s_ for s_ in cnt_ if s_ not in scenes]  # candidate (or comparator) lacks a seed on a counting drive
            zero_ = [s_ for s_ in cnt_ if s_ in scenes and PD[s_][name][2] == 0]
            COUNT[name] = {"counting": [s_ for s_ in cnt_ if s_ in scenes], "missing_counting": missing_, "zero_shared_counting": zero_,
                           "floor": a.floor, "k": CL["k"],
                           "can_pass": (not missing_) and (not zero_) and len(cnt_) >= (a.floor or 0)}

    def stats(pick_scenes):  # equal drive weights; ID / W over counting drives only when a count list is given
        res = {}
        for name in ("ID", "W") + PER:
            ps_ = [s_ for s_ in pick_scenes if (name not in COUNT or s_ in set(COUNT[name]["counting"]))]
            xs = [PD[s_][name] for s_ in ps_ if np.isfinite(PD[s_][name][0]) and np.isfinite(PD[s_][name][1])]
            res[name] = (np.mean([x[0] for x in xs]) if xs else np.nan, np.mean([x[1] for x in xs]) if xs else np.nan,
                         sum(x[2] for x in xs), len(xs))
        return res
    def lag_median(D, pos="lag_end"):
        v = []
        for s_ in scenes:
            for r_ in by_scene[s_]:
                q = D["drives"][r_]
                if "lag_recovered" in q:
                    v.append(q[pos] if q["lag_recovered"] else float("inf"))
        return float(np.median(v)) if v else None
    full = stats(scenes)
    rng = np.random.default_rng(0); boots = {k: [] for k in full}
    for _ in range(2000):
        s = stats(list(rng.choice(scenes, len(scenes))))
        for k in s:
            boots[k].append(s[k][0] - s[k][1])
    out = {"A": a.compare[0], "B": a.compare[1], "drives": len(scenes), "runs": sum(len(v) for v in by_scene.values()),
           "unpaired_drives": unpaired, "weighting": "equal per drive; seeds pooled within drive",
           "no_shared_transition_drives": no_shared,
           "shared_per_drive": {name: {s_: PD[s_][name][2] for s_ in scenes} for name in ("ID", "W")}}
    for nm, D in (("A", A), ("B", B)):
        le = [D["drives"][r_].get("link_edge") for s_ in scenes for r_ in by_scene[s_]]
        le = [q for q in le if q]
        if le:
            tt_ = sum(q["total"] for q in le)
            out[f"link_share_{nm}"] = {"iou": round(sum(q["iou"] for q in le) / max(1, tt_), 3), "flow": round(sum(q["flow"] for q in le) / max(1, tt_), 3), "edge_cars": tt_}
        wv = [(D["drives"][r_].get("w_valid_px", 0), D["drives"][r_].get("w_total_px", 0)) for s_ in scenes for r_ in by_scene[s_]]
        if sum(q[1] for q in wv):
            out[f"w_valid_share_{nm}"] = round(sum(q[0] for q in wv) / sum(q[1] for q in wv), 3)
    out["lag_median_ticks"] = {pos: {"A": lag_median(A, pos), "B": lag_median(B, pos)} for pos in ("lag_start", "lag_centre", "lag_end")}
    out["lag_median_ticks"]["gate"] = ("interim rule (drift check 4): median START <= 1 passes under both readings, > 3 fails under both, "
                                       "in between = lag undecided (no adoption on it; owner rules); inf = not recovered")
    out["lag_excluded_runs"] = {nm: sum(1 for s_ in scenes for r_ in by_scene[s_] if D["drives"][r_].get("lag_excluded")) for nm, D in (("A", A), ("B", B))}
    for k, (x, y, n, nd) in full.items():
        bb = np.array(boots[k]); bb = bb[np.isfinite(bb)]
        out[k] = {"A": round(float(x), 4), "B": round(float(y), 4), "diff": round(float(x - y), 4), "n": int(n), "drives": int(nd),
                  "ci95": np.percentile(bb, [2.5, 97.5]).round(4).tolist() if len(bb) else None,
                  "rel": round(float((x - y) / y), 3) if y and np.isfinite(y) and y > 0 else None,
                  "can_pass": (COUNT[k]["can_pass"] if k in COUNT else (not no_shared.get(k))) if k in ("ID", "W") else None}
    out["shared_per_drive_seed"] = {name: {s_: {d[-1]: len({tuple(t[:4]) for t in A["drives"][d]["trans_" + name] if t[5] and np.isfinite(t[4])}
                                                        & {tuple(t[:4]) for t in B["drives"][d]["trans_" + name] if t[5] and np.isfinite(t[4])})
                                                   for d in by_scene[s_]} for s_ in scenes} for name in ("ID", "W")}
    if CL is not None:
        out["comparator_counts_per_seed"] = {name: CL[name]["per_seed"] for name in ("ID", "W")}
    out["cold_ref"] = a.cold_ref
    out["counting"] = COUNT if COUNT else "NO COUNT LIST GIVEN (amendment 3 requires one for decision / final comparisons)"
    print("TCOMPARE", json.dumps(out)); raise SystemExit

