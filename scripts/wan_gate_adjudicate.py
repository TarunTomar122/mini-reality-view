"""Exit-6 adjudicator for the S1 overfit gate: the signed rule of council/review-20261128-1330-swap2-astra.md (context
council/review-20261128-1330-swap-fable.md). CPU only, stdlib only; reads what overfit job 3757187 wrote and edits no file
except one appended line in the gate registry.

Only job 3757187's FIRST NOT_REACHED (trainer exit 6) read at update 2,000 qualifies. ALL must hold, else it refuses (exit 1,
nothing appended = hold, diagnose with council):
  identity    : --job is 3757187; overfit.json (as scripts/train_wan_teacher.py writes it) has status NOT_REACHED, PASS false,
                updates == declared_updates == 2000, gate_key == --gate_key, prior_not_reached == 0, eight distinct signed clips,
                the signed declared_ratio 0.25 / declared_f1_frac 0.8 / none_margin 0.02; gates.jsonl holds exactly one matching
                overfit NOT_REACHED record for that job, output directory and gate key (same ratio and F1s) and it is the FIRST
                NOT_REACHED of that key; no overfit PASS (adjudicated or not) exists yet for the key (one adjudicated PASS max);
                overfit_samples.png and log.jsonl exist.
  wiring      : key cars present (round-trip TP + FN > 0); round-trip car F1 > 0; hybrid car F1 >= 0.80 x round-trip F1;
                hybrid car F1 > no-input car F1 + 0.02 (each F1 re-derived from the recorded counts and equal to the record).
  fitting     : fixed-noise eval loss at 2,000 <= 0.50 x loss at 0 (0.50 = an UNCALIBRATED cutoff; ratio == final / update0);
                strict decreases 1,700 -> 1,800 -> 1,900 -> 2,000 in eval_history, each eval present in log.jsonl.
  verified    : this file's sha256 is pinned in council/FREEZE-s1.md (Neon verifies it first; --freeze).
Any missing, non-finite or inconsistent value -> refuse. On success it appends ONE same-key overfit PASS record marked
{"adjudicated": true, "job", "adjudicator_sha256", "original_verdict": "NOT_REACHED", the measured ratio and the three F1s},
never altering existing lines; the frozen trainer's MODE=train check (mrv.wan_plan.gates_ok: kind overfit, status PASS, same
gate_key) accepts it. Report it to the owner as "the wiring check passed; fitting was slower than our guess", never as a plain
PASS.
  python scripts/wan_gate_adjudicate.py --run_dir runs/wan_ctest_overfit_3757187 --gates runs/wan_gates/gates.jsonl \
         --job 3757187 --gate_key <gate key from overfit.json>
"""
import argparse, hashlib, json, math, os, sys, time

ALLOWED_JOB = "3757187"
UPDATES = 2000
CLIPS = 8
SIGNED = {"declared_ratio": 0.25, "declared_f1_frac": 0.8, "none_margin": 0.02}
RATIO_CUTOFF = 0.50        # signed, uncalibrated
F1_FRAC = 0.80
NONE_MARGIN = 0.02
TAIL = (1700, 1800, 1900, 2000)
RULE = "council/review-20261128-1330-swap2-astra.md"
THIS = os.path.abspath(__file__)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def records(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def f1_of(c):
    tp, fp, fn = c
    return 2 * tp / max(1, 2 * tp + fp + fn)


def adjudicate(run_dir, gates, job, gate_key, freeze, self_sha=None):
    """-> (ok, why list, record to append | None). Pure check: writes nothing."""
    why = []
    sha = self_sha or sha256_file(THIS)
    if str(job) != ALLOWED_JOB:
        return False, [f"job {job} does not qualify (only {ALLOWED_JOB})"], None
    if not freeze or not os.path.exists(freeze) or f"{sha}  scripts/wan_gate_adjudicate.py" not in open(freeze).read():
        why.append(f"adjudicator sha256 {sha[:12]} is not pinned in {freeze} (Neon verifies it first)")
    oj = os.path.join(run_dir, "overfit.json")
    lj = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(oj) or not os.path.exists(lj) or not os.path.exists(gates):
        return False, why + [f"missing evidence: overfit.json {os.path.exists(oj)}, log.jsonl {os.path.exists(lj)}, gates {os.path.exists(gates)}"], None
    smp = os.path.join(run_dir, "overfit_samples.png")
    if not os.path.exists(smp) or os.path.getsize(smp) == 0:
        why.append("overfit_samples.png missing")
    try:
        O = json.load(open(oj))
    except Exception as e:  # noqa: BLE001
        return False, why + [f"overfit.json unreadable ({type(e).__name__})"], None
    # ---- identity
    if not str(O.get("status", "")).startswith("NOT_REACHED") or O.get("PASS") is not False:
        why.append(f"overfit.json verdict {O.get('status')!r} / PASS {O.get('PASS')!r} is not the NOT_REACHED exit")
    if O.get("updates") != UPDATES or O.get("declared_updates") != UPDATES:
        why.append(f"read at update {O.get('updates')} (declared {O.get('declared_updates')}), not {UPDATES}")
    if O.get("gate_key") != gate_key:
        why.append(f"overfit.json gate_key {O.get('gate_key')} != {gate_key}")
    if O.get("prior_not_reached") != 0:
        why.append(f"prior_not_reached {O.get('prior_not_reached')} != 0 (not the first NOT_REACHED)")
    clips = O.get("clips")
    if not (isinstance(clips, list) and len(clips) == CLIPS and len({json.dumps(c) for c in clips}) == CLIPS):
        why.append(f"not {CLIPS} distinct signed clips")
    for k, v in SIGNED.items():
        if O.get(k) != v:
            why.append(f"{k} {O.get(k)} != signed {v}")
    # ---- wiring (F1s re-derived from the counts)
    cnt = O.get("counts") or {}
    ok_c = all(isinstance(cnt.get(n), list) and len(cnt[n]) == 3 and all(isinstance(x, int) and x >= 0 for x in cnt[n]) for n in ("sample", "roundtrip", "none"))
    hy, rt, no = O.get("car_f1_sample"), O.get("car_f1_roundtrip"), (O.get("car_f1_controls") or {}).get("none")
    if not ok_c or not all(fin(x) for x in (hy, rt, no)):
        why.append("car counts / F1s missing or non-finite")
    else:
        if (O.get("car_f1_controls") or {}).get("hybrid") != hy:
            why.append("car_f1_controls.hybrid != car_f1_sample")
        for nm, val in (("sample", hy), ("roundtrip", rt), ("none", no)):
            if abs(f1_of(cnt[nm]) - val) > 1e-12:
                why.append(f"{nm} car F1 {val} != F1 of its counts {f1_of(cnt[nm])}")
        if cnt["roundtrip"][0] + cnt["roundtrip"][2] <= 0:
            why.append("no key cars")
        if not rt > 0:
            why.append(f"round-trip car F1 {rt} not > 0")
        if not hy >= F1_FRAC * rt:
            why.append(f"hybrid car F1 {hy} < {F1_FRAC} x round-trip {rt}")
        if not hy > no + NONE_MARGIN:
            why.append(f"hybrid car F1 {hy} not > no-input {no} + {NONE_MARGIN}")
    # ---- fitting
    l0, lf, ratio = O.get("eval_loss_update0"), O.get("eval_loss_final"), O.get("ratio")
    hist = O.get("eval_history")
    H = {}
    if isinstance(hist, list):
        for e in hist:
            if isinstance(e, list) and len(e) == 2 and isinstance(e[0], int) and fin(e[1]):
                if e[0] in H and H[e[0]] != e[1]:
                    why.append(f"eval_history has two values for update {e[0]}")
                H[e[0]] = e[1]
            else:
                why.append(f"eval_history entry {e!r} malformed or non-finite")
    if not (fin(l0) and fin(lf) and fin(ratio) and l0 > 0 and lf >= 0):
        why.append("eval losses / ratio missing or non-finite")
    else:
        if abs(ratio - lf / l0) > 1e-12 * max(1.0, abs(ratio)):
            why.append(f"ratio {ratio} != final / update-0 {lf / l0}")
        if not ratio <= RATIO_CUTOFF:
            why.append(f"loss ratio {ratio:.4f} > {RATIO_CUTOFF} (uncalibrated cutoff)")
        if H.get(0) != l0:
            why.append(f"eval_history update 0 {H.get(0)} != eval_loss_update0 {l0}")
        if UPDATES in H and abs(H[UPDATES] - lf) > 1e-4 * max(abs(lf), 1e-12):
            why.append(f"eval_history update {UPDATES} {H.get(UPDATES)} inconsistent with eval_loss_final {lf}")
    miss = [s for s in TAIL if s not in H]
    if miss:
        why.append(f"eval_history lacks updates {miss}")
    else:
        for s0, s1 in zip(TAIL, TAIL[1:]):
            if not H[s1] < H[s0]:
                why.append(f"eval loss not strictly decreasing {s0} -> {s1} ({H[s0]} -> {H[s1]})")
    logm = {}
    for r in records(lj):
        if "overfit_eval" in r and isinstance(r.get("update"), int) and fin(r.get("mean")):
            logm.setdefault(r["update"], []).append(r["mean"])
    for s in (0,) + TAIL:
        if s in H and not any(abs(round(H[s], 5) - m) <= 1.5e-5 for m in logm.get(s, [])):
            why.append(f"log.jsonl has no eval at update {s} matching eval_history ({H[s]})")
    # ---- registry
    G = records(gates)
    mine = [r for r in G if r.get("kind") == "overfit" and str(r.get("job")) == ALLOWED_JOB and r.get("gate_key") == gate_key]
    nr_key = [r for r in G if r.get("kind") == "overfit" and r.get("gate_key") == gate_key and str(r.get("status", "")).startswith("NOT_REACHED")]
    if len(mine) != 1 or not str(mine[0].get("status", "")).startswith("NOT_REACHED"):
        why.append(f"gates.jsonl: {len(mine)} overfit records for job {ALLOWED_JOB} / key (need exactly one NOT_REACHED)")
    else:
        m = mine[0]
        if os.path.normpath(str(m.get("out"))) != os.path.normpath(run_dir):
            why.append(f"gates.jsonl record out {m.get('out')} != run dir {run_dir}")
        if m.get("status") != O.get("status") or m.get("ratio") != ratio or m.get("f1") != O.get("car_f1_controls"):
            why.append("gates.jsonl record (status / ratio / F1s) differs from overfit.json")
        if not nr_key or nr_key[0] is not m:
            why.append("job 3757187's NOT_REACHED is not the first NOT_REACHED of this gate key")
    if any(r.get("kind") == "overfit" and r.get("gate_key") == gate_key and r.get("status") == "PASS" for r in G):
        why.append("an overfit PASS already exists for this gate key (one adjudicated PASS max)")
    if why:
        return False, why, None
    rec = {"kind": "overfit", "gate_key": gate_key, "status": "PASS", "out": mine[0].get("out"), "f1": O["car_f1_controls"], "ratio": ratio,
           "job": ALLOWED_JOB, "adjudicated": True, "adjudicator_sha256": sha, "original_verdict": "NOT_REACHED",
           "original_status": O["status"], "measured_ratio": ratio, "ratio_cutoff": RATIO_CUTOFF, "ratio_cutoff_calibrated": False,
           "car_f1_hybrid": hy, "car_f1_roundtrip": rt, "car_f1_none": no, "rule": RULE,
           "owner_wording": "the wiring check passed; fitting was slower than our guess"}
    return True, [], rec


def append_once(gates, rec):
    """append ONE line; existing bytes must be unchanged afterwards (checked)"""
    before = open(gates, "rb").read()
    with open(gates, "a") as f:
        f.write("\n" + json.dumps({**rec, "time": time.time()}) + "\n")
    after = open(gates, "rb").read()
    if not after.startswith(before):
        raise SystemExit("gates.jsonl changed underneath the append: inspect it")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True); p.add_argument("--gates", default="runs/wan_gates/gates.jsonl")
    p.add_argument("--job", required=True); p.add_argument("--gate_key", required=True)
    p.add_argument("--freeze", default="council/FREEZE-s1.md", help="must pin this file's sha256 (Neon-verified)")
    a = p.parse_args(argv)
    ok, why, rec = adjudicate(a.run_dir, a.gates, a.job, a.gate_key, a.freeze)
    if not ok:
        print("ADJUDICATE REFUSED (hold; nothing appended):", json.dumps(why))
        return 1
    append_once(a.gates, rec)
    print("ADJUDICATE PASS appended", json.dumps(rec))
    print(f"owner: the wiring check passed; fitting was slower than our guess (loss ratio {rec['measured_ratio']:.4f} vs an uncalibrated "
          f"0.50 cutoff; car F1 hybrid {rec['car_f1_hybrid']:.4f}, round trip {rec['car_f1_roundtrip']:.4f}, no input {rec['car_f1_none']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
