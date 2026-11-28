"""Production gate record for vclean / vmix (council/AGREED-causal-r1.md A4, A5 + A1 + N1; council/codereview-vclean-astra.md).
--arm vclean (A4) or vmix (A5: the probe predicate 2 is REPLACED by the cache check 2'; 1, 3, 4 stay). Binds, with sha256 of
every file read:
  1. the signed vpriv outcome: runs/vpriv_gate/a1_8k.json, a1_12k.json and a1_16k.json (or the A1 stop at 12k), each re-verified
     against its comparator by scripts/a1_decide.py (which only reuses a decision equal to the recomputed one); joint_pass false
     in every one (a vpriv pass -> its exam takes precedence under N1 -> refuse);
  2. a valid probe record (scripts/clean_probe.py) with decision LAUNCH; its two checkpoint digests recomputed now; its code ==
     the suite-tested code;
  2'. (vmix) the cache bind (A@20k digest, cache dir, inputs, the 144 train drives, the suite-tested hashes of builder /
     checker / matched rollout / dependencies / verifier) is recorded in both states; passed = a scripts/cache_check.py record
     accepted by mrv/cache_verify.verify (the SAME verifier the trainer runs; every cache file loaded and checked) -> its digest
     is the approved record; --cache_pending (A5: the cache is built on 2080 Tis while the slice trains) -> the trainer verifies
     a record against the same bind at update 2000, holds without one, and pins the first record it uses;
  2''. (--arm vclean --control) the A5 signed pure key + jitter attribution control: no probe predicate (A5 signs the control);
  3. the clean suite: logs2/cleantests-<id>.out last line "CLEAN_TESTS_SH FAILS=0 dir=runs/ctest_clean_<id>", its code manifest
     = exactly the reviewed file set and == the files on disk now, and its timing record (100 updates, finite step_s / mem_s / GB);
  4. T2: logs2/cleant2-<id>.out ("CLEAN_T2 FAILS=<n> dir=runs/ctest_clean_t2_<id>") + t2.json on matched inputs (ticks, source
     ticks, flags, amp equal): max |diff| < 1e-2 with FAILS=0 -> rho_gate 1.0; a finite max |diff| >= 1e-2 with FAILS=1 (documented
     numeric failure) -> 0.95; anything else (missing comparator, failed job, unmatched fields) -> refuse (rho undecided).
Exit 0 + record (mode "x") only when everything holds. --verify recomputes a record from its own inputs and requires equality.

  python scripts/vclean_gate.py --arm vclean --suite <job id> --t2 <job id> --probe runs/probe/clean_probe.json --out runs/vclean_gate/record.json
  python scripts/vclean_gate.py --arm vmix --suite <job id> --t2 <job id> --cache_record runs/cache/check.json --out runs/vclean_gate/vmix.json
  python scripts/vclean_gate.py --arm vmix --suite <job id> --t2 <job id> --cache_pending --out runs/vclean_gate/vmix.json
  python scripts/vclean_gate.py --arm vclean --control --suite <job id> --t2 <job id> --out runs/vclean_gate/control.json
  python scripts/vclean_gate.py --verify runs/vclean_gate/record.json
"""
import argparse, hashlib, json, math, os, subprocess, sys
from mrv.cache_verify import make_bind, verify, A_PATH

SUITE_FILES = sorted(["mrv/clean_video.py", "mrv/noise.py", "mrv/sd.py", "scripts/train_clean.py", "scripts/clean_rollout.py", "scripts/clean_tests.py",
                      "scripts/clean_probe.py", "scripts/vclean_gate.py", "scripts/det_run.py", "slurm/clean_tests.sh",
                      "scripts/cache_check.py", "scripts/build_cache.py", "scripts/causal_rollout.py", "mrv/causal_video.py", "mrv/cache_verify.py"])
T2_FILES = sorted(["mrv/clean_video.py", "mrv/sd.py", "scripts/clean_rollout.py", "slurm/clean_t2.sh"])
PY = sys.executable


class Refuse(Exception):
    pass


def need(c, m):
    if not c:
        raise Refuse(m)


sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
fin = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def last_line(log, prefix, d):
    need(os.path.isfile(log), f"missing log {log}")
    lines = open(log).read().rstrip("\n").splitlines()
    need(lines, f"{log} is empty")
    ln = lines[-1].strip()
    need(ln.startswith(prefix + " FAILS=") and ln.endswith(f" dir={d}"), f"{log}: last line '{ln[:120]}' is not '{prefix} FAILS=<n> dir={d}'")
    return int(ln.split("FAILS=")[1].split()[0])


def manifest(path, want_files):
    need(os.path.isfile(path), f"missing manifest {path}")
    rows = [l_.split() for l_ in open(path).read().splitlines() if l_.strip()]
    got = {f_: h_ for h_, f_ in rows}
    need(len(got) == len(rows) and sorted(got) == want_files, f"{path}: file set {sorted(got)} != reviewed {want_files}")
    for f_, h_ in got.items():
        need(os.path.isfile(f_) and sha(f_) == h_, f"{path}: {f_} changed since it was tested")
    return got


CACHE_DIR, LISTS, ENC, RAYS_F = "runs/cache/A20k_off_rho1_train", "data/temporal_lists.json", "data/enc_tpv10", "data/calib_rays.pt"


def build(suite, t2, probe, arm="vclean", cache_record="", cache_pending=False, control=False):
    need(arm in ("vclean", "vmix"), f"unknown arm {arm}")
    need(not control or arm == "vclean", "--control is the vclean key + jitter recipe")
    recipe = "vmix" if arm == "vmix" else ("vclean_keyjit_control" if control else "vclean_A4")
    rec = {"inputs": {"suite": suite, "t2": t2, "probe": probe, "arm": arm, "cache_record": cache_record, "cache_pending": cache_pending, "control": control},
           "arm": arm, "recipe": recipe, "a_sha": sha(A_PATH)[:16]}
    # 1. vpriv outcome (A1): 8k and 12k decided; 16k decided unless the A1 stop acted at 12k
    out = {}
    for k in (8, 12, 16):
        f = f"runs/vpriv_gate/a1_{k}k.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        r = subprocess.run([PY, "scripts/a1_decide.py", str(k)], capture_output=True, text=True)
        need(r.returncode == 0 and "existing, verified" in r.stdout, f"{f} does not re-verify against its comparator: {r.stdout[-300:]} {r.stderr[-300:]}")
        need(isinstance(d.get("joint_pass"), bool) and isinstance(d.get("stop"), bool), f"{f}: malformed")
        need(d["joint_pass"] is False, f"vpriv JOINT PASS at {k}k: its exam takes precedence (N1); no vclean launch")
        out[f"{k}k"] = {"file": f, "sha256": sha(f), "joint_pass": False, "safe": d.get("safe"), "stop": d["stop"]}
    need("8k" in out and "12k" in out, "vpriv 8k / 12k A1 decisions missing: outcome not final")
    final = "16k" in out or out["12k"]["stop"] is True
    need(final, "vpriv outcome not final (no 16k decision and no A1 stop at 12k)")
    rec["vpriv"] = {"decisions": out, "outcome": "A1 stop at 12k" if "16k" not in out else "no joint pass through 16k"}
    # 3. suite (first: the probe's code must be the tested code)
    sdir = f"runs/ctest_clean_{suite}"
    fails = last_line(f"logs2/cleantests-{suite}.out", "CLEAN_TESTS_SH", sdir)
    need(fails == 0, f"clean suite {suite}: FAILS={fails}")
    code = manifest(f"{sdir}/code_sha256.txt", SUITE_FILES)
    tf = f"{sdir}/timing.json"
    need(os.path.isfile(tf), f"missing timing record {tf}")
    tm = json.load(open(tf))
    need(tm.get("updates") == 100 and all(fin(tm.get(k_)) for k_ in ("step_s", "mem_s", "peak_gb")) and tm.get("frozen_check") is True, f"{tf}: incomplete {tm}")
    rec["suite"] = {"job": suite, "dir": sdir, "log_sha256": sha(f"logs2/cleantests-{suite}.out"), "manifest_sha256": sha(f"{sdir}/code_sha256.txt"),
                    "timing": tm, "timing_sha256": sha(tf)}
    rec["code_sha256"] = code
    if arm == "vmix":  # 2'. A5: the cache check replaces the probe predicate
        need(not probe, "vmix: the probe predicate is replaced by the cache check (no --probe)")
        bind = make_bind(A_PATH, CACHE_DIR, LISTS, ENC, RAYS_F, code)
        rec["cache_bind"] = bind
        if cache_pending:
            need(not cache_record, "--cache_pending and --cache_record are exclusive")
            rec["cache"] = {"status": "pending: the trainer verifies a record against cache_bind at update 2000 (mrv/cache_verify), holds without one, pins the first"}
        else:
            import torch
            data = {"ticks": {sc_: torch.load(f"{ENC}/{sc_}.pt")["ticks"] for sc_ in bind["train"]}, "rays": torch.load(RAYS_F)}
            ok, why, rsha, _ = verify(cache_record, bind, data)
            need(ok, f"cache record refused by the shared verifier: {why}")
            cr = json.load(open(cache_record))
            rec["cache"] = {"status": "passed", "record": cache_record, "sha256": rsha, "parity": cr.get("parity"),
                            "mse": {k_: cr["mse"].get(k_) for k_ in ("train_mean", "dev_mean")}}
        return finish(rec, t2)
    if control:  # 2''. A5 signed attribution control: key + jitter, no probe predicate, no cache
        need(not probe and not cache_record and not cache_pending, "control: no probe / cache flags")
        return finish(rec, t2)
    # 2. probe (vclean)
    need(not cache_record and not cache_pending, "vclean: no cache flags")
    need(os.path.isfile(probe), f"missing probe record {probe}")
    pr = json.load(open(probe)); rd = pr.get("a4_reading", {})
    need(rd.get("valid") is True and rd.get("decision") == "LAUNCH" and rd.get("launch") is True, f"probe reading is {rd.get('decision')} (valid {rd.get('valid')}): no launch")
    for f_, h_ in pr.get("code_sha256", {}).items():
        need(code.get(f_) == h_, f"probe code {f_} is not the suite-tested code")
    need(set(pr.get("code_sha256", {})) == {"scripts/clean_probe.py", "mrv/clean_video.py", "mrv/sd.py"}, "probe code record incomplete")
    for path, v in pr["ckpts"].items():
        need(v.get("valid") is True and os.path.isfile(path) and sha(path) == v.get("ckpt_sha256"), f"probe checkpoint {path} invalid or changed")
    need(sorted(pr["ckpts"]) == ["runs/vpriv/ckpt_12000.pt", "runs/vpriv/ckpt_8000.pt"], "probe checkpoints are not vpriv 8k / 12k")
    rec["probe"] = {"file": probe, "sha256": sha(probe), "decision": "LAUNCH", "jitter_allowed": rd.get("jitter_allowed"),
                    "ckpt_sha256": {p_: v["ckpt_sha256"] for p_, v in pr["ckpts"].items()}}
    return finish(rec, t2)


def finish(rec, t2):
    # 4. T2 -> gate rho (both arms)
    tdir = f"runs/ctest_clean_t2_{t2}"
    tfails = last_line(f"logs2/cleant2-{t2}.out", "CLEAN_T2", tdir)
    manifest(f"{tdir}/code_sha256.txt", T2_FILES)
    jf = f"{tdir}/t2.json"
    need(os.path.isfile(jf), f"missing T2 result {jf} (render or comparison did not complete): rho undecided")
    t = json.load(open(jf))
    need(t.get("complete") is True and t.get("fields_equal") is True and fin(t.get("max_diff")), f"T2 not on matched inputs / incomplete {t}: rho undecided")
    need(t.get("rho") == 1.0 and t.get("method") == "clean_off" and t.get("ckpt_sha") == rec["a_sha"], f"T2 is not clean memory-off of A@20k at rho 1.0 ({t.get('method')}, {t.get('rho')})")
    if t["max_diff"] < 1e-2 and tfails == 0:
        rho = 1.0
    elif t["max_diff"] >= 1e-2 and tfails == 1:
        rho = 0.95
    else:
        raise Refuse(f"T2 log FAILS={tfails} inconsistent with max |diff| {t['max_diff']}: rho undecided")
    rec["t2"] = {"job": t2, "dir": tdir, "result": t, "sha256": sha(jf), "log_sha256": sha(f"logs2/cleant2-{t2}.out")}
    rec["rho_gate"] = rho
    rec["valid"] = True
    return rec


p = argparse.ArgumentParser()
p.add_argument("--suite", default=""); p.add_argument("--t2", default=""); p.add_argument("--probe", default="")
p.add_argument("--out", default=""); p.add_argument("--verify", default="")
p.add_argument("--arm", default="vclean", choices=["vclean", "vmix"])
p.add_argument("--cache_record", default=""); p.add_argument("--cache_pending", action="store_true")
p.add_argument("--control", action="store_true", help="with --arm vclean: the A5 key + jitter attribution control")
a = p.parse_args()
try:
    if a.verify:
        old = json.load(open(a.verify)); i = old["inputs"]
        new = build(i["suite"], i["t2"], i["probe"], i.get("arm", "vclean"), i.get("cache_record", ""), i.get("cache_pending", False), i.get("control", False))
        need(json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True), "record differs from what its inputs give now")
        print("VCLEAN_GATE OK rho_gate", new["rho_gate"], a.verify)
    else:
        need(a.suite.isdigit() and a.t2.isdigit() and a.out and (a.probe or a.arm == "vmix" or a.control), "need --suite <job id> --t2 <job id> [--probe <json>] --out <record>")
        rec = build(a.suite, a.t2, a.probe, a.arm, a.cache_record, a.cache_pending, a.control)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "x") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
        print("VCLEAN_GATE PASS arm", rec["arm"], "recipe", rec["recipe"], "rho_gate", rec["rho_gate"], a.out)
except Refuse as e:
    print("VCLEAN_GATE REFUSE:", e); sys.exit(1)
