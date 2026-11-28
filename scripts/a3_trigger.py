"""A3 trigger for vpriv (council/AGREED-causal-r1.md A3 CONVERGED; Astra final review + launch review 28 Nov).
Recomputes the decision panels with the signed comparator (A-derived k=3 count lists, floor 0, cold_ref B) on IMMUTABLE
snapshots of the metrics files, after checking provenance:
  candidate metrics roll == runs/troll/V<k>k_dev, cold == runs/troll/Aoff_rho0.95_dev, scorer 764dd3ec95db, drives == exactly
  the 6 dev drives x seeds 0 / 1; every rollout file: snapshot runs/vlive/ckpt_<k>000.pt with ckpt_sha == its sha256[:16],
  step, causal_stream, K 4, rho 0.95, reviewed roll_code, no reset / swap / salt / start / truncation; metrics newer than rollouts;
  comparators: roll path, drives, scorer.
Writes a NEW record directory (never overwrites): --launch -> runs/vpriv_gate/<UTC stamp>/, else runs/s_lock/gate_vpriv/dry-<stamp>/.
stdout last line: "<private_clips> <record dir>" when the trigger is met, else "NOTRIGGER ..."; exit 3 on any failed check.
  .venv/bin/python scripts/a3_trigger.py [--launch]
"""
import datetime, hashlib, json, math, os, shutil, subprocess, sys

ROLL_CODE = "ccccba4dfda8"  # reviewed rollout fingerprint (causal_rollout.py + causal_video.py + sd.py; rho decision binding)


def need(cond, msg):
    if not cond:
        print(f"GATE_ERR {msg}"); sys.exit(3)


sha1 = lambda f: hashlib.sha1(open(f, "rb").read()).hexdigest()[:12]
sha256 = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
need(sha1("scripts/temporal_compare.py") == "fc7ee2b886a5", "temporal_compare.py is not the signed comparator")
need(sha1("scripts/temporal_metrics.py") == "764dd3ec95db", "temporal_metrics.py is not the signed scorer")
launch = "--launch" in sys.argv[1:]
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
REC = f"runs/vpriv_gate/{stamp}" if launch else f"runs/s_lock/gate_vpriv/dry-{stamp}"
os.makedirs(REC, exist_ok=False)
DEV = json.load(open("data/temporal_lists.json"))["dev"]
need(len(DEV) == 6, "dev list is not 6 drives")
RUNS = {f"{d}_s{s}" for d in DEV for s in (0, 1)}
COMP = {"A20k": ("runs/troll/m_A20k_indep_dev.json", "runs/troll/A20k_indep_dev", "runs/count/count_A20k_dev_k3.json"),
        "cold": ("runs/troll/m_Aoff_rho0.95_dev.json", "runs/troll/Aoff_rho0.95_dev", "runs/count/count_Aoff0.95_dev_k3.json")}
rec = {"stamp": stamp, "launch": launch, "tools": {"temporal_compare": "fc7ee2b886a5", "temporal_metrics": "764dd3ec95db"},
       "trigger_sha256": sha256(__file__), "roll_code": ROLL_CODE, "panels": {}, "checked": {}}
snap = {}


def snapshot(f):
    """copy once into the record dir; every later read (checks, comparator, digest) uses the copy"""
    if f not in snap:
        dst = os.path.join(REC, os.path.basename(f)); shutil.copyfile(f, dst); os.chmod(dst, 0o444); snap[f] = dst
    return snap[f]


def check_metrics(path, roll, cold=None):
    m = json.load(open(snapshot(path)))
    need(m.get("roll") == roll, f"{path}: roll {m.get('roll')!r} != {roll}")
    if cold is not None:
        need(m.get("cold") == cold, f"{path}: cold {m.get('cold')!r} != {cold}")
    rec["checked"][path] = {"roll": m.get("roll"), "cold": m.get("cold")}
    need(set(m["drives"]) == RUNS, f"{path}: drives {sorted(m['drives'])} != the 6 dev drives x seeds 0 / 1")
    need(all(v.get("scorer_sha") == "764dd3ec95db" for v in m["drives"].values()), f"{path}: not scored by the signed scorer")
    return m


def check_rollouts(k, roll, mpath):
    import torch
    ck = f"runs/vlive/ckpt_{k * 1000}.pt"; cks = sha256(ck)[:16]; newest = 0
    for r in sorted(RUNS):
        f = f"{roll}/{r}.pt"; x = torch.load(f, map_location="cpu", weights_only=False); newest = max(newest, os.path.getmtime(f))
        want = {"snapshot": ck, "snapshot_step": k * 1000, "ckpt_sha": cks, "method": "causal_stream", "K": 4, "rho": 0.95,
                "roll_code": ROLL_CODE, "reset_at": -1, "swap": False, "salt_from": -1, "start_tick": 0, "max_ticks": 0}
        bad = {kk: x.get(kk) for kk, v in want.items() if x.get(kk) != v}
        need(not bad, f"{f}: provenance mismatch {bad}")
    need(os.path.getmtime(mpath) > newest, f"{mpath}: older than its rollouts")
    rec["checked"][f"V{k}k"] = {"ckpt": ck, "ckpt_sha256_16": cks, "rollouts": len(RUNS)}


def panel(k, ref):
    cand, roll = f"runs/troll/m_V{k}k_dev.json", f"runs/troll/V{k}k_dev"
    comp, croll, cl = COMP[ref]
    check_metrics(cand, roll, "runs/troll/Aoff_rho0.95_dev"); check_rollouts(k, roll, cand); check_metrics(comp, croll)
    cs, bs, cls = snapshot(cand), snapshot(comp), snapshot(cl)
    out = subprocess.run([sys.executable, "scripts/temporal_compare.py", "--compare", cs, bs, "--count_list", cls, "--floor", "0",
                          "--cold_ref", "B"], capture_output=True, text=True)
    need(out.returncode == 0 and "TCOMPARE " in out.stdout, f"comparator failed on V{k}k vs {ref}: {out.stderr[-300:]}")
    o = os.path.join(REC, f"cmp_V{k}k__{ref}.json"); open(o, "w").write(out.stdout); os.chmod(o, 0o444)
    d = json.loads(out.stdout.split("TCOMPARE ", 1)[1])
    need(d["A"] == cs and d["B"] == bs and d["drives"] == 6 and d["runs"] == 12 and not d["unpaired_drives"], f"{o}: panel identity")
    need(d["cold_ref"] == "B" and d["count_list"] == cls, f"{o}: recipe")
    for key, fields in (("ID", ("rel", "diff")), ("W", ("rel", "diff")), ("car_f1", ("diff",)), ("late_lpips_gap", ("diff",))):
        for f in fields:
            v = d[key][f]
            need(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v), f"{o}: {key}.{f} = {v!r} not finite")
    rec["panels"][f"V{k}k_{ref}"] = {"cand": [cand, sha256(cs)], "comp": [comp, sha256(bs)], "count_list": [cl, sha256(cls)],
                                     "out": [o, sha256(o)]}
    return d


def finish(result):
    rec["result"] = result
    with open(os.path.join(REC, "record.json"), "x") as fh:
        json.dump(rec, fh, indent=1)
    os.chmod(os.path.join(REC, "record.json"), 0o444)


for k in (8, 12, 16):
    d = panel(k, "A20k")
    elig = d["ID"].get("can_pass") is True and d["W"].get("can_pass") is True
    red = min(-d["ID"]["rel"], -d["W"]["rel"])
    joint = elig and red > 0.094 and d["car_f1"]["diff"] >= -0.03 and d["late_lpips_gap"]["diff"] <= 0.03
    print(f"V{k}k vs A: eligible {elig} smaller reduction {red:+.3f}  F1 diff {d['car_f1']['diff']:+.4f}  late {d['late_lpips_gap']['diff']:+.4f}  joint pass {joint}", file=sys.stderr)
    if joint:
        finish(f"NOTRIGGER joint pass at {k}k"); print(f"NOTRIGGER joint pass at {k}k"); sys.exit(0)
if not (d["car_f1"]["diff"] >= -0.03 and d["late_lpips_gap"]["diff"] <= 0.03):
    finish("NOTRIGGER 16k safety miss"); print("NOTRIGGER 16k safety miss"); sys.exit(0)
c = panel(16, "cold")
priv = 1 if (c["ID"]["diff"] < 0 and c["W"]["diff"] < 0) else 2
print(f"V16k vs own cold: ID diff {c['ID']['diff']:+.4f}  W diff {c['W']['diff']:+.4f} -> private_clips {priv}", file=sys.stderr)
rec["private_clips"] = priv; finish(f"TRIGGER private_clips {priv}")
print(priv, REC)
