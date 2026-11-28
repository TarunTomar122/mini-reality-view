"""A3 trigger for vpriv (council/AGREED-causal-r1.md A3 CONVERGED; Astra final review 28 Nov): recompute the decision panels
NOW with the signed comparator (fixed recipe: A-derived k=3 count lists, floor 0, cold_ref B) on signed-scorer metrics; check
panel identity, 6 drives x seeds 0/1, finite estimates; bind every input / output digest in council/vpriv_launch.json.
Prints 1 or 2 (private clips) on stdout when the trigger is met, else a NOTRIGGER line; asserts on any malformed input.
  .venv/bin/python scripts/a3_trigger.py      (run by slurm/launch_vpriv.sh)
"""
import hashlib, json, math, os, subprocess, sys

sha1 = lambda f: hashlib.sha1(open(f, "rb").read()).hexdigest()[:12]
sha256 = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
assert sha1("scripts/temporal_compare.py") == "fc7ee2b886a5" and sha1("scripts/temporal_metrics.py") == "764dd3ec95db", "unsigned tools"
COMP = {"A20k": ("runs/troll/m_A20k_indep_dev.json", "runs/count/count_A20k_dev_k3.json"),
        "cold": ("runs/troll/m_Aoff_rho0.95_dev.json", "runs/count/count_Aoff0.95_dev_k3.json")}
OUT = "runs/s_lock/gate_vpriv"; os.makedirs(OUT, exist_ok=True)
bind = {"tools": {"temporal_compare": "fc7ee2b886a5", "temporal_metrics": "764dd3ec95db"}, "trigger_sha": sha1(__file__), "panels": {}}


def panel(k, ref):
    cand = f"runs/troll/m_V{k}_dev.json"; comp, cl = COMP[ref]
    for f in (cand, comp):
        m = json.load(open(f))["drives"]
        assert len(m) == 12 and len({r.rsplit("_s", 1)[0] for r in m}) == 6 and {r.rsplit("_s", 1)[1] for r in m} == {"0", "1"}, f"{f}: not 6 drives x seeds 0/1"
        assert all(v.get("scorer_sha") == "764dd3ec95db" for v in m.values()), f"{f}: not scored by the signed scorer"
    out = subprocess.run([sys.executable, "scripts/temporal_compare.py", "--compare", cand, comp, "--count_list", cl, "--floor", "0",
                          "--cold_ref", "B"], capture_output=True, text=True, check=True).stdout
    o = f"{OUT}/cmp_V{k}__{ref}.json"; open(o, "w").write(out)
    d = json.loads(out.split("TCOMPARE ", 1)[1])
    assert d["A"] == cand and d["B"] == comp and d["drives"] == 6 and d["runs"] == 12 and not d["unpaired_drives"], f"{o}: panel identity"
    assert d["cold_ref"] == "B" and d["count_list"] == cl, f"{o}: recipe"
    for key, fields in (("ID", ("rel", "diff")), ("W", ("rel", "diff")), ("car_f1", ("diff",)), ("late_lpips_gap", ("diff",))):
        for f in fields:
            v = d[key][f]
            assert isinstance(v, (int, float)) and math.isfinite(v), f"{o}: {key}.{f} = {v!r} not finite"
    bind["panels"][f"V{k}_{ref}"] = {"cand": [cand, sha256(cand)], "comp": [comp, sha256(comp)], "count_list": [cl, sha256(cl)], "out": [o, sha256(o)]}
    return d


for k in ("8k", "12k", "16k"):
    d = panel(k, "A20k")
    elig = d["ID"].get("can_pass") is True and d["W"].get("can_pass") is True
    red = min(-d["ID"]["rel"], -d["W"]["rel"])
    joint = elig and red > 0.094 and d["car_f1"]["diff"] >= -0.03 and d["late_lpips_gap"]["diff"] <= 0.03
    print(f"V{k} vs A: eligible {elig} smaller reduction {red:+.3f}  F1 diff {d['car_f1']['diff']:+.4f}  late {d['late_lpips_gap']['diff']:+.4f}  joint pass {joint}", file=sys.stderr)
    if joint:
        print("NOTRIGGER joint pass at", k); sys.exit(0)
if not (d["car_f1"]["diff"] >= -0.03 and d["late_lpips_gap"]["diff"] <= 0.03):
    print("NOTRIGGER 16k safety miss"); sys.exit(0)
c = panel("16k", "cold")
priv = 1 if (c["ID"]["diff"] < 0 and c["W"]["diff"] < 0) else 2
print(f"V16k vs own cold: ID diff {c['ID']['diff']:+.4f}  W diff {c['W']['diff']:+.4f} -> private_clips {priv}", file=sys.stderr)
bind["private_clips"] = priv
json.dump(bind, open("council/vpriv_launch.json", "w"), indent=1)
print(priv)
