#!/bin/bash
# Pre-training tests of vclean / vmix (council/AGREED-causal-r1.md A4 "T1-T7 re-checked for the new ring", A5; council/codereview-vclean-astra.md)
# except T2 (slurm/clean_t2.sh, on a 2080 Ti like the signed A render). Run on the training slice type (fp32 tests); exits nonzero
# on ANY failure; the last line is the result line read by scripts/vclean_gate.py.
#   sbatch -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=4:00:00 -J cleantests -o logs2/%x-%j.out slurm/clean_tests.sh
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
PY=.venv/bin/python; A=runs/longA/ckpt_20000.pt; T=runs/ctest_clean_${SLURM_JOB_ID:-manual}; mkdir -p $T; FAILS=0
sha256sum mrv/clean_video.py mrv/noise.py mrv/sd.py scripts/train_clean.py scripts/clean_rollout.py scripts/clean_tests.py scripts/clean_probe.py \
  scripts/vclean_gate.py scripts/det_run.py slurm/clean_tests.sh scripts/cache_check.py scripts/build_cache.py \
  scripts/causal_rollout.py mrv/causal_video.py mrv/cache_verify.py | tee $T/code_sha256.txt
chk() { if [ "$1" != 0 ]; then echo "FAIL $2 (exit $1)"; FAILS=$((FAILS+1)); fi; }
nvidia-smi --query-gpu=name --format=csv,noheader
echo "== clean_tests.py part 0 + tiny UNet (CPU fp32)"; $PY scripts/clean_tests.py --small; chk $? clean_tests-small
echo "== clean_tests.py full model T1 T3a T4 (A@20k, fp32, TF32 off)"; $PY scripts/clean_tests.py --init $A --K 4; chk $? clean_tests-full
[ $FAILS = 0 ] || { echo "fail-fast: clean_tests.py failed, skipping the rest"; echo "CLEAN_TESTS_SH FAILS=$FAILS dir=$T"; exit $FAILS; }
DRV=$($PY -c "import json;print(json.load(open('data/temporal_lists.json'))['dev'][0])"); echo $DRV > $T/one.txt
echo "== random-weight vclean checkpoint at rho 0.5 (T5 / T6 only; rho < 1 so --salt_from really changes the noise)"
$PY - <<P; chk $? rand-ckpt
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig
from mrv.sd import LORA_TARGETS
from mrv.clean_video import add_clean_temporal
ck = torch.load("$A", map_location="cpu"); r = ck["args"]["rank"]
u = UNet2DConditionModel.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="unet"); u.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS))
TW = add_clean_temporal(u, 4); g = torch.Generator().manual_seed(7)
for w in TW:
    for n, q in w.temporal.named_parameters():
        if not (n.startswith("norm") or n.startswith("ln")): q.data.add_(torch.randn(q.shape, generator=g) * 0.05)
d = DDIMScheduler.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="scheduler"); d.set_timesteps(25)
torch.save({**ck, "temporal": [w.temporal.state_dict() for w in TW], "K": 4, "rho": 0.5, "step": 0, "memory": "tstar_final_call",
            "t_star": int(d.timesteps[-1]), "block_names": [n for n, m in u.named_modules() if m.__class__.__name__ == "WithCleanTemporal"]}, "$T/rand.pt")
P
# T5 / T6 compare separate processes bit-exactly: deterministic GPU kernels switched on from outside (as the causal suite)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
R="$PY scripts/det_run.py scripts/clean_rollout.py --ckpt $T/rand.pt --split dev --drive_list $T/one.txt --seeds 0"
echo "== T5: 60 ticks; from tick 30 replace cameras only / private draws only / both"
$R --max_ticks 60 --out $T/t5_ctrl > /dev/null; chk $? T5-ctrl
$R --max_ticks 60 --swap --swap_at 30 --out $T/t5_cam > /dev/null; chk $? T5-cam
$R --max_ticks 60 --salt_from 30 --out $T/t5_salt > /dev/null; chk $? T5-salt
$R --max_ticks 60 --swap --swap_at 30 --salt_from 30 --out $T/t5_both > /dev/null; chk $? T5-both
echo "== T6: --reset_at 20 suffix == fresh start at 20; unreset control differs"
$R --max_ticks 40 --reset_at 20 --out $T/t6_reset > /dev/null; chk $? T6-reset
$R --max_ticks 40 --start_tick 20 --out $T/t6_fresh > /dev/null; chk $? T6-fresh
$R --max_ticks 40 --out $T/t6_ctrl > /dev/null; chk $? T6-ctrl
$PY - <<P; chk $? T5-T6
import torch, sys
f = lambda d: torch.load(f"$T/{d}/${DRV}_s0.pt")
c = f("t5_ctrl"); cam, salt, both = f("t5_cam"), f("t5_salt"), f("t5_both")
ok5 = all(len(x["lat"]) == 60 and torch.equal(x["ticks"], c["ticks"]) and not bool(x["hard"].any()) for x in (c, cam, salt, both))
ok5 = ok5 and all(x["rho"] == 0.5 and x["method"] == "clean_stream" and x["memory"] == "tstar_final_call" for x in (c, cam, salt, both))
msgs = []
for name, x in (("cam", cam), ("salt", salt), ("both", both)):
    pre = (c["lat"][:30].float() - x["lat"][:30].float()).abs().max().item(); post = (c["lat"][30:].float() - x["lat"][30:].float()).abs().max().item()
    ok5 = ok5 and pre == 0.0 and post > 1e-3; msgs.append(f"{name}: prefix {pre:.3g} (=0) suffix {post:.3g} (>0)")
ok5 = ok5 and cam["other"] not in ("", "${DRV}") and torch.equal(cam["src_ticks"][:30], c["src_ticks"][:30]) and torch.equal(both["src_ticks"], cam["src_ticks"]) \
      and torch.equal(salt["src_ticks"], c["src_ticks"]) and cam["salt_from"] == -1 and salt["salt_from"] == 30 and both["salt_from"] == 30
D = cam["other"]
prov = lambda x, cams_from, salt_from: x["src_scene"] == ["${DRV}"] * cams_from + [D] * (60 - cams_from) and \
       [k.endswith("|salt") for k in x["noise_key"]] == [False] * salt_from + [True] * (60 - salt_from)
ok5 = ok5 and prov(c, 60, 60) and prov(cam, 30, 60) and prov(salt, 60, 30) and prov(both, 30, 30)
msgs.append("provenance ctrl/cam/salt/both %s %s %s %s" % (prov(c, 60, 60), prov(cam, 30, 60), prov(salt, 60, 30), prov(both, 30, 30)))
d_bc = (both["lat"][30:].float() - cam["lat"][30:].float()).abs().max().item(); d_bs = (both["lat"][30:].float() - salt["lat"][30:].float()).abs().max().item()
ok5 = ok5 and d_bc > 1e-3 and d_bs > 1e-3
print(("PASS" if ok5 else "FAIL"), "T5 60 aligned ticks (rho 0.5), donor %s;" % cam["other"], "; ".join(msgs), "; both vs cam %.3g, both vs salt %.3g (>0)" % (d_bc, d_bs))
rs, fr, ct = f("t6_reset"), f("t6_fresh"), f("t6_ctrl")
s = (rs["lat"][20:].float() - fr["lat"].float()).abs().max().item(); dc = (ct["lat"][20:25].float() - rs["lat"][20:25].float()).abs().max().item()
ok6 = len(fr["lat"]) == 20 and s == 0.0 and dc > 1e-3 and bool(rs["restart"][20]) and bool(fr["restart"][0])
print(("PASS" if ok6 else "FAIL"), "T6 reset suffix vs fresh start: %.3g (=0); unreset control differs at ticks 20-24: %.3g (>0); restart flags %s %s" % (s, dc, bool(rs["restart"][20]), bool(fr["restart"][0])))
sys.exit(0 if ok5 and ok6 else 1)
P
echo "== T7: the real trainer (--timing): frozen guard + movement; resume after update 1 == straight run; changed recipe (rays) refused"
TR="$PY scripts/train_clean.py --arm vmix --init $A --K 4 --steps 100000 --timing"  # default jitter {1,41,81}, mix from 2001
$TR --out $T/timing_full --max_steps_this_job 3 > $T/full.log 2>&1; chk $? T7-full
$TR --out $T/timing_split --max_steps_this_job 1 > $T/split1.log 2>&1; chk $? T7-split1
$TR --out $T/timing_split --max_steps_this_job 2 --resume > $T/split2.log 2>&1; chk $? T7-split2
nb=$(grep -c '^{"update"' $T/timing_split/log.jsonl); sb=$(sha256sum $T/timing_split/ckpt.pt | cut -c1-16)
cp data/calib_rays.pt $T/rays_alt.pt
$TR --out $T/timing_split --rays $T/rays_alt.pt --max_steps_this_job 1 --resume > $T/bad.log 2>&1; rb=$?
na=$(grep -c '^{"update"' $T/timing_split/log.jsonl); sa=$(sha256sum $T/timing_split/ckpt.pt | cut -c1-16)
$PY - <<P; chk $? T7
import json, sys
L = lambda d: {x["update"]: x for x in (json.loads(l) for l in open(f"$T/{d}/log.jsonl") if l.startswith('{"update"'))}
F_, S_ = L("timing_full"), L("timing_split")
full, s1 = open("$T/full.log").read(), open("$T/split1.log").read()
okf = all("frozen check after first update" in x and "16/16 temporal out weights moved" in x for x in (full, s1, open("$T/split2.log").read()))
okr = all(u in F_ and u in S_ and F_[u]["batch_sha"] == S_[u]["batch_sha"] and F_[u]["mem_sha"] == S_[u]["mem_sha"]
          and abs(F_[u]["loss"] - S_[u]["loss"]) <= 1e-3 * max(abs(F_[u]["loss"]), 1e-6) + 1e-4 for u in (1, 2, 3))
bad = open("$T/bad.log").read(); okb = $rb != 0 and "incompatible resume: recipe rays" in bad and $nb == $na and "$sb" == "$sa"
tstar = [l for l in full.splitlines() if l.startswith("t* = ")]
print(("PASS" if okf else "FAIL"), "T7 frozen guard (LoRA + adapter bit-identical, 16/16 out weights moved) in straight, split and resumed runs;", tstar[0] if tstar else "t* not printed")
print(("PASS" if okr else "FAIL"), "T7 resume after update 1 == straight run, updates 1-3 (batch_sha, mem_sha, loss):",
      [(u, F_.get(u, {}).get("batch_sha"), S_.get(u, {}).get("batch_sha"), F_.get(u, {}).get("mem_sha"), S_.get(u, {}).get("mem_sha"), F_.get(u, {}).get("loss"), S_.get(u, {}).get("loss")) for u in (1, 2, 3)])
print(("PASS" if okb else "FAIL"), "T7 resume with changed rays refused before any update (exit $rb; log lines $nb -> $na; ckpt sha $sb -> $sa)")
sys.exit(0 if okf and okr and okb and tstar else 1)
P
echo "== T7b (A5 vmix): jitter / source streams across resume with the cache branch (--mix_from 1); pin; resume identity; hold; pickup"
echo "   TEST-ONLY fixtures: a synthetic cache in the builder's format, a synthetic parity reference artifact and dev seed-check artifacts"
echo "   that the PRODUCTION verifier (mrv/cache_verify.py) re-measures (parity / seed diffs recomputed from the artifacts, digests bound)"
$PY - <<P; chk $? fake-cache
import hashlib, io, json, os, torch
from mrv.cache_verify import sha, read, hb, file_code, builder_roll_code, rollout_roll_code, ray_sha, DEV_REF
L = json.load(open("data/temporal_lists.json")); RAYS = torch.load("data/calib_rays.pt"); C = "$T/fake_cache"
for d in (C, "$T/fake_parity", "$T/fake_devseed"):
    os.makedirs(d, exist_ok=True)
A_SHA, ROLL, RROLL = sha("$A")[:16], builder_roll_code(), rollout_roll_code(); files, encs = {}, {}
tr, dv = sorted(L["train"]), list(L["dev"])
for sc in tr:
    eb = read(f"data/enc_tpv10/{sc}.pt"); encs[sc] = hb(eb)[:16]; E = torch.load(io.BytesIO(eb)); g = torch.Generator().manual_seed(len(sc))
    hard = [bool(h) for h in E["hard"]]; rs, fr = [], True
    for h in hard:
        rs.append(True if h else fr); fr = h
    lat = (E["lat"].float() + 0.5 * torch.randn(E["lat"].shape, generator=g)).half(); lat[torch.tensor(hard)] = 0
    torch.save({"lat": lat, "ticks": E["ticks"], "src_ticks": torch.tensor([int(t) for t in E["ticks"]]), "ok": torch.tensor([bool(x) for x in E["ok"]]),
                "hard": torch.tensor(hard), "restart": torch.tensor(rs), "amp": "torch.float16", "ckpt_sha": A_SHA, "roll_code": ROLL, "method": "cache_off",
                "rho": 1.0, "seed": 0, "merge": False, "salt_from": -1, "start_tick": 0, "ray_sha": ray_sha(RAYS[sc]), "enc_sha": encs[sc], "note": "TEST ONLY"}, f"{C}/{sc}_s0.pt")
    files[sc] = sha(f"{C}/{sc}_s0.pt")
drv = tr[0]; c = torch.load(f"{C}/{drv}_s0.pt")
ref = dict(c, method="causal_off", roll_code=RROLL, swap=False, noise_key=[f"{drv}|0|{int(t)}" for t, h in zip(c["ticks"], c["hard"]) if not bool(h)], note="TEST ONLY")
torch.save(ref, "$T/fake_parity/ok.pt"); torch.save(dict(ref, lat=(ref["lat"].float() + 0.05).half()), "$T/fake_parity/bad.pt")
dev_files, dev_seed = {}, {}
for sc in dv:
    db = read(f"{DEV_REF}/{sc}_s0.pt"); dev_files[sc] = hb(db); dr = torch.load(io.BytesIO(db))
    s = {"lat": dr["lat"][:5], "ticks": dr["ticks"][:5], "hard": dr["hard"][:5], "ckpt_sha": A_SHA, "rho": 1.0, "method": "causal_off", "roll_code": RROLL,
         "noise_key": [f"{sc}|0|{int(t)}" for t, h in zip(dr["ticks"][:5], dr["hard"][:5]) if not bool(h)], "note": "TEST ONLY"}
    torch.save(s, f"$T/fake_devseed/{sc}_s0.pt"); dev_seed[sc] = {"file": f"$T/fake_devseed/{sc}_s0.pt", "sha256": sha(f"$T/fake_devseed/{sc}_s0.pt")}
par = lambda f: {"drive": drv, "exit": 0, "fields_equal": True, "whole": True, "max_diff": 0.0, "ref_ckpt_sha": A_SHA, "ref_rho": 1.0, "ref_method": "causal_off",
                 "ref_roll_code": RROLL, "ref_file": f, "ref_file_sha256": sha(f)}
rec = {"kind": "cache_check", "pass": True, "failures": [], "cache": C, "ckpt_sha": A_SHA, "roll_code": ROLL, "lists": "data/temporal_lists.json",
       "enc": "data/enc_tpv10", "rays": "data/calib_rays.pt", "build": {"merge": False, "batch": 8, "amp": "torch.float16"}, "files": files, "n_files": len(files),
       "enc_sha": encs, "parity": par("$T/fake_parity/ok.pt"), "dev_ref": DEV_REF, "dev_files": dev_files, "dev_seed": dev_seed,
       "mse": {"train_mean": 0.1, "dev_mean": 0.1, "train_per_drive": {s: 0.1 for s in tr}, "dev_per_drive": {s: 0.1 for s in dv}}, "code_sha256": file_code(), "note": "TEST ONLY"}
W = lambda n, r: json.dump(r, open(f"$T/fake_{n}.json", "w"), indent=1)
W("ok", rec); W("ok2", dict(rec, note="TEST ONLY: a second, different but valid record"))
W("bad_digest", dict(rec, files=dict(files, **{drv: "0" * 64})))
W("wrong_ckpt", dict(rec, ckpt_sha="0" * 16))  # another renderer's cache
W("wrong_roll", dict(rec, roll_code="deadbeef0000"))  # a cache from a different builder
W("bad_parity", dict(rec, parity=par("$T/fake_parity/bad.pt")))  # the record ASSERTS parity; the re-measured artifact differs by 0.05
W("stale_enc", dict(rec, enc_sha=dict(encs, **{drv: "f" * 16})))  # cache built from another encoding than the one the trainer loads
P
TM="$TR --mix_from 1 --cache $T/fake_cache --cache_record $T/fake_ok.json"
$TM --out $T/timing_mix_full --max_steps_this_job 3 > $T/mix_full.log 2>&1; chk $? T7b-full
$TM --out $T/timing_mix_split --max_steps_this_job 2 > $T/mix_split1.log 2>&1; chk $? T7b-split1
np=$(grep -c '^{"update"' $T/timing_mix_split/log.jsonl); sp=$(sha256sum $T/timing_mix_split/ckpt.pt | cut -c1-16)
$TR --mix_from 1 --cache $T/fake_cache --cache_record $T/fake_ok2.json --out $T/timing_mix_split --max_steps_this_job 1 --resume > $T/mix_pin.log 2>&1; rp=$?
np2=$(grep -c '^{"update"' $T/timing_mix_split/log.jsonl); sp2=$(sha256sum $T/timing_mix_split/ckpt.pt | cut -c1-16)
$TM --out $T/timing_mix_split --max_steps_this_job 1 --resume > $T/mix_split2.log 2>&1; chk $? T7b-split2
# resume identity negatives: (a) a checkpoint past mix_from with no pin; (b) a checkpoint made under a (passed) gate, resumed without it
$PY - <<P; chk $? T7b-craft
import os, torch
s = torch.load("$T/timing_mix_full/ckpt.pt", map_location="cpu")
for d, mod in (("timing_nopin", {"cache_record_sha256": None}), ("timing_gated", {"gate_record_sha256": "ab" * 32, "cache_approved_sha256": s["cache_record_sha256"]})):
    os.makedirs(f"$T/{d}", exist_ok=True); torch.save(dict(s, **mod), f"$T/{d}/ckpt.pt")
P
$TM --out $T/timing_nopin --max_steps_this_job 1 --resume > $T/nopin.log 2>&1; rn=$?
$TM --out $T/timing_gated --max_steps_this_job 1 --resume > $T/gated.log 2>&1; rg=$?
TH="$TR --mix_from 2 --cache $T/fake_cache"
$TH --out $T/timing_hold_ctrl --cache_record $T/fake_ok.json --max_steps_this_job 5 > $T/hold_ctrl.log 2>&1; chk $? T7b-hold-ctrl
$TH --out $T/timing_hold --cache_record $T/absent.json --max_steps_this_job 5 > $T/hold1.log 2>&1; rh1=$?
nh1=$(grep -c '^{"update"' $T/timing_hold/log.jsonl)
NEG="bad_digest wrong_ckpt wrong_roll bad_parity stale_enc"; RNEG=""
for n in $NEG; do $TH --out $T/timing_hold --cache_record $T/fake_$n.json --max_steps_this_job 5 --resume > $T/hold_$n.log 2>&1; RNEG="$RNEG $?"; done
nh2=$(grep -c '^{"update"' $T/timing_hold/log.jsonl)
$TH --out $T/timing_hold --cache_record $T/fake_ok.json --max_steps_this_job 3 --resume > $T/hold3.log 2>&1; rh3=$?
rm -f $T/late.json  # pickup: the record appears while the run trains toward update 30; no resume needed
$TR --mix_from 30 --cache $T/fake_cache --cache_record $T/late.json --out $T/timing_late --max_steps_this_job 34 > $T/late.log 2>&1 &
PID=$!
for k in $(seq 900); do grep -q '^{"update": 3,' $T/timing_late/log.jsonl 2>/dev/null && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
cp $T/fake_ok.json $T/late.json.tmp && mv $T/late.json.tmp $T/late.json
wait $PID; rl=$?
$PY - <<P; chk $? T7b
import json, sys, torch
rows = lambda d: [json.loads(l) for l in open(f"$T/{d}/log.jsonl") if l.strip()]
L = lambda d: {x["update"]: x for x in rows(d) if "update" in x}
F_, S_ = L("timing_mix_full"), L("timing_mix_split")
same = lambda A, B, us: all(u in A and u in B and all(A[u][k] == B[u][k] for k in ("batch_sha", "mem_sha", "mem_t", "mem_src"))
                            and abs(A[u]["loss"] - B[u]["loss"]) <= 1e-3 * max(abs(A[u]["loss"]), 1e-6) + 1e-4 for u in us)
ok1 = same(F_, S_, (1, 2, 3)) and "resumed at 2" in open("$T/mix_split2.log").read()
src_ok = F_.get(1, {}).get("mem_src") == ["key", "key"] and all(sorted(F_.get(u, {}).get("mem_src", [])) == ["cache", "key"] for u in (2, 3))
jit_ok = all(t_ in (1, 41, 81) for u in (1, 2, 3) for t_ in F_.get(u, {}).get("mem_t", [0]))
print(("PASS" if ok1 and src_ok and jit_ok else "FAIL"), "T7b split after update 2 + resume == straight run, updates 1-3 (batch_sha, mem_sha incl. jitter + coin, mem_t, mem_src, loss):",
      [(u, F_.get(u, {}).get("mem_t"), F_.get(u, {}).get("mem_src"), F_.get(u, {}).get("mem_sha"), S_.get(u, {}).get("mem_sha")) for u in (1, 2, 3)])
pin = open("$T/mix_pin.log").read()
okp = $rp != 0 and "pinned" in pin and $np == $np2 and "$sp" == "$sp2" and any("cache_pinned" in x for x in rows("timing_mix_full"))
print(("PASS" if okp else "FAIL"), "T7b a different VALID record on resume after the pin is refused (exit $rp; log lines $np -> $np2; ckpt $sp -> $sp2)")
nopin, gated = open("$T/nopin.log").read(), open("$T/gated.log").read()
okr = $rn != 0 and "no pinned cache record" in nopin and $rg != 0 and "gate record" in gated
print(("PASS" if okr else "FAIL"), "T7b resume identity: a checkpoint past mix_from without a pin is refused (exit $rn); a gated checkpoint resumed without its gate is refused (exit $rg)")
neg = dict(zip("$NEG".split(), [int(x) for x in "$RNEG".split()]))
want = {"bad_digest": "digest differs", "wrong_ckpt": "ckpt_sha", "wrong_roll": "reviewed builder", "bad_parity": "parity recomputed", "stale_enc": "trainer loaded"}
hn = {n: open(f"$T/hold_{n}.log").read() for n in neg}
okn = all(neg[n] == 0 and "CLEAN_TRAIN_HOLD 2" in hn[n] and want[n] in hn[n] for n in neg)
h1, h3 = open("$T/hold1.log").read(), open("$T/hold3.log").read()
ck = torch.load("$T/timing_hold/ckpt.pt", map_location="cpu")
okh = $rh1 == 0 and "CLEAN_TRAIN_HOLD 2" in h1 and $nh1 == 2 and okn and $nh2 == 2 and $rh3 == 0 and "resumed at 2" in h3 and "CLEAN_TRAIN_DONE 5" in h3 and ck["step"] == 5
C_, H_ = L("timing_hold_ctrl"), L("timing_hold")
okc = same(C_, H_, (1, 2, 3, 4, 5)) and all(sorted(H_.get(u, {}).get("mem_src", [])) == ["cache", "key"] for u in (3, 4, 5))
print(("PASS" if okh else "FAIL"), "T7b hold at mix_from 2: no record (exit $rh1) and every negative fixture holds with its reason and no update",
      {n: (neg[n], want[n] in hn[n]) for n in neg}, "(%d -> %d rows); resumed with the passing record to 5 (exit $rh3)" % ($nh1, $nh2))
print(("PASS" if okc else "FAIL"), "T7b hold + resume == uninterrupted run with the record, updates 1-5 (draws, sources, loss)")
LT = rows("timing_late"); U = {x["update"]: x for x in LT if "update" in x}
okl = $rl == 0 and any(x.get("cache_acquired") == 30 for x in LT) and not any("hold" in x for x in LT) and "CLEAN_TRAIN_DONE 34" in open("$T/late.log").read() \
      and U.get(30, {}).get("mem_src") == ["key", "key"] and all(sorted(U.get(u, {}).get("mem_src", [])) == ["cache", "key"] for u in range(31, 35))
print(("PASS" if okl else "FAIL"), "T7b a record arriving during training is picked up at update mix_from 30 without a resume (exit $rl)")
sys.exit(0 if ok1 and src_ok and jit_ok and okp and okr and okh and okc and okl else 1)
P
echo "== timing: 100 throwaway updates on this slice (step_s, mem_s, peak GB)"
$TR --out $T/timing_100 --max_steps_this_job 100 > $T/t100.log 2>&1; chk $? timing-100
$PY - <<P; chk $? timing-record
import json, subprocess, sys
R = [json.loads(l) for l in open("$T/timing_100/log.jsonl") if l.startswith('{"update"')]
U = {x["update"]: x for x in R}; W = [U[u] for u in range(11, 101) if u in U]  # updates 11..100 (after warm-up of kernels)
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
ok = 100 in U and len(W) == 90
rec = {"updates": max(U) if U else 0, "step_s": sum(x["step_s"] for x in W) / len(W) if W else None, "mem_s": sum(x["mem_s"] for x in W) / len(W) if W else None,
       "data_s": sum(x["data_s"] for x in W) / len(W) if W else None, "e2e_s": sum(x["e2e_window_s"] for x in W) / len(W) if W else None,
       "peak_gb": max(x["mem_gb"] for x in R) if R else None, "gpu": gpu, "code_sha": R[-1]["code_sha"] if R else None,
       "frozen_check": "frozen check after first update" in open("$T/t100.log").read(), "job": "${SLURM_JOB_ID:-manual}"}
json.dump(rec, open("$T/timing.json", "w"), indent=1)
print(("PASS" if ok and rec["frozen_check"] else "FAIL"), "timing 100 updates:", rec)
sys.exit(0 if ok and rec["frozen_check"] else 1)
P
echo "CLEAN_TESTS_SH FAILS=$FAILS dir=$T"
exit $FAILS
