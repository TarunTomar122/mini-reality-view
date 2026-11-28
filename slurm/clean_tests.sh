#!/bin/bash
# Pre-training tests of vclean (council/AGREED-causal-r1.md A4 "T1-T7 re-checked for the new ring"; council/codereview-vclean-astra.md)
# except T2 (slurm/clean_t2.sh, on a 2080 Ti like the signed A render). Run on the training slice type (fp32 tests); exits nonzero
# on ANY failure; the last line is the result line read by scripts/vclean_gate.py.
#   sbatch -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=4:00:00 -J cleantests -o logs2/%x-%j.out slurm/clean_tests.sh
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
PY=.venv/bin/python; A=runs/longA/ckpt_20000.pt; T=runs/ctest_clean_${SLURM_JOB_ID:-manual}; mkdir -p $T; FAILS=0
sha256sum mrv/clean_video.py mrv/noise.py mrv/sd.py scripts/train_clean.py scripts/clean_rollout.py scripts/clean_tests.py scripts/clean_probe.py \
  scripts/vclean_gate.py scripts/det_run.py slurm/clean_tests.sh | tee $T/code_sha256.txt
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
TR="$PY scripts/train_clean.py --init $A --K 4 --steps 100000 --timing"
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
