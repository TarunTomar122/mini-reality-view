#!/bin/bash
# Pre-training tests of council/AGREED-causal-r1.md except T2 (slurm/causal_t2.sh, on a 2080 Ti). Run on an A6000 (fp32 tests need > 11 GB); exits nonzero on ANY failure.
#   sbatch --gres=gpu:nvidia_rtx_a6000:1 --cpus-per-task=4 --mem=96G --time=2:00:00 -J ctests -o logs2/%x-%j.out slurm/causal_tests.sh
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
PY=.venv/bin/python; A=runs/longA/ckpt_20000.pt; T=runs/ctest_${SLURM_JOB_ID:-manual}; mkdir -p $T; FAILS=0
sha256sum scripts/train_causal.py scripts/causal_rollout.py scripts/causal_tests.py mrv/causal_video.py mrv/noise.py mrv/sd.py slurm/causal_tests.sh scripts/rho_decision.py scripts/rho_rule.py scripts/test_rho_rule.py scripts/test_noise.py scripts/det_run.py slurm/launch_vlive.sh | tee $T/code_sha256.txt
chk() { if [ "$1" != 0 ]; then echo "FAIL $2 (exit $1)"; FAILS=$((FAILS+1)); fi; }
echo "== rho rule CPU fixtures"; (cd scripts && ../$PY test_rho_rule.py); chk $? test_rho_rule
echo "== noise CPU fixtures (A3 vpriv)"; $PY scripts/test_noise.py; chk $? test_noise
echo "== causal_tests.py (T1 T3a T3b T4)"; $PY scripts/causal_tests.py --init $A --K 4; chk $? causal_tests.py
[ $FAILS = 0 ] || { echo "fail-fast: causal_tests.py failed, skipping the rest"; echo "CAUSAL_TESTS_SH FAILS=$FAILS dir=$T"; exit $FAILS; }
DRV=$($PY -c "import json;print(json.load(open('data/temporal_lists.json'))['dev'][0])"); echo $DRV > $T/one.txt
echo "== T2 runs separately on a 2080 Ti (slurm/causal_t2.sh: same GPU type / fp16 as the signed A render)"
echo "== random-weight causal checkpoint (T5 / T6 only)"
$PY - <<P; chk $? rand-ckpt
import torch
from diffusers import UNet2DConditionModel
from peft import LoraConfig
from mrv.sd import LORA_TARGETS
from mrv.causal_video import add_causal_temporal
ck = torch.load("$A", map_location="cpu"); r = ck["args"]["rank"]
u = UNet2DConditionModel.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="unet"); u.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=LORA_TARGETS))
TW = add_causal_temporal(u, 4); g = torch.Generator().manual_seed(7)
for w in TW:
    for n, q in w.temporal.named_parameters():
        if not (n.startswith("norm") or n.startswith("ln")): q.data.add_(torch.randn(q.shape, generator=g) * 0.05)
torch.save({**ck, "temporal": [w.temporal.state_dict() for w in TW], "K": 4, "rho": 0.5, "step": 0,
            "block_names": [n for n, m in u.named_modules() if m.__class__.__name__ == "WithCausalTemporal"]}, "$T/rand.pt")
P
# T5 / T6 compare separate processes bit-exactly: run the UNCHANGED rollout script with deterministic GPU kernels switched on
# from outside (a first-tick mismatch with empty history was seen on an H200 = cross-process kernel nondeterminism, not a leak)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
R="$PY scripts/det_run.py scripts/causal_rollout.py --ckpt $T/rand.pt --split dev --drive_list $T/one.txt --seeds 0"
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
msgs = []
for name, x in (("cam", cam), ("salt", salt), ("both", both)):
    pre = (c["lat"][:30].float() - x["lat"][:30].float()).abs().max().item(); post = (c["lat"][30:].float() - x["lat"][30:].float()).abs().max().item()
    ok5 = ok5 and pre == 0.0 and post > 1e-3; msgs.append(f"{name}: prefix {pre:.3g} (=0) suffix {post:.3g} (>0)")
# the camera arms really fed the donor's cameras from tick 30 (and only from 30); the salt arm did not
ok5 = ok5 and cam["other"] not in ("", "${DRV}") and torch.equal(cam["src_ticks"][:30], c["src_ticks"][:30]) and torch.equal(both["src_ticks"], cam["src_ticks"]) \
      and torch.equal(salt["src_ticks"], c["src_ticks"]) and cam["salt_from"] == -1 and salt["salt_from"] == 30 and both["salt_from"] == 30
# recorded per-tick provenance: whose cameras were fed and which private draw was used
D = cam["other"]
prov = lambda x, cams_from, salt_from: x["src_scene"] == ["${DRV}"] * cams_from + [D] * (60 - cams_from) and \
       [k.endswith("|salt") for k in x["noise_key"]] == [False] * salt_from + [True] * (60 - salt_from)
ok5 = ok5 and prov(c, 60, 60) and prov(cam, 30, 60) and prov(salt, 60, 30) and prov(both, 30, 30)
msgs.append("provenance ctrl/cam/salt/both %s %s %s %s" % (prov(c, 60, 60), prov(cam, 30, 60), prov(salt, 60, 30), prov(both, 30, 30)))
# the combined arm depends on BOTH interventions: it differs from cameras-only and from salt-only after tick 30
d_bc = (both["lat"][30:].float() - cam["lat"][30:].float()).abs().max().item(); d_bs = (both["lat"][30:].float() - salt["lat"][30:].float()).abs().max().item()
ok5 = ok5 and d_bc > 1e-3 and d_bs > 1e-3
print(("PASS" if ok5 else "FAIL"), "T5 60 aligned ticks, donor %s;" % cam["other"], "; ".join(msgs), "; both vs cam %.3g, both vs salt %.3g (>0)" % (d_bc, d_bs))
rs, fr, ct = f("t6_reset"), f("t6_fresh"), f("t6_ctrl")
s = (rs["lat"][20:].float() - fr["lat"].float()).abs().max().item(); dc = (ct["lat"][20:25].float() - rs["lat"][20:25].float()).abs().max().item()
ok6 = len(fr["lat"]) == 20 and s == 0.0 and dc > 1e-3 and bool(rs["restart"][20]) and bool(fr["restart"][0])
print(("PASS" if ok6 else "FAIL"), "T6 reset suffix vs fresh start: %.3g (=0); unreset control differs at ticks 20-24: %.3g (>0); restart flags %s %s" % (s, dc, bool(rs["restart"][20]), bool(fr["restart"][0])))
sys.exit(0 if ok5 and ok6 else 1)
P
echo "== T7: the real trainer, 2 updates (--timing), frozen guard must pass"
$PY scripts/train_causal.py --init $A --out $T/timing_t7 --K 4 --rho 0.5 --steps 100000 --max_steps_this_job 2 --timing > $T/t7.log 2>&1; r7=$?
grep -q "frozen check after first update" $T/t7.log && [ $r7 = 0 ] && echo "PASS T7 trainer frozen guard: $(grep 'frozen check' $T/t7.log)" || { echo "FAIL T7 (exit $r7)"; tail -5 $T/t7.log; FAILS=$((FAILS+1)); }
echo "CAUSAL_TESTS_SH FAILS=$FAILS dir=$T"
exit $FAILS
