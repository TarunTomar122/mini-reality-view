#!/bin/bash
# All pre-training tests of council/AGREED-causal-r1.md (T1 T3 T4 T7 in causal_tests.py; T2 T5 T6 here). One 2080 Ti.
#   sbatch --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=4 --mem=48G --time=2:00:00 -J ctests2 -o logs/%x-%j.out slurm/causal_tests.sh
cd $HOME/mini-reality-view
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 TMPDIR=$PWD/tmp
PY=.venv/bin/python; A=runs/longA/ckpt_20000.pt; T=runs/ctest; rm -rf $T; mkdir -p $T
echo "== causal_tests.py (T1 T3 T4 T7)"; $PY scripts/causal_tests.py --init $A --K 4 --F 12; r1=$?
DRV=$($PY -c "import json;print(json.load(open('data/temporal_lists.json'))['dev'][0])"); echo $DRV > $T/one.txt
echo "== T2: --off at rho 1.0 vs signed A@20k ($DRV, seed 0)"
$PY scripts/causal_rollout.py --ckpt $A --mode off --rho 1.0 --split dev --drive_list $T/one.txt --seeds 0 --out $T/off_rho1 > /dev/null
$PY - <<P
import torch,sys
a=torch.load("$T/off_rho1/${DRV}_s0.pt"); b=torch.load("runs/troll/A20k_indep_dev/${DRV}_s0.pt")
n=min(len(a["lat"]),len(b["lat"])); d=(a["lat"][:n].float()-b["lat"][:n].float()).abs().max().item()
same=bool((a["ok"][:n]==b["ok"][:n]).all() and (a["hard"][:n]==b["hard"][:n]).all() and (a["restart"][:n]==b["restart"][:n]).all())
print(("PASS" if d<1e-2 and same and len(a["lat"])==len(b["lat"]) else "FAIL"), "T2 off@rho1 == signed A: max|diff| %.3g, flags equal %s, ticks %d vs %d"%(d,same,len(a["lat"]),len(b["lat"])))
P
echo "== random-weight causal checkpoint for T5 / T6"
$PY - <<P
import torch
from diffusers import UNet2DConditionModel
from peft import LoraConfig
from mrv.sd import LORA_TARGETS
from mrv.causal_video import add_causal_temporal
ck=torch.load("$A",map_location="cpu"); r=ck["args"]["rank"]
u=UNet2DConditionModel.from_pretrained("sd-legacy/stable-diffusion-v1-5",subfolder="unet"); u.add_adapter(LoraConfig(r=r,lora_alpha=r,target_modules=LORA_TARGETS))
TW=add_causal_temporal(u,4); g=torch.Generator().manual_seed(7)
for w in TW:
    for n,q in w.temporal.named_parameters():
        if not (n.startswith("norm") or n.startswith("ln")): q.data.add_(torch.randn(q.shape,generator=g)*0.05)
torch.save({**ck,"temporal":[w.temporal.state_dict() for w in TW],"K":4,"rho":0.5,"step":0},"$T/rand.pt"); print("saved")
P
echo "== T5: cameras + private draws of ticks 30..59 replaced -> ticks 0..29 identical"
$PY scripts/causal_rollout.py --ckpt $T/rand.pt --split dev --drive_list $T/one.txt --seeds 0 --max_ticks 60 --out $T/t5a > /dev/null
$PY scripts/causal_rollout.py --ckpt $T/rand.pt --split dev --drive_list $T/one.txt --seeds 0 --max_ticks 60 --swap --swap_at 30 --salt_from 30 --out $T/t5b > /dev/null
echo "== T6: --reset_at 20 suffix == fresh run from tick 20"
$PY scripts/causal_rollout.py --ckpt $T/rand.pt --split dev --drive_list $T/one.txt --seeds 0 --max_ticks 40 --reset_at 20 --out $T/t6a > /dev/null
$PY scripts/causal_rollout.py --ckpt $T/rand.pt --split dev --drive_list $T/one.txt --seeds 0 --max_ticks 40 --start_tick 20 --out $T/t6b > /dev/null
$PY - <<P
import torch
f=lambda d: torch.load(f"$T/{d}/${DRV}_s0.pt")
a,b=f("t5a"),f("t5b"); hard=a["hard"]
pre=(a["lat"][:30].float()-b["lat"][:30].float()).abs().max().item(); post=(a["lat"][30:].float()-b["lat"][30:].float()).abs().max().item()
print(("PASS" if pre==0.0 and post>1e-3 else "FAIL"), "T5 prefix max|diff| %.3g (must be 0), suffix %.3g (must be >0)"%(pre,post))
c,d=f("t6a"),f("t6b"); s=(c["lat"][20:].float()-d["lat"].float()).abs().max().item()
print(("PASS" if s==0.0 and len(d["lat"])==20 else "FAIL"), "T6 reset_at 20 suffix vs fresh start at 20: max|diff| %.3g (must be 0), len %d"%(s,len(d["lat"])))
P
echo "CAUSAL_TESTS_SH_DONE r1=$r1"
