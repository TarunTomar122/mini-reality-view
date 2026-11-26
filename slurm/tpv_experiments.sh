#!/bin/bash
# Experiment T (council/tpv-s1-fable.md): three 10k-step arms on third-person game keys, real camera tokens in.
#   E1: init from the real-key renderer (sd_run1 40k: adapter + LoRA, style rows 2 -> 3), all drives
#   E2: plain SD 1.5 + fresh adapter, all drives
#   E3: like E1 but a fixed quarter of the training drives
# All: style 2, bs 16 (one set), seed 0 (identical draws where the data match), no phone copies, hole weight 0.
# usage (from ~/mini-reality-view): bash slurm/tpv_experiments.sh <dependency job id>
cd $HOME/mini-reality-view
DEP=${1:+--dependency=afterok:$1}
envs/sim/bin/python -c "
import pickle, random
from nuscenes.utils.splits import train
n = [x for x in pickle.load(open('data/scn_8k/dataset_summary.pkl','rb')) ]
sc = sorted({s for s in train if any(s in x for x in n)})
random.Random(0).shuffle(sc)
open('tmp/tpv_quarter.txt','w').write('\n'.join(sorted(sc[:len(sc)//4])))
print('quarter', len(sc)//4, 'of', len(sc))"
G="--gres=gpu:h200_1g.18gb:1 --time=12:00:00 --mem=64G"
C="--set data/lat512_tpv:data/enc_full:2:16 --n_styles 3 --seed 0 --steps 10000 --save_at 2500,5000,7500,10000 --render_scenes 0 --hole_w 0"
E1=$(sbatch --parsable -J tpvE1 $G $DEP slurm/py.sbatch scripts/train_sd.py $C --init runs/sd_run1/ckpt.pt --out runs/tpvE1)
E2=$(sbatch --parsable -J tpvE2 $G $DEP slurm/py.sbatch scripts/train_sd.py $C --out runs/tpvE2)
E3=$(sbatch --parsable -J tpvE3 $G $DEP slurm/py.sbatch scripts/train_sd.py $C --init runs/sd_run1/ckpt.pt --scene_list tmp/tpv_quarter.txt --out runs/tpvE3)
J=""
for e in E1 E2 E3; do
  id=$(eval echo \$$e)
  for s in 5000 10000; do
    extra=""; [ "$e$s" = "E110000" ] && extra="--vae"
    j=$(sbatch --parsable -J tj-$e-$s --dependency=afterok:$id slurm/tpv_judge.sh tpv${e}_${s} runs/tpv$e/ckpt_$s.pt $extra); J="$J $j"
  done
done
echo "E1 $E1 E2 $E2 E3 $E3 judges$J"
