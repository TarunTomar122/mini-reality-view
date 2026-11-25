#!/bin/bash
# Submit the three pilot arms of council/PLAN-sim.md (identical init, paired real rows: 11 lidar + 5 splat).
# usage: bash slurm/sim_arms.sh <dependency job ids joined by :>
DEP=$1
cd $HOME/mini-reality-view
G="--gres=gpu:h200_1g.18gb:1 --time=1-00:00:00 --mem=120G"
REAL="--set data/lat512_full_low:data/enc_full:0:11:data/enc_aug_0 --set data/lat512_splat_tv_low_trust:data/enc_full:1:5:data/enc_aug_0"
SIM="--set data/lat512_sim:data/enc_sim:2:2:data/enc_sim_aug"
TWIN="--set data/lat512_sim:data/enc_full:2:2:data/enc_aug_0"
COMMON="--n_styles 3 --seed 7 --render_scenes 0"
A=$(sbatch --parsable -J simA $G --dependency=afterok:$DEP slurm/py.sbatch scripts/train_sd.py $REAL $COMMON --steps 7000 --save_at 3000,5000,7000 --out runs/simA)
B1=$(sbatch --parsable -J simB1 $G --dependency=afterok:$DEP slurm/py.sbatch scripts/train_sd.py --set data/lat512_sim:data/enc_sim:2:16:data/enc_sim_aug $COMMON --steps 2000 --out runs/simB1)
B2=$(sbatch --parsable -J simB2 $G --dependency=afterok:$B1 slurm/py.sbatch scripts/train_sd.py $REAL $COMMON --init runs/simB1/ckpt.pt --steps 5000 --save_at 3000,5000 --out runs/simB)
C=$(sbatch --parsable -J simC $G --dependency=afterok:$DEP slurm/py.sbatch scripts/train_sd.py $REAL $SIM $TWIN $COMMON --steps 5000 --save_at 3000,5000 --out runs/simC)
echo "A $A  B1 $B1 -> B2 $B2  C $C"
