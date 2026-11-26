#!/bin/bash
# Sim verdict (council/PLAN-sim.md): B passed at equal real steps, C failed -> scale B: ~8,000 replay pairs, then a
# new main run = 5k sim-only steps, then 40k real steps with sd_run1's settings (judged vs sd_run1 at 40k = equal real
# steps, and vs sd_run1 continued to 45k = equal updates).
# usage (from ~/mini-reality-view): bash slurm/sim8k_chain.sh
cd $HOME/mini-reality-view
CV=$(sbatch --parsable -J scn-8k --time=4:00:00 --mem=64G --cpus-per-task=4 --output=logs/%x-%j.out \
     --wrap "envs/sim/bin/python scripts/scn_convert_subset.py data/scn_8k 600 3 2>&1 | grep --line-buffered -vE 'pkg_resources|from pkg'")
G="--gres=gpu:nvidia_geforce_rtx_2080_ti:1"
R=""
for i in 0 1 2 3 4 5 6 7; do
  r=$(sbatch --parsable $G -J sim8k-render-$i --time=12:00:00 --dependency=afterok:$CV slurm/sim_render.sh $i/8 "" data/scn_8k 600)
  R="$R:$r"
done
E0=$(sbatch --parsable -J sim8k-enc-0 --gres=gpu:1 --time=6:00:00 --dependency=afterany$R slurm/py.sbatch scripts/sim_encode.py --shard 0/2)
E1=$(sbatch --parsable -J sim8k-enc-1 --gres=gpu:1 --time=6:00:00 --dependency=afterany$R slurm/py.sbatch scripts/sim_encode.py --shard 1/2)
L=$(sbatch --parsable -J sim8k-lat --gres=gpu:1 --time=4:00:00 --dependency=afterany$R slurm/py.sbatch scripts/encode_latents.py --targets data/sim/keys --out data/lat512_sim)
Gh="--gres=gpu:h200_1g.18gb:1 --time=2-00:00:00 --mem=120G"
S1=$(sbatch --parsable -J sdB-sim $Gh --dependency=afterok:$E0:$E1:$L slurm/py.sbatch scripts/train_sd.py \
     --set data/lat512_sim:data/enc_sim:2:16:data/enc_sim_aug --n_styles 3 --steps 5000 --render_scenes 0 --out runs/sdB_sim)
S2=$(sbatch --parsable -J sdB-real $Gh --dependency=afterok:$S1 slurm/py.sbatch scripts/train_sd.py \
     --keys data/lat512_full_low --keys2 data/lat512_splat_tv_low_trust --mix 0.3 --steps 40000 --bs 16 --render_scenes 8 \
     --tokens_aug data/enc_aug_0 --hole_w 0 --cover_min 0.8,0.3 --n_styles 3 --init runs/sdB_sim/ckpt.pt \
     --save_at 10000,20000,30000,40000 --out runs/sdB_real)
echo "convert $CV render$R enc $E0 $E1 lat $L sim-phase $S1 real-phase $S2"
# equal-update reference: sd_run1 continued from its 40k state to 45k in a copy of its folder
CP=$(sbatch --parsable -J sd45-copy --time=0:30:00 --mem=4G --dependency=afterok:3751192 --output=logs/%x-%j.out --wrap "cp -r runs/sd_run1 runs/sd_run1_45k")
S45=$(sbatch --parsable -J sd-run1-45k $Gh --dependency=afterok:$CP slurm/py.sbatch scripts/train_sd.py \
     --keys data/lat512_full_low --keys2 data/lat512_splat_tv_low_trust --mix 0.3 --steps 45000 --bs 16 --render_scenes 8 \
     --tokens_aug data/enc_aug_0 --hole_w 0 --cover_min 0.8,0.3 --out runs/sd_run1_45k)
echo "sd_run1 -> 45k: $CP $S45"
