#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out
# Wait for sd_run1's next checkpoint, then stop it and resume from that checkpoint with phone-style token copies.
# usage: sbatch slurm/sd_restart_at.sh <old_jobid>
OLD=$1
cd $HOME/mini-reality-view
T0=$(stat -c %Y runs/sd_run1/ckpt.pt)
echo "waiting for a newer checkpoint than $(date -d @$T0)"
while [ "$(stat -c %Y runs/sd_run1/ckpt.pt)" = "$T0" ]; do sleep 30; done
sleep 60  # let the save finish
python3 -c "import torch" 2>/dev/null; .venv/bin/python -c "import torch; print('ckpt step', torch.load('runs/sd_run1/ckpt.pt', map_location='cpu')['step'])"
scancel $OLD; sleep 20
NEW=$(sbatch --parsable -J sd-run1 --gres=gpu:h200_1g.18gb:1 --time=2-00:00:00 --mem=120G slurm/py.sbatch scripts/train_sd.py \
  --keys data/lat512_full_low --keys2 data/lat512_splat_tv_low_trust --mix 0.3 --steps 40000 --bs 16 --render_scenes 8 \
  --tokens_aug data/enc_aug_0 --out runs/sd_run1)
echo "stopped $OLD, resumed as $NEW at $(date)"
