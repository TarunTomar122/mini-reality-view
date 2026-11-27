#!/bin/bash
# Long bet, arm C: one training stage to step $1, using the newest COMPLETE own-rollout cache (48 drives).
#   bash slurm/longC_stage.sh 20000   (run inside an sbatch job from ~/mini-reality-view)
cd $HOME/mini-reality-view
STEPS=$1; SA="5000,8000,10000,12000,15000,16000,20000,25000,30000"
CACHE=data/own_4k; CS=4000; SNAP=runs/tmpC/ckpt_4000.pt
for k in 8 12 16 20 24; do
  d=data/own_${k}k
  [ -d $d ] && [ $(ls $d | grep -c "pt$") -ge 48 ] && [ -z "$(ls $d | grep tmp)" ] && { CACHE=$d; CS=${k}000; SNAP=runs/longC/ckpt_${k}000.pt; }
done
echo "stage to $STEPS with cache $CACHE (step $CS, snapshot $SNAP)"
exec bash slurm/py.sbatch scripts/train_temporal.py --arm C --init runs/tmpB/ckpt_1000.pt --pool_list tmp/longC_pool48.txt \
  --cache $CACHE --cache_step $CS --cache_snapshot $SNAP --steps $STEPS --save_at $SA --out runs/longC
