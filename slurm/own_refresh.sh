#!/bin/bash
# Live own cache (council/drift-s2): redraw the 48-drive own-history pool from one checkpoint, then mark it complete.
#   bash slurm/own_refresh.sh runs/longC2/ckpt_10000.pt data/longC2/own_10000
# 8 rollout jobs (4 history-noise groups x 2 shards; owner cap 16 jobs, seed 0, same drive/strength assignment as own_4k / own_8k) and a
# CPU job that checks 48 files, all from this checkpoint, and writes <out>/DONE ({"snapshot": ckpt}). Called by
# scripts/train_temporal.py --refresh_every from inside the training job; run from ~/mini-reality-view.
set -e
cd $HOME/mini-reality-view
CK=$1; OUT=$2; mkdir -p $OUT
G="--gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=4 --mem=48G --time=2:00:00"
N=$(basename $OUT); ids=""
for s in 0 0.1 0.3 0.5; do for sh in 0 1; do
  ids="$ids:$(sbatch --parsable $G -J $N-s$s-$sh slurm/py.sbatch scripts/temporal_rollout.py --ckpt $CK --method hist --s $s --split train --drive_list tmp/longC_pool_s$s.txt --seeds 0 --shard $sh/2 --out $OUT)"
done; done
d=$(sbatch --parsable -d afterok$ids --cpus-per-task=1 --mem=8G --time=0:20:00 -J $N-done -o logs/%x-%j.out --wrap "cd $HOME/mini-reality-view && PYTHONPATH=. .venv/bin/python -c \"
import glob, json, torch, sys
f = sorted(glob.glob('$OUT/*.pt')); snaps = {torch.load(x)['snapshot'] for x in f}
pool = set(open('tmp/longC_pool48.txt').read().split())
ok = len(f) == 48 and {x.split('/')[-1].rsplit('_s', 1)[0] for x in f} == pool and snaps == {'$CK'}
print('files', len(f), 'snapshots', snaps, 'ok', ok)
if not ok: sys.exit(1)
json.dump({'snapshot': '$CK'}, open('$OUT/DONE.tmp', 'w')); import os; os.replace('$OUT/DONE.tmp', '$OUT/DONE')
\"")
echo "refresh $N rollouts$ids done-writer $d"
