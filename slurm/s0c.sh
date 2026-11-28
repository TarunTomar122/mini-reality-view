#!/bin/bash
# PLAN-video S0c on 2080 Tis (council/PLAN-video.md, SIGNATURE ROUND RESULT; scripts/s0c_readers.py). Submission helper only:
# run by hand on the login node from ~/mini-reality-view. Run scripts/guide_preview.py on the pinned drive FIRST (Fable 06:30).
#   bash slurm/s0c.sh [ACTIVE=16] [DRIVES=train48,dev]
# 16 jobs: fit depth scale (1, train48) -> camera-depth guides (4 shards) + LiDAR-ceiling guides (4 shards); key-mask / view
# cache (2 shards, no dependency) -> readers guide / lidar / tokens / hybrid (1 each) -> report (afterany: an unfinished
# reader is reported as UNRESOLVED). Jobs beyond ACTIVE are submitted held (-H; owner cap 16; release with scontrol release).
# Budget <= 5 aggregate 2080 Ti-h incl. caching and scoring: every stage appends its wall time to runs/s0c/gpu_time.jsonl and
# the report sums it (gpu_hours_aggregate, over_budget).
cd $HOME/mini-reality-view
ACTIVE=${1:-16}; DRIVES=${2:-train48,dev}
G="--gres=gpu:nvidia_geforce_rtx_2080_ti:1 --exclude=damnii10 --mem=48G"
TL=runs/s0c/gpu_time.jsonl
mkdir -p logs2 runs/s0c data/guides_v1 data/guides_v1_lidar data/s0c_cache
n=0
# sub sets JID (no command substitution around it: the held-job counter n must persist in this shell)
sub() { local H=""; [ $n -ge $ACTIVE ] && H="-H"; n=$((n + 1)); JID=$(sbatch --parsable $H "$@") || { echo "sbatch failed: $*" >&2; exit 1; }; }
# offline-cache check (jobs run with HF_HUB_OFFLINE=1): depth checkpoints + DINOv2-small must already be in hf/
HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 .venv/bin/python -c "
from transformers import AutoModelForDepthEstimation, AutoModel
AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf', local_files_only=True)
AutoModel.from_pretrained('facebook/dinov2-small', local_files_only=True); print('hf cache ok')" || { echo "missing HF weights in hf/ (download on the login node first)"; exit 1; }

sub $G --cpus-per-task=4 --time=1:00:00 -J s0c-fit -o logs2/%x-%j.out \
      slurm/py.sbatch scripts/fit_depth_scale.py --drives train48 --out data/guides_v1/depth_scale.json --time_log $TL
  FIT=$JID
echo "fit scale: $FIT"
GIDS=""; LIDS=""
for i in 0 1 2 3; do
  sub $G --cpus-per-task=6 --time=2:00:00 -d afterok:$FIT -J s0c-guide$i -o logs2/%x-%j.out \
      slurm/py.sbatch scripts/build_guides.py --drives $DRIVES --shard $i/4 --workers 5 --time_log $TL
  J=$JID
  GIDS="$GIDS:$J"
  sub $G --cpus-per-task=6 --time=2:00:00 -d afterok:$FIT -J s0c-lidar$i -o logs2/%x-%j.out \
      slurm/py.sbatch scripts/build_guides.py --drives $DRIVES --shard $i/4 --workers 5 --lidar_ceiling --time_log $TL
  J=$JID
  LIDS="$LIDS:$J"
done
echo "guides:$GIDS  lidar guides:$LIDS"
CIDS=""
for i in 0 1; do
  sub $G --cpus-per-task=4 --time=2:00:00 -J s0c-cache$i -o logs2/%x-%j.out \
      slurm/py.sbatch scripts/s0c_readers.py --stage cache --drives $DRIVES --shard $i/2
  J=$JID
  CIDS="$CIDS:$J"
done
echo "cache:$CIDS"
TIDS=""
for arm in guide tokens hybrid lidar; do
  DEP="afterok$GIDS$CIDS"; [ $arm = lidar ] && DEP="afterok$LIDS$GIDS$CIDS"
  sub $G --cpus-per-task=8 --time=3:00:00 -d $DEP -J s0c-$arm -o logs2/%x-%j.out \
      slurm/py.sbatch scripts/s0c_readers.py --stage train --arm $arm --updates 6000 --batch 16 --workers 6
  J=$JID
  TIDS="$TIDS:$J"; echo "reader $arm: $J"
done
sub $G --cpus-per-task=4 --time=1:00:00 -d afterany$TIDS -J s0c-report -o logs2/%x-%j.out \
    slurm/py.sbatch scripts/s0c_readers.py --stage report
  R=$JID
echo "report: $R -> runs/s0c/report.json, runs/s0c/sheet.jpg ($n jobs submitted)"
