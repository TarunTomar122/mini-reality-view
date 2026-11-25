#!/bin/bash
# Stream nuScenes v1.0 trainval (meta + camera + lidar blobs) straight into tar, 3 parts at a time.
# Safe to rerun: finished parts are listed in data/full_done.txt.
cd $HOME/mini-reality-view/data
mkdir -p nuscenes_full
touch full_done.txt
rm -f v1.0-trainval*.tgz
get() {
  f=$1; BASE=https://d36yt3mvayqw5m.cloudfront.net/public/v1.0
  grep -qx "$f" full_done.txt && return
  for try in 1 2 3 4 5; do
    echo "$(date +%H:%M) get $f (try $try)"
    curl -sSfL --retry 5 "$BASE/$f" | tar -xz -C nuscenes_full && { echo "$f" >> full_done.txt; echo "$(date +%H:%M) done $f"; return; }
    sleep 30
  done
  echo "$(date +%H:%M) FAILED $f"
}
export -f get
L=""
for i in 01 02 03 04 05 06 07 08 09 10; do L="$L v1.0-trainval${i}_blobs_camera.tgz v1.0-trainval${i}_blobs_lidar.tgz"; done
echo v1.0-trainval_meta.tgz $L | tr ' ' '\n' | grep . | xargs -P 3 -I{} bash -c 'get {}'
echo FULL_DONE
