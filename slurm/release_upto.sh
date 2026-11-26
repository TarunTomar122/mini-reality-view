#!/bin/bash
# Release held Mini Reality View jobs in priority order while (running + pending-not-held) < CAP (owner: max 12).
# usage: bash slurm/release_upto.sh 12 <jobid> <jobid> ...   (job ids in priority order; non-held / finished ids are skipped)
CAP=$1; shift
active() { squeue -u $USER -h -o "%j %r" | grep -E "enc|lit|troll|tmet|bev|tpv|cm|tmp" | grep -vc JobHeldUser; }
for j in "$@"; do
  [ "$(active)" -ge "$CAP" ] && break
  squeue -j $j -h -o "%r" 2>/dev/null | grep -q JobHeldUser && scontrol release $j && echo released $j
done
echo ACTIVE $(active)
