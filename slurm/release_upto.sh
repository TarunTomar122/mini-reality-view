#!/bin/bash
# Owner's cap: at most CAP Mini Reality View jobs active at once (other slots belong to other projects).
# A job belongs to this project if its working directory is ~/mini-reality-view (independent of its name).
# Active = running + pending-but-not-held (pending jobs start on their own, so they count).
# usage: bash slurm/release_upto.sh <cap> [held job ids to release, in priority order]
#        bash slurm/release_upto.sh <cap> --all   (release held project jobs in submission order)
CAP=$1; shift
proj() { squeue -u $USER -h -o "%i %T %r %Z" | awk -v d="$HOME/mini-reality-view" '$4==d'; }
active() { proj | awk '$2=="RUNNING" || ($2=="PENDING" && $3!="JobHeldUser")' | wc -l; }
ids="$@"; [ "$1" = "--all" ] && ids=$(proj | awk '$3=="JobHeldUser"{print $1}' | sort -n)
for j in $ids; do
  [ "$(active)" -ge "$CAP" ] && break
  squeue -j $j -h -o "%r" 2>/dev/null | grep -q JobHeldUser && scontrol release $j && echo released $j
done
echo ACTIVE $(active) RUNNING $(proj | awk '$2=="RUNNING"' | wc -l)
