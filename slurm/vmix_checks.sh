#!/bin/bash
# CPU-only watcher for the vmix checks (A5 = A4 gate under A1: 2k / 4k informational, 8k / 12k / 16k gated). Copy of
# slurm/vpriv_checks.sh (unchanged) adapted to vmix. For every saved runs/vmix/ckpt_<k>000.pt it submits the dev panel: 6 stream
# rollout shards on 2080 Tis (scripts/clean_rollout.py --mode stream, rho from the checkpoint = the gate rho 1.0, seeds 0 / 1),
# signed metrics with --cold runs/troll/A20k_indep_dev (at rho 1.0 the memory-off cold IS signed A@20k), then
# slurm/vmix_score.sh (signed compare vs A@20k with panel checks, A1 decision at 8k+ with the A4 ghost / road pass conditions,
# the 12k stage-2 resume, LODO), plus at 8k ONLY the non-gating own-history probe (scripts/clean_probe.py --own). No GPU waits
# for a checkpoint.
# Admission (Astra 01:30 (d) + watcher review 2): ONE job per pass, then the next pass rescans from 8k, so a ready gate always
# goes first; slots = min(16 - active MRV jobs (held not counted), 20 - all account jobs); 2k / 4k keep 9 slots in reserve
# until every gate (8k / 12k / 16k) is queued or skipped. Each job's unique name is recorded BEFORE sbatch (intent) and its id
# after; on restart an intent is reconciled by name (exactly one job -> adopt; none in both squeue and sacct -> resubmit under
# the same name; anything else -> stop loudly). Dependencies on old ids are checked: running / pending -> kept, COMPLETED ->
# dropped, failed / unknown / never satisfiable -> stop loudly. A failed 8k / 12k score job stops the watcher with its id.
# If training dies, saved checkpoints are still drained before the watcher reports the missing one.
#   sbatch --cpus-per-task=1 --mem=8G --time=9:00:00 -J vmix-watch -o logs2/%x-%j.out slurm/vmix_checks.sh
cd $HOME/mini-reality-view
G="--gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=4 --mem=48G --exclude=damnii10"  # damnii10: broken GPU (Neon 01:30)
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD
mkdir -p $PWD/runs/tmp && export TMPDIR=$PWD/runs/tmp
S=runs/vmix_gate/watch; mkdir -p $S || exit 1
die() { echo "WATCH_FAIL $*" >&2; exit 1; }
rec() { echo "$2" > $S/$1 && [ "$(cat $S/$1)" = "$2" ] || die "cannot record $1"; }
free() {  # free slots (min of both limits); returns 1 if squeue failed
  local q m a; q=$(squeue -u $USER -h -o "%R %Z") || return 1
  m=$(echo "$q" | grep mini-reality-view | grep -vc JobHeldUser); a=$(echo "$q" | grep -c .)
  local f=$(( 16 - m )); [ $(( 20 - a )) -lt $f ] && f=$(( 20 - a )); echo $f; }
gates_done() { [ -e $S/8.done ] && [ -e $S/12.done ] && [ -e $S/16.done ]; }
take() {  # is a slot free for job class of $k? (the 2k / 4k reserve applies only while training runs and gates remain)
  local f need=1; case $k in 2|4) gates_done || ! running || need=10;; esac; f=$(free) || return 1; [ "$f" -ge $need ]; }
jobstate() {  # active | done | failed | unknown
  local s; s=$(squeue -h -j $1 -o "%T %r" 2>/dev/null)
  if [ -n "$s" ]; then [[ "$s" == *DependencyNeverSatisfied* ]] && echo failed || echo active; return; fi
  s=$(sacct -n -X -P -S now-3days -j $1 -o State 2>/dev/null | head -1)
  case "$s" in COMPLETED) echo done;; PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING|COMPLETING) echo active;; "") echo unknown;; *) echo failed;; esac; }
deps() {  # deps <id...>: prints ":id" for still-active ids; COMPLETED ids are dropped; failed / unknown -> stop
  local j st; for j in "$@"; do st=$(jobstate $j)
    case $st in active) printf ":%s" $j;; done) ;; *) die "dependency $j is $st";; esac; done; }
byname() {  # byname <name> [<start time>]: ids carrying exactly this name, from squeue AND sacct (both must succeed); sacct
           # searches from the persisted intent time (default: 3 days back)
  local q1 q2; q1=$(squeue -u $USER -h -n $1 -o %i) || return 1; q2=$(sacct -u $USER -X -n -P -S ${2:-now-3days} --name=$1 -o JobID) || return 1
  printf "%s\n%s\n" "$q1" "$q2" | grep -E '^[0-9]+$' | sort -u; }
submit() {  # submit <state key> <base name> <sbatch args...>: intent first, then one sbatch, then the id
  local key=$1 base=$2 n j q; shift 2
  if [ -s $S/$key.name ]; then  # an earlier attempt: reconcile before anything else
    n=$(cat $S/$key.name); [ -s $S/$key.time ] || die "intent $n has no recorded time: reconcile by hand"
    q=$(byname $n $(cat $S/$key.time)) || die "lookup failed for intent $n"
    case $(echo "$q" | grep -c .) in 1) rec $key $q; echo "adopted $n = $q"; return;; 0) ;; *) die "intent $n matches several jobs: $q";; esac
  else n="$base.$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n')"; rec $key.time $(date -d '-5 minutes' +%Y-%m-%dT%H:%M:%S); rec $key.name $n; fi
  j=$(sbatch --parsable -J $n "$@")
  if [[ "$j" =~ ^[0-9]+$ ]]; then rec $key $j; echo "$(date -u +%H:%M) $key: $n = $j"; return; fi
  sleep 30; q=$(byname $n $(cat $S/$key.time)) || die "ambiguous sbatch for $n (lookup failed)"
  [ "$(echo "$q" | grep -c .)" = 1 ] && { rec $key $q; echo "adopted $n = $q"; return; }
  die "ambiguous sbatch for $n (matches: ${q:-none})"; }
running() { local q; q=$(squeue -u $USER -h -n vmix -o %i) || return 0; [ -n "$q" ]; }  # queue error: assume running
held() { tail -n 1 runs/vmix/log.jsonl 2>/dev/null | grep -q '^{"hold"'; }  # the trainer's last act was an A5 hold

reconcile() {  # outstanding intents (name recorded, no id) are resolved BEFORE any admission check
  local f key n q
  for f in $S/*.name; do [ -e "$f" ] || continue; key=${f%.name}; [ -s $key ] && continue
    n=$(cat $f); [ -s $key.time ] || die "intent $n has no recorded time: reconcile by hand"
    q=$(byname $n $(cat $key.time)) || die "lookup failed for intent $n"
    case $(echo "$q" | grep -c .) in 1) rec ${key#$S/} $q; echo "adopted $n = $q";; 0) ;; *) die "intent $n matches several jobs: $q";; esac
  done; }
check_chain() {  # any failed job in a gated chain (shards, metrics, score) stops the watcher with its id; unknown state is
                 # tolerated for at most 10 passes per job
  local f st u
  for f in $S/8.sh* $S/8.m $S/8.c $S/12.sh* $S/12.m $S/12.c $S/16.sh* $S/16.m $S/16.c; do
    [ -e "$f" ] || continue; case $f in *.name|*.unk|*.time) continue;; esac
    st=$(jobstate $(cat $f))
    case $st in failed) die "gated chain job $(cat $f) (${f#$S/}) failed";;
      unknown) u=$(( $(cat $f.unk 2>/dev/null || echo 0) + 1 )); echo $u > $f.unk
        [ $u -ge 10 ] && die "gated chain job $(cat $f) (${f#$S/}) state unknown for 10 passes";;
      *) rm -f $f.unk;; esac
  done; }
pass() {  # admits at most ONE job; returns 0 = admitted (rescan now), 1 = nothing admissible now (sleep), 2 = all done
  local k ck out sh p dep missing=""
  alldone=1
  reconcile; check_chain
  # early continuation (Fable 01:30): 12k saved and the 8k decision safe -> the 12k decision cannot stop the run -> resume now
  # requires the CURRENT 8k score job to have COMPLETED (not just any a1_8k.json); run INLINE by the watcher (no helper job, so
  # no slot race with the watcher's own passes); one attempt only; any failure is left for manual reconciliation and the
  # regular 12k path (Astra early reviews 1-2)
  if [ -f runs/vmix/ckpt_12000.pt ] && [ -s $S/8.c ] && [ "$(jobstate $(cat $S/8.c))" = done ] && [ -s runs/vmix_gate/a1_8k.json ] \
     && [ ! -e $S/12.early ] && [ ! -e runs/vmix_gate/CONTINUED ]; then
    if [ "$(.venv/bin/python -c "import json;print(json.load(open('runs/vmix_gate/a1_8k.json')).get('safe') is True)")" = True ]; then
      f=$(free) || return 1; [ "$f" -ge 1 ] || return 1
      rec 12.early started
      bash slurm/vmix_score.sh early $(cat $S/8.c) > $S/early.log 2>&1; rc=$?
      rec 12.early "rc=$rc"; echo "$(date -u +%H:%M) early continuation rc=$rc: $(tail -1 $S/early.log)"; return 0
    fi
  fi
  for k in 8 12 16 2 4; do
    [ -e $S/$k.done ] && continue
    alldone=0; ck=runs/vmix/ckpt_${k}000.pt
    if [ ! -f $ck ]; then
      if [ $k = 16 ]; then
        [ -e runs/vmix_gate/A1_STOPPED ] && { echo "A1 stop at 12k: no 16k check"; rec 16.done skipped; return 0; }
        [ -e runs/vmix_gate/CONTINUED ] && ! running && missing="$missing $k"
      elif [ -e runs/vmix/args.json ] && ! running && ! held; then missing="$missing $k"
      elif held && ! running && [ ! -e $S/held.note ]; then  # A5 hold at the cache boundary: not a failure; reported once
        rec held.note "$(date -u +%FT%TZ)"; echo "vmix HELD at the cache boundary (last row of runs/vmix/log.jsonl): resume by hand with --resume once a passing cache record verifies"
      fi
      continue
    fi
    out=runs/troll/X${k}k_dev; mkdir -p $out; lfs setstripe -i 0 -c 1 $out 2>/dev/null
    for sh in 0 1 2 3 4 5; do
      [ -s $S/$k.sh$sh ] && continue
      take || return 1
      submit $k.sh$sh xchk${k}k-$sh $G --time=3:00:00 -o logs2/%x-%j.out slurm/py.sbatch scripts/clean_rollout.py \
          --ckpt $ck --mode stream --split dev --shard $sh/6 --out $out; return 0
    done
    if [ ! -s $S/$k.m ]; then
      take || return 1
      dep=$(deps $(for sh in 0 1 2 3 4 5; do cat $S/$k.sh$sh; done)) || exit 1
      submit $k.m tmet-X${k}k ${dep:+-d afterok$dep} $G --time=1:00:00 -o logs2/%x-%j.out slurm/py.sbatch scripts/temporal_metrics.py \
          --roll $out --cold runs/troll/A20k_indep_dev --out runs/troll/m_X${k}k_dev.json; return 0
    fi
    if [ ! -s $S/$k.c ]; then
      case $k in 12) p=8;; 16) p=12;; *) p="";; esac  # each gated decision also waits for the previous gated score job
      if [ -n "$p" ] && [ ! -s $S/$p.c ]; then continue; fi
      take || return 1
      dep=$(deps $(cat $S/$k.m) ${p:+$(cat $S/$p.c)}) || exit 1
      submit $k.c xcmp${k}k ${dep:+-d afterok$dep} --cpus-per-task=1 --mem=4G --time=0:30:00 -o logs2/%x-%j.out slurm/vmix_score.sh $k
      return 0
    fi
    if [ $k = 8 ] && [ ! -s $S/$k.p ]; then  # A5: the 8k own-history probe (trained reader on its own generated prefixes), non-gating
      take || return 1
      submit $k.p xprobe${k}k $G --time=1:30:00 -o logs2/%x-%j.out slurm/py.sbatch scripts/clean_probe.py --own --ckpts $ck \
          --out runs/probe/vmix_own_${k}000.json
      return 0
    fi
    rec $k.done queued; return 0
  done
  if [ $alldone = 1 ]; then  # keep monitoring until every gated chain job is terminal (check_chain dies on failures)
    for f in $S/8.sh* $S/8.m $S/8.c $S/12.sh* $S/12.m $S/12.c $S/16.sh* $S/16.m $S/16.c; do
      [ -e "$f" ] || continue; case $f in *.name|*.unk|*.time) continue;; esac
      [ "$(jobstate $(cat $f))" = done ] || return 1
    done
    return 2
  fi
  # everything saved is queued; a checkpoint that can no longer appear ends the watcher (after draining the saved work)
  [ -n "$missing" ] && die "vmix not running and checkpoint(s)$missing missing"
  return 1
}
while :; do
  pass; r=$?
  [ $r = 2 ] && { echo "VMIX_CHECKS all queued"; exit 0; }
  [ $r = 1 ] && sleep 60
done
