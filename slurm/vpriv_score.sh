#!/bin/bash
# Signed compares for one vpriv check (CPU): vs A@20k and vs own cold (off at 0.95), fixed recipe (A-derived k=3 count lists,
# floor 0, cold_ref B); both outputs must contain a complete panel (6 drives, 12 runs) or the job FAILS (Astra launch review 2);
# then the A1 decision at 8k / 12k / 16k (scripts/a1_decide.py: pass flag, two consecutive safety misses -> stop vpriv);
# leave-one-drive-out last (report only, cannot mask a failure above).
#   sbatch -d afterok:<metrics> --cpus-per-task=1 --mem=4G --time=0:30:00 -J pcmp<k>k -o logs2/%x-%j.out slurm/vpriv_score.sh <k>
cd $HOME/mini-reality-view; set -o pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD
k=$1; PY=.venv/bin/python; M=runs/troll/m_P${k}k_dev.json
[ -s $M ] || { echo "SCORE_FAIL missing $M"; exit 1; }
for x in "A20k_indep_dev A20k" "Aoff_rho0.95_dev Aoff0.95"; do set -- $x
  o=runs/s_lock/cmp_P${k}k__$1.json
  $PY scripts/temporal_compare.py --compare $M runs/troll/m_$1.json --count_list runs/count/count_$2_dev_k3.json --floor 0 \
      --cold_ref B > $o || { echo "SCORE_FAIL compare vs $1"; exit 1; }
  $PY -c "import json,sys;d=json.loads(open('$o').read().split('TCOMPARE ',1)[1]);sys.exit(0 if d['drives']==6 and d['runs']==12 and not d['unpaired_drives'] else 1)" \
      || { echo "SCORE_FAIL incomplete panel $o"; exit 1; }
  echo "compare vs $1 ok"
done
case $k in 8|12|16) $PY scripts/a1_decide.py $k || { echo "SCORE_FAIL A1 decision"; exit 1; };; esac
$PY scripts/lodo.py $M runs/troll/m_A20k_indep_dev.json --out runs/s_lock/lodo_P${k}k_vs_A20k.json || echo "LODO failed (report only)"
echo "SCORE_DONE ${k}k"
