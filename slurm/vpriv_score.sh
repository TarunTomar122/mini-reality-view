#!/bin/bash
# Signed compares for one vpriv check (CPU): vs A@20k and vs own cold (off at 0.95), fixed recipe (A-derived k=3 count lists,
# floor 0, cold_ref B); both outputs must contain a complete panel (6 drives, 12 runs) or the job FAILS (Astra launch review 2);
# then the A1 decision at 8k / 12k / 16k (scripts/a1_decide.py), and at 12k the stage-2 resume to 16k unless A1 says stop;
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
if [ $k = 12 ]; then  # stage 2: resume 12k -> 16k only if the 12k A1 decision allows (training paused at 12k by design)
  G2=runs/vpriv_gate; exec 9>$G2/.lock; flock 9
  STOP=$($PY -c "import json;print(json.load(open('$G2/a1_12k.json'))['stop'])") || { echo "SCORE_FAIL reading a1_12k"; exit 1; }
  if [ "$STOP" = "True" ]; then echo "A1 STOP at 12k (two consecutive safety misses): vpriv is NOT continued"; touch $G2/A1_STOPPED
  elif [ -e $G2/CONTINUED ]; then echo "already continued ($(cat $G2/CONTINUED))"
  elif [ -e $G2/CONT_PENDING ]; then echo "SCORE_FAIL ambiguous continuation ($(cat $G2/CONT_PENDING)): reconcile by hand"; exit 1
  else
    Q=$(squeue -u $USER -h -n vpriv -o %i) || { echo "SCORE_FAIL squeue"; exit 1; }
    [ -z "$Q" ] || { echo "SCORE_FAIL vpriv job $Q still queued at the 12k decision"; exit 1; }
    STEP=$($PY -c "import torch;print(torch.load('runs/vpriv/ckpt.pt',map_location='cpu',weights_only=False)['step'])") || exit 1
    [ "$STEP" = 12000 ] || { echo "SCORE_FAIL runs/vpriv/ckpt.pt step $STEP != 12000"; exit 1; }
    PRIV=$(grep -o "private_clips=[12]" $G2/SUBMITTED | cut -d= -f2); [[ "$PRIV" =~ ^[12]$ ]] || { echo "SCORE_FAIL no private_clips in SUBMITTED"; exit 1; }
    echo "intent $(date -u +%FT%TZ) resume 12000->16000" > $G2/CONT_PENDING
    J=$(sbatch --parsable -w saxa --gres=gpu:h200_3g.71gb:1 --cpus-per-task=8 --mem=96G --time=3:00:00 -J vpriv -o logs2/%x-%j.out \
        slurm/py.sbatch scripts/train_causal.py --init runs/longA/ckpt_20000.pt --out runs/vpriv --K 4 --rho 0.95 \
        --rho_decision council/rho_decision.json --private_clips $PRIV --steps 16000 --resume)
    [[ "$J" =~ ^[0-9]+$ ]] || { echo "SCORE_FAIL continuation sbatch returned '$J' (CONT_PENDING kept; check sacct)"; exit 1; }
    echo "job=$J resume 12000->16000 private_clips=$PRIV" > $G2/CONTINUED && grep -q "job=$J" $G2/CONTINUED || { echo "SCORE_FAIL receipt"; exit 1; }
    echo "CONTINUED vpriv to 16k: job $J"
  fi
  exec 9>&-
fi
$PY scripts/lodo.py $M runs/troll/m_A20k_indep_dev.json --out runs/s_lock/lodo_P${k}k_vs_A20k.json || echo "LODO failed (report only)"
echo "SCORE_DONE ${k}k"
