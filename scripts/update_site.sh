#!/bin/bash
# Refresh the live page (GitHub Pages, docs/index.html) with whatever is new on the cluster, then push.
#   bash scripts/update_site.sh ["commit message"]
# Pulls: training logs (runs/tpvE*/log.jsonl), placement scores (runs/judge/tpv*_m2f.json), preview renders
# (runs/preview/) -> rebuilds the comparison sheets for every preview step -> scripts/build_site.py -> commit + push.
set -e
cd "$(dirname "$0")/.."
mkdir -p pulled/site/scores pulled/preview
for e in E1 E2 E3; do rsync -a --timeout=40 -e ssh "mlp:mini-reality-view/runs/tpv$e/log.jsonl" "pulled/site/tpv$e.jsonl" < /dev/null || true; done
rsync -a --timeout=60 -e ssh --include='tpv*_m2f.json' --exclude='*' "mlp:mini-reality-view/runs/judge/" pulled/site/scores/ < /dev/null || true
rsync -a --timeout=60 -e ssh "mlp:mini-reality-view/runs/preview/" pulled/preview/ < /dev/null || true
for d in pulled/preview/tpvE1_*; do
  s=${d##*_}; [[ "$s" =~ ^[0-9]+$ ]] || continue
  python3 scripts/preview_sheet.py "$s" pulled/preview pulled/prev_cams "pulled/prev$s" > /dev/null
done
python3 scripts/build_site.py
git add docs/index.html docs/site docs/videos scripts/site_news.json
git diff --cached --quiet || git commit -q -m "${1:-Live page update}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q && echo SITE_PUSHED
