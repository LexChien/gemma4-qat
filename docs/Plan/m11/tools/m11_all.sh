#!/bin/sh
# Resumable: M8 conversion chain, then 50-prompt gemma_native eval for orig / m9 / m8, then grade.
set -u
cd ~/Documents/AI/gemma4-qat; R=docs/Plan/m11/raw; S=out/m11/STATUS.txt; PY=.venv-m0/bin/python
ORIG=$WEBSITE_ROOT/packages/Gemma4-Portable/models/gemma4-e2b-q4_0.sbs
[ -e out/m11/gemma4-e2b-m8-q4_0.sbs.sha256 ] || docs/Plan/m11/tools/m11_m8_chain.sh || { echo "$(date '+%F %T') stage=m8_chain FAILED" >> $S; exit 1; }
for T in orig m9 m8; do
  case $T in orig) W=$ORIG;; *) W=$PWD/out/m11/gemma4-e2b-$T-q4_0.sbs;; esac
  [ -f $R/eval_$T.DONE ] && continue
  $PY docs/Plan/m11/tools/m11_native_eval.py native_$T $W > $R/eval_$T.log 2>&1 && grep -q EVAL_DONE $R/eval_$T.log && touch $R/eval_$T.DONE
done
RAW=$R $PY docs/Plan/m7/tools/m7_grade.py native_orig native_m9 native_m8 > $R/grade.log 2>&1
echo "$(date '+%F %T') stage=ALL_DONE pct=100" >> $S
