#!/bin/sh
# Resumable sequential eval chain (no other heavy jobs in parallel, so timings are comparable).
set -u
cd /Users/lexchien/Documents/AI/gemma4-qat; PY=.venv-m0/bin/python; R=docs/Plan/m12/raw; ST=docs/Plan/m12/tools/status.sh
NB=$PWD/out/m12/bin/mac/gemma_native; OB=/Users/lexchien/Documents/ChatGPT/XsOin/packages/Gemma4-Portable/mac/bin/gemma_native
ORIG=/Users/lexchien/Documents/ChatGPT/XsOin/packages/Gemma4-Portable/models/gemma4-e2b-q4_0.sbs
run() { T=$1; W=$2; B=$3; X=${4:-}; [ -f $R/eval_$T.DONE ] && return
  $ST 7 9 "stage7 eval $T start"
  $PY docs/Plan/m12/tools/m12_native_eval.py $T $W $B $X > $R/eval_$T.log 2>&1 && grep -q EVAL_DONE $R/eval_$T.log && touch $R/eval_$T.DONE
  $ST 7 9 "stage7 eval $T done" ; }
run Bm8 $PWD/out/m12/gemma4-e2b-m8-v154624-q4_0.sbs $NB
run Bm9 $PWD/out/m12/gemma4-e2b-m9-v154624-q4_0.sbs $NB
run orig_newbin $ORIG $NB
run Am9 $PWD/out/m12/gemma4-e2b-m9-q4emb.sbs $OB
run orig_oldbin $ORIG $OB facts
run m11m9_oldbin $PWD/out/m11/gemma4-e2b-m9-q4_0.sbs $OB facts
run m11m8_oldbin $PWD/out/m11/gemma4-e2b-m8-q4_0.sbs $OB facts
RAW=$R $PY docs/Plan/m7/tools/m7_grade.py Bm8 Bm9 orig_newbin Am9 > $R/grade.log 2>&1
$ST 7 9 "stage7 ALL EVAL DONE (grade.log)"
