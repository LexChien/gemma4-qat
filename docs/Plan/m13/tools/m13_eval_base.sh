#!/bin/sh
cd ~/Documents/AI/gemma4-qat; PY=.venv-m0/bin/python; R=docs/Plan/m13/raw; ST=docs/Plan/m13/tools/status.sh
ORIG=$WEBSITE_ROOT/packages/Gemma4-Portable/models/gemma4-e2b-q4_0.sbs
run() { [ -f $R/eval_$1_$3.DONE ] && return; $PY docs/Plan/m13/tools/m13_native_eval.py $1 $2 $3 > $R/eval_$1_$3.log 2>&1 && touch $R/eval_$1_$3.DONE; }
run orig $ORIG sets; run Am8 $PWD/out/m13/gemma4-e2b-m8-q4emb.sbs sets; run Am9 $PWD/out/m12/gemma4-e2b-m9-q4emb.sbs sets
run m11m8 $PWD/out/m11/gemma4-e2b-m8-q4_0.sbs sets; run m11m9 $PWD/out/m11/gemma4-e2b-m9-q4_0.sbs sets
run Am8 $PWD/out/m13/gemma4-e2b-m8-q4emb.sbs p50
$ST 3 7 "baseline evals done"
