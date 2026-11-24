#!/bin/sh
# clean sequential 50-prompt runs (nothing else heavy running) for speed/peak-mem/grades; waits for the sets chain first
cd ~/Documents/AI/gemma4-qat; PY=.venv-m0/bin/python; R=docs/Plan/m13/raw; ST=docs/Plan/m13/tools/status.sh
while pgrep -f m13_eval_sets2.sh >/dev/null; do sleep 20; done
ORIG=$WEBSITE_ROOT/packages/Gemma4-Portable/models/gemma4-e2b-q4_0.sbs
run() { [ -f $R/eval_$1_p50.DONE ] && return; $PY docs/Plan/m13/tools/m13_native_eval.py $1 $2 p50 > $R/eval_$1_p50.log 2>&1 && touch $R/eval_$1_p50.DONE; }
run Am8 $PWD/out/m13/gemma4-e2b-m8-q4emb.sbs; run orig $ORIG; run Am9 $PWD/out/m12/gemma4-e2b-m9-q4emb.sbs
RAW=$R $PY docs/Plan/m7/tools/m7_grade.py Am8 orig Am9 > $R/grade_p50.log 2>&1
$ST 6 7 "clean p50 runs + grading done"
