#!/bin/sh
# website-envelope runs (exact production prompt), sequential after the p50 chain
cd ~/Documents/AI/gemma4-qat; PY=.venv-m0/bin/python; R=docs/Plan/m13/raw; ST=docs/Plan/m13/tools/status.sh
while pgrep -f "m13_eval_sets2.sh|m13_eval_p50.sh" >/dev/null; do sleep 20; done
ORIG=$WEBSITE_ROOT/packages/Gemma4-Portable/models/gemma4-e2b-q4_0.sbs
run() { [ -f $R/eval_$1_$3.DONE ] && return; $PY docs/Plan/m13/tools/m13_native_eval.py $1 $2 $3 > $R/eval_$1_$3.log 2>&1 && touch $R/eval_$1_$3.DONE; }
run Am8 $PWD/out/m13/gemma4-e2b-m8-q4emb.sbs envsets; run orig $ORIG envsets; run Am9 $PWD/out/m12/gemma4-e2b-m9-q4emb.sbs envsets
run Am8 $PWD/out/m13/gemma4-e2b-m8-q4emb.sbs envp50; run orig $ORIG envp50
RAW=$R $PY docs/Plan/m7/tools/m7_grade.py envAm8 envorig > $R/grade_envp50.log 2>&1
$ST 6 7 "website-envelope runs done"
