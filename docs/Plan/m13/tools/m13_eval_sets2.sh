#!/bin/sh
# full 120-item set (base tier resumes, hard tier appended) on the stock binary + MLX references
cd /Users/lexchien/Documents/AI/gemma4-qat; PY=.venv-m0/bin/python; R=docs/Plan/m13/raw; ST=docs/Plan/m13/tools/status.sh
ORIG=/Users/lexchien/Documents/ChatGPT/XsOin/packages/Gemma4-Portable/models/gemma4-e2b-q4_0.sbs
run() { [ -f $R/eval_$1_sets120.DONE ] && return; $PY docs/Plan/m13/tools/m13_native_eval.py $1 $2 sets > $R/eval_$1_sets120.log 2>&1 && touch $R/eval_$1_sets120.DONE; }
run Am8 $PWD/out/m13/gemma4-e2b-m8-q4emb.sbs; run Am9 $PWD/out/m12/gemma4-e2b-m9-q4emb.sbs; run orig $ORIG
run Am9s150 $PWD/out/m13/gemma4-e2b-m9s150-q4emb.sbs; run Am9s246 $PWD/out/m13/gemma4-e2b-m9s246-q4emb.sbs
run m11m8 $PWD/out/m11/gemma4-e2b-m8-q4_0.sbs
$ST 4 7 "120-item native sets done"
