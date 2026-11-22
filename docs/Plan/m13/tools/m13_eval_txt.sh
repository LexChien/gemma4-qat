#!/bin/sh
# experiment: same website policy content as plain text (m13_env_text.py), after the envelope chain
cd /Users/lexchien/Documents/AI/gemma4-qat; PY=.venv-m0/bin/python; R=docs/Plan/m13/raw; ST=docs/Plan/m13/tools/status.sh
while pgrep -f "m13_eval_env.sh" >/dev/null; do sleep 20; done
ORIG=/Users/lexchien/Documents/ChatGPT/XsOin/packages/Gemma4-Portable/models/gemma4-e2b-q4_0.sbs
export M13_ENVFILE=envelopes_txt.json M13_ENVPFX=txt
run() { [ -f $R/eval_$1_txt$3.DONE ] && return; $PY docs/Plan/m13/tools/m13_native_eval.py $1 $2 $3 > $R/eval_$1_txt$3.log 2>&1 && touch $R/eval_$1_txt$3.DONE; }
run Am8 $PWD/out/m13/gemma4-e2b-m8-q4emb.sbs envsets; run orig $ORIG envsets; run Am9 $PWD/out/m12/gemma4-e2b-m9-q4emb.sbs envsets
$ST 6 7 "plain-text policy experiment done"
