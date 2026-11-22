#!/bin/sh
# usage: m11_exact_run.sh <tag m9|m8> <mlx_dir>
set -eu
cd /Users/lexchien/Documents/AI/gemma4-qat
T=$1; MLX=$2; S=out/m11/STATUS.txt; R=docs/Plan/m11/raw; PY=.venv-m0/bin/python
OUT=out/m11/gemma4-e2b-$T-q4_0.sbs; ORIG=/Users/lexchien/Documents/ChatGPT/XsOin/packages/Gemma4-Portable/models/gemma4-e2b-q4_0.sbs
echo "$(date '+%F %T') stage=exact_patch:$T pct=0" >> $S
[ -e $OUT ] || $PY docs/Plan/m11/tools/m11_exact.py patch $MLX out/m11/work/$T-pipeline.sbs $OUT > $R/exact_patch_$T.log 2>&1
echo "$(date '+%F %T') stage=exact_patch:$T pct=100; stage=verify:$T pct=0" >> $S
$PY docs/Plan/m11/tools/m11_exact.py verify $MLX $OUT $ORIG $R/verify_$T.json > $R/verify_$T.log 2>&1
(cd out/m11 && shasum -a 256 gemma4-e2b-$T-q4_0.sbs > gemma4-e2b-$T-q4_0.sbs.sha256)
echo "$(date '+%F %T') stage=verify:$T pct=100 PASS=$(grep -o '"PASS": [a-z]*' $R/verify_$T.json) $(cat out/m11/gemma4-e2b-$T-q4_0.sbs.sha256)" >> $S
