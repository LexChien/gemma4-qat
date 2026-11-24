#!/bin/sh
set -eu
cd ~/Documents/AI/gemma4-qat; S=out/m11/STATUS.txt; R=docs/Plan/m11/raw; W=$PWD/out/m11/work
echo "$(date '+%F %T') stage=build_hf:m8 pct=0" >> $S
[ -e $W/m8-hf-text.safetensors ] || .venv-m0/bin/python docs/Plan/m11/tools/m11_build_hf.py out/m8/mlx-m8-g32exact $W/m8-hf-text.safetensors > $R/build_hf_m8.log 2>&1
echo "$(date '+%F %T') stage=build_hf:m8 pct=100" >> $S
[ -e $W/m8-pipeline.sbs ] || docs/Plan/m11/tools/m11_convert.sh m8-pipeline $W/m8-hf-text.safetensors $W/m8-pipeline.sbs $W/m8-pipeline.csv > $R/convert_m8-pipeline.log 2>&1
docs/Plan/m11/tools/m11_exact_run.sh m8 out/m8/mlx-m8-g32exact
