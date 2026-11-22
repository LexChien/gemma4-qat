#!/bin/sh
# A-layout SBS (c_embedding + ple_model_proj Q4_0, every Q4_0 block bit-exact to the MLX g32exact model) for the STOCK binary.
# usage: m13_build.sh <tag> <mlx_dir>   -> out/m13/gemma4-e2b-<tag>-q4emb.sbs (+ .sha256), verify json in docs/Plan/m13/raw
set -eu
cd /Users/lexchien/Documents/AI/gemma4-qat; T=$1; M=$2; W=out/m13/work; R=docs/Plan/m13/raw; O=out/m13/gemma4-e2b-$T-q4emb.sbs
[ -e $O ] && { echo exists $O; exit 0; }
[ -e $W/$T-hf-text.safetensors ] || .venv-m0/bin/python docs/Plan/m11/tools/m11_build_hf.py $M $W/$T-hf-text.safetensors > $R/build_hf_$T.log 2>&1
M12_Q4=c_embedding,ple_model_proj /Users/lexchien/Documents/AI/ALife-Platform/.venv/bin/python -u docs/Plan/m12/tools/m12_convert.py $W/$T-hf-text.safetensors /Users/lexchien/Documents/AI/ALife-Platform/Google-Gemma4/originals/models/gemma4-e2b-qat/tokenizer.json $W/$T-A-pipeline.sbs $W/$T-A-pipeline.csv > $R/convert_$T.log 2>&1
.venv-m0/bin/python docs/Plan/m12/tools/m12_exact.py patch $M $W/$T-A-pipeline.sbs $O > $R/patch_$T.log 2>&1
.venv-m0/bin/python docs/Plan/m12/tools/m12_exact.py verify $M $O $R/verify_$T.json > $R/verify_$T.log 2>&1
grep -q '"PASS": true' $R/verify_$T.json
rm -f $W/$T-A-pipeline.sbs
(cd out/m13 && shasum -a 256 $(basename $O) > $(basename $O).sha256)
docs/Plan/m13/tools/status.sh 5 7 "built $O $(stat -f %z $O) B verify PASS"
