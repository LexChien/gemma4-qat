#!/bin/bash
# M0: MLX conversions (no llama.cpp). Output: out/m0/<name>
set -x
cd ~/Documents/AI/gemma4-qat
PY=.venv-m0/bin/python
W=weights
for src in it:gemma-4-E2B-it qat:gemma-4-E2B-it-qat-q4_0-unquantized; do
  n=${src%%:*}; d=${src#*:}
  for g in 64 32; do
    [ -d out/m0/mlx-$n-q4g$g ] || $PY -m mlx_lm convert --hf-path $W/$d --mlx-path out/m0/mlx-$n-q4g$g -q --q-bits 4 --q-group-size $g
  done
done
[ -d out/m0/mlx-qat-safe-g64 ] || $PY docs/Plan/m0/tools/m0_quant_mixed.py --hf $W/gemma-4-E2B-it-qat-q4_0-unquantized --out out/m0/mlx-qat-safe-g64 --profile safe --group 64
[ -d out/m0/mlx-qat-mobile-g64 ] || $PY docs/Plan/m0/tools/m0_quant_mixed.py --hf $W/gemma-4-E2B-it-qat-q4_0-unquantized --out out/m0/mlx-qat-mobile-g64 --profile mobile --group 64
echo CONVERT_ALL_DONE
