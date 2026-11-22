#!/bin/bash
# M3 step 1: convert orbax it -> bf16 HF, then tensor-by-tensor compare vs weights/gemma-4-E2B-it
cd /Users/lexchien/Documents/AI/gemma4-qat
PY=.venv-m0/bin/python; RAW=docs/Plan/m3/raw
{
echo "=== convert it bf16 $(date)"
rm -rf out/m3/it-bf16
/usr/bin/time -l $PY tools/convert_jax_to_hf.py --orbax weights/orbax/checkpoints/gemma4-e2b-it --out out/m3/it-bf16 --dtype bf16 2>&1 | grep -vE "Redirect" | tail -25
echo "=== compare it $(date)"
$PY docs/Plan/m3/tools/m3_compare_tensors.py out/m3/it-bf16 weights/gemma-4-E2B-it $RAW/compare_it_bf16.json 2>&1 | grep -v Redirect
echo "M3_IT_DONE $(date)"
} > $RAW/run_it.log 2>&1
