#!/bin/bash
# M0 phase 3: re-dequantize official CT (lm_head tie fix) -> MLX PPL; base PT bf16 reference
cd /Users/lexchien/Documents/AI/gemma4-qat
PY=.venv-m0/bin/python; B=docs/Plan/m0/tools/m0_bench_mlx.py; RAW=docs/Plan/m0/raw; L=$RAW/bench_phase3.log
until grep -q PHASE2_DONE $RAW/bench_phase2.log 2>/dev/null; do sleep 15; done
{
for n in w4a16 mobile; do
  echo "=== dequant $n $(date)"
  rm -rf out/m0/ct-$n-dequant-bf16
  $PY docs/Plan/m0/tools/m0_ct_dequant.py weights/gemma-4-E2B-it-qat-$n-ct out/m0/ct-$n-dequant-bf16 2>&1 | grep -v Redirect | tail -2
  echo "=== bench $n $(date)"
  $PY $B mlx-ct-$n-dequant-bf16 out/m0/ct-$n-dequant-bf16 2>&1 | grep -v Warning | tail -2 | cut -c1-600
done
echo "=== base pt bf16 $(date)"
$PY $B mlx-pt-bf16 weights/gemma-4-E2B 2>&1 | grep -v Warning | tail -2 | cut -c1-600
echo "PHASE3_DONE $(date)"
} >> $L 2>&1
