#!/bin/bash
# M0 phase 2 (runs after m0_run_mlx_all.sh): official-CT (dequantized) PPL, torch cross-check, clean speed pass
cd /Users/lexchien/Documents/AI/gemma4-qat
PY=.venv-m0/bin/python; B=docs/Plan/m0/tools/m0_bench_mlx.py; RAW=docs/Plan/m0/raw; L=$RAW/bench_phase2.log
until grep -q MLX_ALL_DONE $RAW/bench_mlx_all.log 2>/dev/null; do sleep 20; done
{
echo "=== CT load attempt (transformers + compressed-tensors) $(date)"
$PY docs/Plan/m0/tools/m0_bench_torch.py ct-mobile weights/gemma-4-E2B-it-qat-mobile-ct --device mps --smoke > $RAW/ct_load_attempt_mobile.log 2>&1; echo "exit $?"
$PY docs/Plan/m0/tools/m0_bench_torch.py ct-w4a16 weights/gemma-4-E2B-it-qat-w4a16-ct --device mps --smoke > $RAW/ct_load_attempt_w4a16.log 2>&1; echo "exit $?"
for n in ct-w4a16-dequant-bf16 ct-mobile-dequant-bf16; do
  echo "=== $n $(date)"; [ -f $RAW/bench_mlx-$n.json ] || $PY $B mlx-$n out/m0/$n 2>&1 | grep -v Warning | tail -2
done
echo "=== xcheck torch vs mlx (first 4096 tokens) $(date)"
$PY docs/Plan/m0/tools/m0_bench_torch.py torch-it-bf16 weights/gemma-4-E2B-it --device mps --limit 4096 --tag _x4096 2>&1 | grep -v -i warn | tail -2
$PY $B mlx-it-bf16 weights/gemma-4-E2B-it --limit 4096 --tag _x4096 2>&1 | grep -v Warning | tail -2
echo "=== speed pass $(date)"
for s in mlx-it-bf16:weights/gemma-4-E2B-it mlx-qat-bf16:weights/gemma-4-E2B-it-qat-q4_0-unquantized \
         mlx-it-q4g64:out/m0/mlx-it-q4g64 mlx-it-q4g32:out/m0/mlx-it-q4g32 mlx-qat-q4g64:out/m0/mlx-qat-q4g64 \
         mlx-qat-q4g32:out/m0/mlx-qat-q4g32 mlx-qat-mobile-g64:out/m0/mlx-qat-mobile-g64; do
  for rep in 1 2; do $PY $B ${s%%:*} ${s#*:} --no-ppl --tag _speed$rep 2>&1 | grep -v Warning | tail -1 | cut -c1-400; done
done
echo "PHASE2_DONE $(date)"
} >> $L 2>&1
