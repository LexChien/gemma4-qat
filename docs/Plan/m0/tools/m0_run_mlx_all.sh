#!/bin/bash
# M0: run MLX benchmarks sequentially (one process per model for clean peak-memory numbers)
cd ~/Documents/AI/gemma4-qat
PY=.venv-m0/bin/python; B=docs/Plan/m0/tools/m0_bench_mlx.py; L=docs/Plan/m0/raw/bench_mlx_all.log
run(){ [ -f docs/Plan/m0/raw/bench_$1.json ] && { echo "skip $1"; return; }; echo "=== $1 $(date)"; $PY $B $1 $2 2>&1 | grep -v Warning | tail -3; }
{
run mlx-it-bf16 weights/gemma-4-E2B-it
run mlx-qat-bf16 weights/gemma-4-E2B-it-qat-q4_0-unquantized
run mlx-it-q4g64 out/m0/mlx-it-q4g64
run mlx-it-q4g32 out/m0/mlx-it-q4g32
run mlx-qat-q4g32 out/m0/mlx-qat-q4g32
run mlx-qat-mobile-g64 out/m0/mlx-qat-mobile-g64
echo "MLX_ALL_DONE $(date)"
} >> $L 2>&1
