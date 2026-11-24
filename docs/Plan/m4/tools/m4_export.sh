#!/bin/bash
# M4: smoke orbax -> HF (tools/convert_jax_to_hf.py) -> MLX 4-bit g64 (M0 route) -> PPL (m0_bench_mlx.py)
set -ex
cd ~/Documents/AI/gemma4-qat
NAME=${1:-smoke}; ORB=${2:-out/m4/smoke_orbax}
.venv-m0/bin/python tools/convert_jax_to_hf.py --orbax $ORB --out out/m4/$NAME-hf --hf_ref weights/gemma-4-E2B-it --dtype bf16 --text_only --drop_shared_kv
rm -rf out/m4/mlx-$NAME-q4g64
.venv-m0/bin/python -m mlx_lm convert --hf-path out/m4/$NAME-hf --mlx-path out/m4/mlx-$NAME-q4g64 -q --q-bits 4 --q-group-size 64
du -sh out/m4/mlx-$NAME-q4g64
.venv-m0/bin/python docs/Plan/m0/tools/m0_bench_mlx.py m4-$NAME-q4g64 out/m4/mlx-$NAME-q4g64
cp docs/Plan/m0/raw/bench_m4-$NAME-q4g64.json docs/Plan/m4/raw/ || true
echo EXPORT_DONE
