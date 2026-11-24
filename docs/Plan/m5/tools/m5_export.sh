#!/bin/bash
# M5 export + eval: HF text dir (bf16) -> MLX g32 exact-grid (+--snap if off-grid), MLX g64 affine, ct g32; PPL (M0 method)
# usage: m5_export.sh NAME HF_DIR [snap]
set -ex
cd ~/Documents/AI/gemma4-qat
NAME=$1; HF=$2; SNAP=${3:+--snap}; PY=.venv-m0/bin/python
rm -rf out/m5/_tmp_g32_$NAME out/m5/mlx-$NAME-g32exact out/m5/mlx-$NAME-g64
$PY -m mlx_lm convert --hf-path $HF --mlx-path out/m5/_tmp_g32_$NAME -q --q-bits 4 --q-group-size 32
$PY tools/mlx_exact_grid.py --hf $HF --mlx out/m5/_tmp_g32_$NAME --out out/m5/mlx-$NAME-g32exact $SNAP
rm -rf out/m5/_tmp_g32_$NAME
$PY -m mlx_lm convert --hf-path $HF --mlx-path out/m5/mlx-$NAME-g64 -q --q-bits 4 --q-group-size 64
$PY tools/pack_compressed_tensors.py --hf $HF --out out/m5/ct-$NAME-g32 $SNAP
du -sh out/m5/mlx-$NAME-g32exact out/m5/mlx-$NAME-g64 out/m5/ct-$NAME-g32
for v in g32exact g64; do
  $PY docs/Plan/m0/tools/m0_bench_mlx.py m5-$NAME-$v out/m5/mlx-$NAME-$v && cp docs/Plan/m0/raw/bench_m5-$NAME-$v.json docs/Plan/m5/raw/
done
echo EXPORT_DONE_$NAME
