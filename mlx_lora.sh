#!/usr/bin/env bash
# PRIMARY 4-bit route on Apple Silicon: MLX-LM (Gemma 4 support in mlx-lm: UNVERIFIED, check release notes)
# data/ must contain train.jsonl valid.jsonl ({"messages":[...]})
# Usage: bash mlx_lora.sh [lora|full|convert-only <merged_hf_dir>]
set -e; pip install -U mlx-lm
M=google/gemma-4-E2B-it-qat-q4_0-unquantized
MODE=${1:-lora}
if [ "$MODE" = "convert-only" ]; then SRC=${2:-out/merged}; else
  FT=$([ "$MODE" = "full" ] && echo "--fine-tune-type full" || echo "--num-layers 16")
  python -m mlx_lm lora --model $M --train --data data --iters 600 --batch-size 1 $FT --adapter-path out/mlx_adapter
  python -m mlx_lm fuse --model $M --adapter-path out/mlx_adapter --save-path out/mlx_merged
  SRC=out/mlx_merged
fi
# 4-bit, group 64. mlx_lm quantizes Linear AND Embedding layers by default (embeddings incl. PLE -> 4-bit).
# If mlx_lm loads the vision/audio towers, strip them first for a text-only artifact (see README).
python -m mlx_lm convert --hf-path "$SRC" --mlx-path out/mlx_q4 -q --q-bits 4 --q-group-size 64
du -sh out/mlx_q4
