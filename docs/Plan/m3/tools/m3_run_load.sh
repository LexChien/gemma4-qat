#!/bin/bash
# M3 step 3: transformers logits + MLX PPL on converted checkpoints
cd ~/Documents/AI/gemma4-qat
PY=".venv-m0/bin/python -u"; RAW=docs/Plan/m3/raw; B=docs/Plan/m0/tools/m0_bench_mlx.py
{
echo "=== transformers logits $(date)"
$PY docs/Plan/m3/tools/m3_logits_torch.py $RAW/logits_torch.json ref_it=weights/gemma-4-E2B-it conv_it=out/m3/it-bf16 conv_it_text=out/m3/it-text-bf16 conv_it_f32=out/m3/it-f32 2>&1 | grep --line-buffered -vE "Redirect|Warning|warn|Loading weights" | cut -c1-700
$PY docs/Plan/m3/tools/m3_logits_torch.py $RAW/logits_torch_pt.json ref_pt=weights/gemma-4-E2B conv_pt=out/m3/pt-bf16 2>&1 | grep --line-buffered -vE "Redirect|Warning|warn|Loading weights" | cut -c1-700
echo "=== MLX PPL (same protocol as M0) $(date)"
for s in m3-conv-it-bf16:out/m3/it-bf16 m3-conv-it-text-bf16:out/m3/it-text-bf16 m3-conv-pt-bf16:out/m3/pt-bf16; do
  $PY $B ${s%%:*} ${s#*:} 2>&1 | grep -v Warning | tail -2 | cut -c1-500
  mv docs/Plan/m0/raw/bench_${s%%:*}.json $RAW/ 2>/dev/null
done
echo "M3_LOAD_DONE $(date)"
} > $RAW/run_load.log 2>&1
