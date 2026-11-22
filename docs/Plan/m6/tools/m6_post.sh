#!/bin/bash
# M6 post-train pipeline; each step skips if its output exists (resumable). usage: m6_post.sh
set -x; cd /Users/lexchien/Documents/AI/gemma4-qat; PY=.venv-m0/bin/python; R=docs/Plan/m6/raw; T=docs/Plan/m6/tools
OFF=out/m5/mlx-qat-g32exact; M5=out/m5/mlx-jaxlora-g32exact; M6=out/m6/mlx-mlxlora-g32exact
# 1 fuse + snap/pack
[ -d out/m6/mlxlora-fused ] || $PY -m mlx_lm fuse --model out/m5/qat-text-hf --adapter-path out/m6/mlx_lora --save-path out/m6/mlxlora-fused
if [ ! -f $M6/exact_grid_report.json ]; then
  rm -rf out/m6/_tmp_g32; $PY -m mlx_lm convert --hf-path out/m6/mlxlora-fused --mlx-path out/m6/_tmp_g32 -q --q-bits 4 --q-group-size 32
  $PY tools/mlx_exact_grid.py --hf out/m6/mlxlora-fused --mlx out/m6/_tmp_g32 --out $M6 --snap && rm -rf out/m6/_tmp_g32; cp $M6/exact_grid_report.json $R/exact_grid_mlxlora.json
fi
[ -d out/m6/mlx-mlxlora-g64 ] || $PY -m mlx_lm convert --hf-path out/m6/mlxlora-fused --mlx-path out/m6/mlx-mlxlora-g64 -q --q-bits 4 --q-group-size 64
du -sh out/m6/mlxlora-fused $M6 out/m6/mlx-mlxlora-g64 $OFF $M5 | tee $R/sizes.txt
# 2 PPL + speed (M0 method) run1, speed run2
b() { [ -f $R/bench_$1$3.json ] || { $PY docs/Plan/m0/tools/m0_bench_mlx.py $1 $2 $4 --tag "$3" && cp docs/Plan/m0/raw/bench_$1$3.json $R/; }; }
b m6-mlxlora-g32exact $M6 ""; b m6-mlxlora-g64 out/m6/mlx-mlxlora-g64 ""; b m6-mlxlora-fused-bf16 out/m6/mlxlora-fused ""
for m in "m6-qat-g32exact $OFF" "m6-jaxlora-g32exact $M5" "m6-mlxlora-g32exact $M6"; do set -- $m
  b $1 $2 _speed1 --no-ppl; b $1 $2 _speed2 --no-ppl; done
# 3+4 latency (+ answers for chat grading). off = enable_thinking=False + empty thought prefill; on = <|think|>, every 2nd prompt
for m in "qat $OFF" "jaxlora $M5" "mlxlora $M6"; do set -- $m
  [ -f $R/lat_$1_off_summary.json ] || $PY $T/m6_latency.py $2 $1_off --think off --runs 2
  [ -f $R/lat_$1_on_summary.json ] || $PY $T/m6_latency.py $2 $1_on --think on --runs 2 --stride 2 --max-tokens 1024; done
$PY $T/m6_grade.py qat_off jaxlora_off mlxlora_off qat_on jaxlora_on mlxlora_on
# 5 speed levers (official model)
[ -f $R/lat_lever_kv8_off_summary.json ] || $PY $T/m6_latency.py $OFF lever_kv8_off --think off --runs 2 --kv-bits 8
[ -f $R/lat_lever_kv4_off_summary.json ] || $PY $T/m6_latency.py $OFF lever_kv4_off --think off --runs 2 --kv-bits 4
[ -f $R/lat_lever_g64_off_summary.json ] || $PY $T/m6_latency.py out/m0/mlx-qat-q4g64 lever_g64_off --think off --runs 2
[ -f $R/lat_lever_budget128_on_summary.json ] || $PY $T/m6_latency.py $OFF lever_budget128_on --think on --runs 1 --stride 2 --max-tokens 1024 --think-budget 128
[ -f $R/lat_lever_budget256_on_summary.json ] || $PY $T/m6_latency.py $OFF lever_budget256_on --think on --runs 1 --stride 2 --max-tokens 1024 --think-budget 256
$PY $T/m6_grade.py lever_budget128_on lever_budget256_on lever_kv4_off
[ -f $R/lever_prompt_cache_qat.json ] || $PY $T/m6_lever_prompt_cache.py $OFF qat
echo M6_POST_DONE
