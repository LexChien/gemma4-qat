#!/bin/bash
# M7 step A: re-measure thinking-off with the CORRECT E2B format (no prefill) on official / M5 / M6; resumable.
cd /Users/lexchien/Documents/AI/gemma4-qat; PY=.venv-m0/bin/python; R=docs/Plan/m7/raw; T=docs/Plan/m7/tools
for m in "qat out/m5/mlx-qat-g32exact" "jaxlora out/m5/mlx-jaxlora-g32exact" "mlxlora out/m6/mlx-mlxlora-g32exact"; do set -- $m
  [ -f $R/lat_$1_off_summary.json ] || $PY $T/m7_latency.py $2 $1_off --think off --runs 2; done
[ -f $R/lat_qat_off_emptyprefill_summary.json ] || $PY $T/m7_latency.py out/m5/mlx-qat-g32exact qat_off_emptyprefill --think off --runs 1 --empty-prefill
$PY $T/m7_grade.py qat_off jaxlora_off mlxlora_off qat_off_emptyprefill
echo M7_BASELINE_DONE
