#!/bin/bash
# M10 tasks 1+2: G3 compressed-tensors build from M9 + HF load/PPL/chat; P1/P3/Q2 measurements. Markers docs/Plan/m10/raw/*.DONE
cd ~/Documents/AI/gemma4-qat; PY=.venv-m0/bin/python; R=docs/Plan/m10/raw; export PYTHONPATH=src HF_DATASETS_OFFLINE=1
M9=out/m9/mlx-m9-g32exact; CT=out/m10/m9-ct-g256; CT32=out/m10/m9-ct-g32; OFFCT=weights/gemma-4-E2B-it-qat-mobile-ct
snap() { { echo "== $1 $(date '+%F %T') UTC+8"; ps -Ao pid,pcpu,pmem,comm -r | head -9 | cut -c1-130; } >> $R/cpu_load.txt; }
snap start
[ -f $R/ct_export.DONE ] || { $PY -m g4qat.export_ct $M9 $CT --ple-group 256 2>&1 | grep -v NOTE | tee $R/ct_export.log | grep -E "embed|layers.0\.|DONE|size"; grep -q EXPORT_CT_DONE $R/ct_export.log && touch $R/ct_export.DONE || exit 1; }
[ -f $R/ct32_export.DONE ] || { $PY -m g4qat.export_ct $M9 $CT32 --ple-group 32 2>&1 | grep -v NOTE > $R/ct32_export.log; grep -q EXPORT_CT_DONE $R/ct32_export.log && touch $R/ct32_export.DONE || exit 1; }
du -sh $CT $CT32 $OFFCT $M9 > $R/sizes.txt
[ -f $R/hf_m9ct.DONE ] || { $PY -m g4qat.hf_eval $CT m9ct $R 2>&1 | grep -v -i "warn\|NOTE" | tee $R/hf_m9ct.log; grep -q HF_EVAL_DONE $R/hf_m9ct.log && touch $R/hf_m9ct.DONE; }
snap after_m9ct
[ -f $R/hf_offct.DONE ] || { $PY -m g4qat.hf_eval $OFFCT offct $R 2>&1 | grep -v -i "warn\|NOTE" | tee $R/hf_offct.log; grep -q HF_EVAL_DONE $R/hf_offct.log && touch $R/hf_offct.DONE; }
[ -f $R/hf_m9ct32.DONE ] || { $PY -m g4qat.hf_eval $CT32 m9ct32 $R --no-chat 2>&1 | grep -v -i "warn\|NOTE" | tee $R/hf_m9ct32.log; grep -q HF_EVAL_DONE $R/hf_m9ct32.log && touch $R/hf_m9ct32.DONE; }
RAW=$R $PY docs/Plan/m7/tools/m7_grade.py m9ct offct 2>&1 | tail -2 | cut -c1-400 | tee $R/grade.log
snap before_mlx_bench
[ -f $R/bench_m10-m9.json ] || { $PY docs/Plan/m0/tools/m0_bench_mlx.py m10-m9 $M9 2>&1 | grep -v regex | tail -2 | tee $R/mlx_bench.log; cp docs/Plan/m0/raw/bench_m10-m9.json $R/; }
snap end; if [ -f $R/hf_m9ct.DONE ] && [ -f $R/hf_offct.DONE ] && [ -f $R/hf_m9ct32.DONE ]; then touch $R/T12_ALL.DONE; echo T12_ALL_DONE; else echo "T12_INCOMPLETE (hf_eval stage failed)"; exit 1; fi
