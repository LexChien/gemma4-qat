#!/bin/bash
# M8 resumable pipeline: distill -> sft -> train (auto-resume loop) -> export(+verify) -> eval. Stage markers raw/<stage>.DONE.
cd ~/Documents/AI/gemma4-qat; PY=.venv-m0/bin/python; R=docs/Plan/m8/raw; T=docs/Plan/m8/tools; T7=docs/Plan/m7/tools
OFF=out/m5/mlx-qat-g32exact; M8=out/m8/mlx-m8-g32exact
while pgrep -f m8_distill.py >/dev/null; do sleep 20; done
if [ ! -f $R/distill.DONE ]; then $PY $T/m8_distill.py --batch 32 2>&1 | grep --line-buffered -v regex | tee -a $R/distill.log
  grep -q DISTILL_DONE $R/distill.log && touch $R/distill.DONE || exit 1; fi
if [ ! -f $R/sft.DONE ]; then $PY $T/m8_build_sft.py 2>&1 | grep -v regex | tee $R/sft.log && touch $R/sft.DONE || exit 1; fi
[ -f $R/train.DONE ] || bash $T/m8_train_loop.sh --save-every 5
[ -f $R/train.DONE ] || { echo "TRAIN NOT DONE"; exit 1; }
if [ ! -f $R/export.DONE ]; then TAG=$(ls out/m8/ckpt/step_*.json | sort | tail -1); TAG=${TAG%.json}
  $PY $T/m8_export.py $TAG $M8 2>&1 | grep -v regex | tee $R/export.log && cp $M8/m8_export_report.json $R/ &&
  $PY $T/m8_verify.py $TAG $M8 2>&1 | grep -v regex | tee -a $R/export.log && du -sh $M8 >> $R/export.log && touch $R/export.DONE || exit 1; fi
if [ ! -f $R/eval.DONE ]; then
  for m in "m8 $M8" "qat $OFF"; do set -- $m
    [ -f $R/lat_$1_off_summary.json ] || $PY $T7/m7_latency.py $2 $1_off --think off --runs 2 --outdir $R 2>&1 | grep -v regex | tee -a $R/eval.log
    [ -f $R/lat_$1_on_summary.json ] || $PY $T7/m7_latency.py $2 $1_on --think on --runs 2 --stride 2 --max-tokens 1024 --outdir $R 2>&1 | grep -v regex | tee -a $R/eval.log
  done
  RAW=$R $PY $T7/m7_grade.py m8_off qat_off m8_on qat_on 2>&1 | tee -a $R/eval.log
  [ -f $R/bench_m8-g32exact.json ] || { $PY docs/Plan/m0/tools/m0_bench_mlx.py m8-g32exact $M8 2>&1 | grep -v regex | tail -3 | tee -a $R/eval.log; cp docs/Plan/m0/raw/bench_m8-g32exact.json $R/; }
  [ -f $R/bench_m8-qat-g32exact.json ] || { $PY docs/Plan/m0/tools/m0_bench_mlx.py m8-qat-g32exact $OFF 2>&1 | grep -v regex | tail -3 | tee -a $R/eval.log; cp docs/Plan/m0/raw/bench_m8-qat-g32exact.json $R/; }
  [ -f $R/bench_m8-g32exact.json ] && touch $R/eval.DONE; fi
[ -f $R/eval.DONE ] && touch $R/M8_ALL.DONE && echo M8_ALL_DONE
