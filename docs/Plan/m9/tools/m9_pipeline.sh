#!/bin/bash
# M9 resumable pipeline: clean (fact-check + rebuild SFT + general text) -> train (auto-resume) -> select (quick PPL + knowledge per ckpt)
# -> export/verify chosen -> full eval (M9 + official, M8 reused). Stage markers docs/Plan/m9/raw/<stage>.DONE.
cd /Users/lexchien/Documents/AI/gemma4-qat; PY=.venv-m0/bin/python; R=docs/Plan/m9/raw; T=docs/Plan/m9/tools; T7=docs/Plan/m7/tools; B=docs/Plan/m0/tools/m0_bench_mlx.py
OFF=out/m5/mlx-qat-g32exact; M9=out/m9/mlx-m9-g32exact; C=$R/cand; export HF_DATASETS_OFFLINE=1; mkdir -p $C out/m9/cand
if [ ! -f $R/clean.DONE ]; then $PY $T/m9_factcheck.py 2>&1 | grep --line-buffered -v regex | tee -a $R/factcheck.log
  grep -q FACTCHECK_DONE $R/factcheck.log || exit 1
  $PY $T/m9_build_sft.py 2>&1 | grep -v -i "regex\|warn" | tee $R/sft.log && cp $R/sft_meta.json $R/sft_meta_final.json && touch $R/clean.DONE || exit 1; fi
[ -f $R/train.DONE ] || bash $T/m9_train_loop.sh
[ -f $R/train.DONE ] || { echo "TRAIN NOT DONE"; exit 1; }
if [ ! -f $R/select.DONE ]; then
  [ -f $R/bench_m9-qat_q.json ] || { $PY $B m9-qat $OFF --limit 10240 --tag _q 2>&1 | grep -v regex | tail -2 | tee -a $R/select.log; cp docs/Plan/m0/raw/bench_m9-qat_q.json $R/; }
  LAST=$(ls out/m9/ckpt/step_*.json | sort | tail -1 | sed 's/.*step_0*//; s/.json//')
  for s in $(ls out/m9/ckpt/step_*.json | sed 's/.*step_0*//; s/.json//' | sort -n); do
    [ $((s % 50)) -eq 0 ] || [ $s -eq $LAST ] || continue
    S=$(printf %06d $s); D=out/m9/cand/step_$S; echo "== candidate $S $(date '+%T')" | tee -a $R/select.log
    [ -f $D/m9_export_report.json ] || $PY $T/m9_export.py out/m9/ckpt/step_$S $D 2>&1 | grep -v regex | tail -1 | tee -a $R/select.log
    [ -f $R/bench_m9c-${S}_q.json ] || { $PY $B m9c-$S $D --limit 10240 --tag _q 2>&1 | grep -v regex | tail -1 | tee -a $R/select.log; cp docs/Plan/m0/raw/bench_m9c-${S}_q.json $R/; }
    [ -f $C/lat_c${S}_off_summary.json ] || $PY $T7/m7_latency.py $D c${S}_off --think off --runs 1 --outdir $C 2>&1 | grep -v regex | tail -1 | tee -a $R/select.log
    RAW=$C $PY $T7/m7_grade.py c${S}_off 2>&1 | tail -1 | cut -c1-300 | tee -a $R/select.log
  done
  $PY $T/m9_select.py 2>&1 | tee -a $R/select.log; grep -q CHOSEN $R/select.log && touch $R/select.DONE || exit 1; fi
CH=$(python3 -c "import json;print('%06d'%json.load(open('$R/select.json'))['chosen'])")
if [ ! -f $R/export.DONE ]; then rm -rf $M9; cp -R out/m9/cand/step_$CH $M9 && cp $M9/m9_export_report.json $R/ &&
  $PY $T/m9_verify.py out/m9/ckpt/step_$CH $M9 2>&1 | grep -v regex | tee $R/export.log && du -sh $M9 >> $R/export.log && touch $R/export.DONE || exit 1; fi
if [ ! -f $R/eval.DONE ]; then
  for m in "m9 $M9" "qat $OFF"; do set -- $m
    [ -f $R/lat_$1_off_summary.json ] || $PY $T7/m7_latency.py $2 $1_off --think off --runs 2 --outdir $R 2>&1 | grep -v regex | tee -a $R/eval.log
    [ -f $R/lat_$1_on_summary.json ] || $PY $T7/m7_latency.py $2 $1_on --think on --runs 2 --stride 2 --max-tokens 1024 --outdir $R 2>&1 | grep -v regex | tee -a $R/eval.log
  done
  for t in m8_off m8_on; do cp docs/Plan/m8/raw/lat_$t.jsonl docs/Plan/m8/raw/lat_${t}_summary.json $R/; done
  RAW=$R $PY $T7/m7_grade.py m9_off qat_off m8_off m9_on qat_on m8_on 2>&1 | tee -a $R/eval.log
  [ -f $R/bench_m9-g32exact.json ] || { $PY $B m9-g32exact $M9 2>&1 | grep -v regex | tail -2 | tee -a $R/eval.log; cp docs/Plan/m0/raw/bench_m9-g32exact.json $R/; }
  cp docs/Plan/m8/raw/bench_m8-qat-g32exact.json docs/Plan/m8/raw/bench_m8-g32exact.json $R/
  [ -f $R/bench_m9-g32exact.json ] && touch $R/eval.DONE; fi
[ -f $R/eval.DONE ] && touch $R/M9_ALL.DONE && echo M9_ALL_DONE
