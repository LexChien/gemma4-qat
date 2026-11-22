#!/bin/bash
# M13 fine-tune pipeline (resumable; markers in out/m13/work/ft): wait for the website-envelope eval chain -> train (auto-resume)
# -> export candidates (step 100 + final; lossless on-grid check) -> SBS build+verify (m13_build.sh) -> wait for plain-text experiment
# -> native website-profile evals (sets, envsets, p50, envp50) -> grading.
cd /Users/lexchien/Documents/AI/gemma4-qat; PY=.venv-m0/bin/python; R=docs/Plan/m13/raw; T=docs/Plan/m13/tools; W=out/m13/work/ft; ST=$T/status.sh
while pgrep -f "m13_eval_env.sh|m13_eval_txt.sh" >/dev/null; do sleep 20; done
[ -f $W/train.DONE ] || { $ST 5 7 "fine-tune start (737 train / 40 valid, 185 steps, rank 8, lr 4e-5, beta 1)"; bash $T/m13_train_loop.sh --rank 8 --lr 4e-5 --batch 4 --epochs 1 --save-every 25 --eval-every 25; }
[ -f $W/train.DONE ] || { $ST 5 7 "TRAIN NOT DONE"; exit 1; }
cp $W/train.log $R/ft_train.log
$ST 5 7 "fine-tune done"
LAST=$(ls out/m13/work/ckpt/step_*.json | sort | tail -1 | sed 's/.*step_0*//; s/.json//')
for s in 100 $LAST; do S=$(printf %06d $s); D=out/m13/work/mlx-ft$s
  [ -f $D/m13_export_report.json ] || $PY $T/m13_export.py out/m13/work/ckpt/step_$S $D 2>&1 | grep -v -i "regex\|warn" | tail -1 > $R/ft_export_$s.log
  cp $D/m13_export_report.json $R/ft_export_$s.json; grep -q '"lossless": true' $D/m13_export_report.json || { $ST 5 7 "export $s NOT lossless"; exit 1; }
  sh $T/m13_build.sh ft$s $D || { $ST 5 7 "build ft$s FAILED"; exit 1; }
done
while pgrep -f "m13_eval_txt.sh" >/dev/null; do sleep 20; done
run() { [ -f $R/eval_$1_$3.DONE ] && return; $PY $T/m13_native_eval.py $1 $2 $3 > $R/eval_$1_$3.log 2>&1 && touch $R/eval_$1_$3.DONE; }
for s in 100 $LAST; do O=$PWD/out/m13/gemma4-e2b-ft$s-q4emb.sbs; for m in sets envsets p50 envp50; do run ft$s $O $m; done; done
RAW=$R $PY docs/Plan/m7/tools/m7_grade.py ft100 ft$LAST envft100 envft$LAST Am8 envAm8 > $R/grade_ft.log 2>&1
$PY $T/m13_table.py > $R/table_all.txt 2>&1
$ST 6 7 "fine-tune candidates evaluated"; touch $W/PIPELINE.DONE
