#!/bin/bash
# M13 auto-resume loop (copied from m9_train_loop.sh): restart from the latest checkpoint on Metal errors, up to 400 attempts.
cd ~/Documents/AI/gemma4-qat; W=out/m13/work/ft; L=$W/train.log; n=0
while [ ! -f $W/train.DONE ] && [ $n -lt 400 ]; do n=$((n+1)); echo "=== attempt $n $(date '+%F %T')" | tee -a $L
  .venv-m0/bin/python docs/Plan/m13/tools/m13_train.py "$@" 2>&1 | grep --line-buffered -v regex | tee -a $L
  [ -f $W/train.DONE ] || { echo "=== crash/exit on attempt $n; restarting in 10s" | tee -a $L; sleep 10; }
done
