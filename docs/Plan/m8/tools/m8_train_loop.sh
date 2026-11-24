#!/bin/bash
# Auto-resume loop for Metal Internal Errors: restart from the latest global checkpoint, up to 400 attempts.
cd ~/Documents/AI/gemma4-qat; R=docs/Plan/m8/raw; n=0
while [ ! -f $R/train.DONE ] && [ $n -lt 400 ]; do n=$((n+1)); echo "=== attempt $n $(date '+%F %T')" | tee -a $R/train.log
  .venv-m0/bin/python docs/Plan/m8/tools/m8_train.py "$@" 2>&1 | grep --line-buffered -v regex | tee -a $R/train.log
  [ -f $R/train.DONE ] || { echo "=== crash/exit on attempt $n; restarting in 10s" | tee -a $R/train.log; grep -c "METAL" $R/train.log; sleep 10; }
done
