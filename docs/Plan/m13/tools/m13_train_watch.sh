#!/bin/bash
# append a training progress bar to out/m13/STATUS.txt every 10 min until train.DONE
cd ~/Documents/AI/gemma4-qat; W=out/m13/work/ft; S=out/m13/STATUS.txt
while [ ! -f $W/train.DONE ]; do
  L=$(grep '^step [0-9]*/' $W/train.log | tail -1); V=$(grep '^step [0-9]* val' $W/train.log | tail -1)
  if [ -n "$L" ]; then n=$(echo "$L" | sed 's#step \([0-9]*\)/.*#\1#'); t=$(echo "$L" | sed 's#step [0-9]*/\([0-9]*\).*#\1#'); sps=$(echo "$L" | sed 's#.*s/step \([0-9.]*\).*#\1#')
    f=$((n*20/t)); bar=$(printf '%*s' $f '' | tr ' ' '#')$(printf '%*s' $((20-f)) '' | tr ' ' '.')
    eta=$(date -v+$(python3 -c "print(int(($t-$n)*$sps/60))")M '+%H:%M')
    echo "$(date '+%F %T %Z')   train [$bar] $n/$t ${sps}s/step ETA $eta | last $V" >> $S; fi
  sleep 600; done
