#!/bin/bash
# usage: m7_progress.sh "<space separated expected files in docs/Plan/m7/raw>" DONE_MARKER LOG  -> visible bar + out/m7/STATUS.txt
cd ~/Documents/AI/gemma4-qat; L="$1"; MK=$2; LOG=$3; N=$(echo $L | wc -w | tr -d ' ')
while true; do d=0; nxt=""; for f in $L; do if [ -f docs/Plan/m7/raw/$f ]; then d=$((d+1)); elif [ -z "$nxt" ]; then nxt=$f; fi; done
  p=$((d*40/N)); bar=$(printf '%*s' $p '' | tr ' ' '#')$(printf '%*s' $((40-p)) '' | tr ' ' '.')
  last=$(grep -aE "^[0-9]+ |step|iter" $LOG 2>/dev/null | tail -1 | cut -c1-70)
  printf "\r[%s] %d/%d next:%-28s | %s %s " "$bar" $d $N "$nxt" "$last" "$(date +%H:%M:%S)"
  { echo "updated $(date '+%F %T') UTC+8"; echo "job log: $LOG ; done $d/$N ; next: $nxt"; echo "last: $last";
    echo "running: $(pgrep -f 'm7_' | grep -v $$ | wc -l | tr -d ' ') m7 procs"; cat out/m7/PLAN_STATE.txt 2>/dev/null; } > out/m7/STATUS.txt
  grep -q $MK $LOG 2>/dev/null && { echo; echo DONE; break; }; sleep 10; done
