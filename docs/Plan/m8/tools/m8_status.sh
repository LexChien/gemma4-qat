#!/bin/bash
# Visible progress line + out/m8/STATUS.txt (stage markers in docs/Plan/m8/raw/*.DONE). Runs until M8_ALL.DONE.
cd /Users/lexchien/Documents/AI/gemma4-qat; R=docs/Plan/m8/raw
ST="distill sft train export eval"
while true; do d=0; cur=""; for s in $ST; do if [ -f $R/$s.DONE ]; then d=$((d+1)); elif [ -z "$cur" ]; then cur=$s; fi; done
  log=$(ls -t $R/*.log 2>/dev/null | head -1); last=$(tail -c 2000 "$log" 2>/dev/null | tr '\r' '\n' | grep -aE "distill|step|iter|EXIT|Error|DONE|lat|run" | tail -1 | cut -c1-110)
  p=$((d*40/5)); bar=$(printf '%*s' $p '' | tr ' ' '#')$(printf '%*s' $((40-p)) '' | tr ' ' '.')
  printf "\r[%s] stage %d/5 %-8s | %s | %s   " "$bar" $d "$cur" "$last" "$(date +%H:%M:%S)"
  { echo "updated $(date '+%F %T') Asia/Taipei"; echo "stages done: $(cd $R; ls *.DONE 2>/dev/null | tr '\n' ' ')"; echo "current stage: $cur"; echo "log: $log"; echo "last: $last";
    echo "resume: relaunch docs/Plan/m8/tools/m8_pipeline.sh (each stage skips if its .DONE marker exists; training resumes from latest global checkpoint)"; } > out/m8/STATUS.txt
  [ -f $R/M8_ALL.DONE ] && { echo; echo ALL DONE; break; }; sleep 10; done
