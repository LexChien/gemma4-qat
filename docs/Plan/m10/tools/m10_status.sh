#!/bin/bash
# Progress line + out/m10/STATUS.txt for all M10 tasks (markers docs/Plan/m10/raw/*.DONE)
cd ~/Documents/AI/gemma4-qat; R=docs/Plan/m10/raw
ST="ct_export ct32_export hf_m9ct hf_offct hf_m9ct32 T12_ALL"
while true; do d=0; cur=""; for s in $ST; do if [ -f $R/$s.DONE ]; then d=$((d+1)); elif [ -z "$cur" ]; then cur=$s; fi; done; n=$(echo $ST | wc -w)
  log=$(ls -t $R/*.log 2>/dev/null | head -1); last=$(tail -c 2000 "$log" 2>/dev/null | tr '\r' '\n' | grep -av "^\s*$" | tail -1 | cut -c1-140)
  p=$((d*40/n)); bar=$(printf '%*s' $p '' | tr ' ' '#')$(printf '%*s' $((40-p)) '' | tr ' ' '.')
  printf "\r[%s] %d/%d %-11s | %s | %s   " "$bar" $d $n "$cur" "$last" "$(date +%H:%M:%S)" | cut -c1-220
  { echo "updated $(date '+%F %T') UTC+8"; echo "task 1-2 stages done: $(cd $R; ls *.DONE 2>/dev/null | tr '\n' ' ')"; echo "current: $cur"; echo "log: $log"; echo "last: $last"
    echo "resume: relaunch docs/Plan/m10/tools/m10_t12_pipeline.sh in Terminal (osascript+caffeinate); finished stages are skipped"; echo "next (not started): refactor B+C (src/g4qat), then G6 PEFT NaN, M10 training, clean-system latency script"; } > out/m10/STATUS.txt
  [ -f $R/T12_ALL.DONE ] && { echo; echo T12 DONE; break; }; sleep 15; done
