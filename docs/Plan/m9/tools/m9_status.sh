#!/bin/bash
# Visible progress bar + out/m9/STATUS.txt (stage markers docs/Plan/m9/raw/*.DONE, training step/ETA). Runs until M9_ALL.DONE.
cd ~/Documents/AI/gemma4-qat; R=docs/Plan/m9/raw; ST="clean train select export eval"
while true; do d=0; cur=""; for s in $ST; do if [ -f $R/$s.DONE ]; then d=$((d+1)); elif [ -z "$cur" ]; then cur=$s; fi; done
  log=$(ls -t $R/*.log 2>/dev/null | head -1); last=$(tail -c 3000 "$log" 2>/dev/null | tr '\r' '\n' | grep -aE "factcheck|step|cand|CHOSEN|EXIT|Error|DONE|run|wiki" | tail -1 | cut -c1-150)
  eta=""; if [ "$cur" = train ]; then l=$(grep -aE "^step [0-9]+/[0-9]+ " $R/train.log | tail -1)
    if [ -n "$l" ]; then a=$(echo "$l" | sed -E 's#^step ([0-9]+)/([0-9]+).*s/step ([0-9.]+).*#\1 \2 \3#'); set -- $a
      eta="train $1/$2, ETA $(date -v+$(python3 -c "print(int(($2-$1)*$3))")S '+%m-%d %H:%M') UTC+8"; fi; fi
  ck=$(ls out/m9/ckpt/step_*.json 2>/dev/null | wc -l | tr -d ' ')
  p=$((d*40/5)); bar=$(printf '%*s' $p '' | tr ' ' '#')$(printf '%*s' $((40-p)) '' | tr ' ' '.')
  printf "\r[%s] stage %d/5 %-6s | %s | %s   " "$bar" $d "$cur" "${eta:-$last}" "$(date +%H:%M:%S)" | cut -c1-200
  { echo "updated $(date '+%F %T') UTC+8"; echo "stages done: $(cd $R; ls *.DONE 2>/dev/null | tr '\n' ' ')"; echo "current stage: $cur"; echo "progress: $eta"; echo "checkpoints: $ck (out/m9/ckpt, never pruned)"
    echo "log: $log"; echo "last: $last"; echo "resume: relaunch docs/Plan/m9/tools/m9_pipeline.sh in Terminal (osascript+caffeinate); stages skip by .DONE marker; training resumes from latest global checkpoint"; } > out/m9/STATUS.txt
  [ -f $R/M9_ALL.DONE ] && { echo; echo ALL DONE; break; }; sleep 15; done
