#!/bin/bash
# visible progress bar for m6_post.sh (counts finished output files)
cd ~/Documents/AI/gemma4-qat/docs/Plan/m6/raw
L="bench_m6-mlxlora-g32exact bench_m6-mlxlora-g64 bench_m6-mlxlora-fused-bf16 bench_m6-qat-g32exact_speed1 bench_m6-qat-g32exact_speed2 bench_m6-jaxlora-g32exact_speed1 bench_m6-jaxlora-g32exact_speed2 bench_m6-mlxlora-g32exact_speed1 bench_m6-mlxlora-g32exact_speed2 lat_qat_off_summary lat_qat_on_summary lat_jaxlora_off_summary lat_jaxlora_on_summary lat_mlxlora_off_summary lat_mlxlora_on_summary lat_lever_kv8_off_summary lat_lever_kv4_off_summary lat_lever_g64_off_summary lat_lever_budget128_on_summary lat_lever_budget256_on_summary lever_prompt_cache_qat"
N=$(echo $L | wc -w)
while true; do d=0; nxt=""; for f in $L; do if [ -f $f.json ]; then d=$((d+1)); elif [ -z "$nxt" ]; then nxt=$f; fi; done
  p=$((d*40/N)); bar=$(printf '%*s' $p '' | tr ' ' '#')$(printf '%*s' $((40-p)) '' | tr ' ' '.')
  cur=$(ls -t lat_*.jsonl chat_*.jsonl 2>/dev/null | head -1); n=$([ -n "$cur" ] && wc -l < $cur | tr -d ' ')
  printf "\r[%s] %d/%d  next: %-32s (%s: %s lines) %s " "$bar" $d $N "$nxt" "$cur" "$n" "$(date +%H:%M:%S)"
  { echo "updated $(date '+%F %T') UTC+8"; echo "pipeline: docs/Plan/m6/tools/m6_post.sh (resumable: each step skips if its output exists; relaunch it to resume)"; echo "done $d/$N; next: $nxt; current: $cur ($n lines)"; echo "post.sh running: $(pgrep -f m6_post.sh >/dev/null && echo yes || echo NO)"; echo "remaining after pipeline: write docs/Plan/06_mlx_lora_chat_M6.md, docs/logs/2026-10-17_M6.md, indexes"; } > ../../../../out/m6/STATUS.txt
  grep -q M6_POST_DONE post.log && { echo; echo DONE; break; }; sleep 10; done
