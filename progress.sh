#!/bin/bash
cd "$(dirname "$0")"
TOTAL=$((67*1024*1024))
while true; do
  USED=$(du -sk weights 2>/dev/null | cut -f1)
  PCT=$((USED*100/TOTAL)); [ $PCT -gt 100 ] && PCT=100
  BAR=$(printf '%*s' $((PCT/2)) '' | tr ' ' '#'); PAD=$(printf '%*s' $((50-PCT/2)) '')
  GB=$(awk -v u=$USED 'BEGIN{printf "%.2f", u/1048576}')
  LIVE=$(pgrep -f "dl_all.py" >/dev/null && echo "下載中" || echo "已停止")
  INC=$(find weights -name "*.incomplete" -exec du -sk {} \; 2>/dev/null | awk '{s+=$1} END{printf "%.2fGB 未完成", s/1048576}')
  LAST=$(grep -E 'HF |GCS |DONE|ERROR' dl_all.log 2>/dev/null | tail -1 | cut -c1-45)
  printf "\r[%s%s] %3d%%  %s/67 GB  [%s] %s | %s   " "$BAR" "$PAD" $PCT "$GB" "$LIVE" "$LAST" "$INC"
  if grep -q "ALL DONE" dl_all.log 2>/dev/null; then echo; echo "✅ 全部下載完成"; break; fi
  if ! pgrep -f "dl_all.py" >/dev/null; then
    echo; echo "⚠️ 下載行程已停止"; break
  fi
  sleep 2
done
