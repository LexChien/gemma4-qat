#!/bin/sh
# M11: run the ORIGINAL gemma.cpp converter (same script, same bindings, same args as
# ALife-Platform/tools/convert_gemma4_chat.sh) with a custom input/output. Never overwrites.
# usage: m11_convert.sh <tag> <model.safetensors> <out.sbs> <out.csv>
set -eu
TAG=$1; IN=$2; OUT=$3; CSV=$4
ALIFE=~/Documents/AI/ALife-Platform
ASSETS=$ALIFE/Google-Gemma4
STATUS=~/Documents/AI/gemma4-qat/out/m11/STATUS.txt
test ! -e "$OUT" || { echo "refuse overwrite $OUT" >&2; exit 1; }
echo "$(date '+%F %T %Z') stage=convert:$TAG pct=0 start" >> "$STATUS"
export GEMMA_CPP_BUILD_DIR="$ASSETS/runtime/bindings"
"$ALIFE/.venv/bin/python" -u "$ASSETS/work/source/gemma.cpp/python/convert_from_safetensors.py" \
  --model_specifier=gemma4-2b-lm-q4_0-it \
  --load_path="$IN" --tokenizer_file="$ASSETS/originals/models/gemma4-e2b-qat/tokenizer.json" \
  --sbs_file="$OUT.partial" --metadata_file="$CSV" 2>&1 | \
  awk -v S="$STATUS" -v T="$TAG" '{print; fflush()} /^Writing/{n++; if(n%50==0){printf "stage=convert:%s pct=%d tensors=%d/~495\n", T, n*100/495, n >> S; fflush(S)}}'
mv "$OUT.partial" "$OUT"
shasum -a 256 "$OUT" > "$OUT.sha256"
echo "$(date '+%F %T %Z') stage=convert:$TAG pct=100 done $(cat "$OUT.sha256")" >> "$STATUS"
