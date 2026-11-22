#!/bin/sh
# usage: m12_make_trim.sh <tag m9|m8> <vocab_dir>   -> out/m12/gemma4-e2b-<tag>-v<N>-q4_0.sbs (+.sha256), verify json
set -eu
cd /Users/lexchien/Documents/AI/gemma4-qat
T=$1; VD=$2; N=$(basename $VD | tr -dc 0-9); W=out/m12/work; R=docs/Plan/m12/raw; PY=.venv-m0/bin/python; ST=docs/Plan/m12/tools/status.sh
OUT=out/m12/gemma4-e2b-$T-v$N${M12_EMB_BF16:+-embbf16}-q4_0.sbs
[ -e $W/$T-hf-v$N.safetensors ] || $PY docs/Plan/m12/tools/m12_trim_hf.py out/m11/work/$T-hf-text.safetensors $VD/keep.npy $W/$T-hf-v$N.safetensors > $R/trim_hf_$T.log 2>&1
$ST 6 9 "stage6 $T: trimmed HF safetensors done; converter running"
[ -e $W/$T-v$N${M12_EMB_BF16:+-embbf16}-pipeline.sbs ] || M12_Q4=${M12_Q4:-c_embedding,ple_model_proj} M12_VOCAB=$N /Users/lexchien/Documents/AI/ALife-Platform/.venv/bin/python -u docs/Plan/m12/tools/m12_convert.py $W/$T-hf-v$N.safetensors $VD/tokenizer.json $W/$T-v$N${M12_EMB_BF16:+-embbf16}-pipeline.sbs $W/$T-v$N${M12_EMB_BF16:+-embbf16}-pipeline.csv > $R/convert_${T}_v$N${M12_EMB_BF16:+-embbf16}.log 2>&1
$ST 6 9 "stage6 $T: converter done; exact patch running"
[ -e $OUT ] || $PY docs/Plan/m12/tools/m12_exact.py patch out/$T/mlx-$T-g32exact $W/$T-v$N${M12_EMB_BF16:+-embbf16}-pipeline.sbs $OUT $VD/keep.npy > $R/patch_${T}_v$N${M12_EMB_BF16:+-embbf16}.log 2>&1
$PY docs/Plan/m12/tools/m12_exact.py verify out/$T/mlx-$T-g32exact $OUT $R/verify_${T}_v$N${M12_EMB_BF16:+-embbf16}.json $VD/keep.npy > $R/verify_${T}_v$N${M12_EMB_BF16:+-embbf16}.log 2>&1
(cd out/m12 && shasum -a 256 $(basename $OUT) > $(basename $OUT).sha256)
$ST 6 9 "stage6 $T: $(basename $OUT) $(stat -f %z $OUT) B PASS=$(grep -o '"PASS": [a-z]*' $R/verify_${T}_v$N${M12_EMB_BF16:+-embbf16}.json) sha256=$(cut -c1-16 $OUT.sha256)…"
echo MAKE_DONE $OUT
