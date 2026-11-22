#!/bin/bash
# M3 step 2: pt bf16, it f32, it text_only, it drop_shared_kv; compare each; file hashes
cd /Users/lexchien/Documents/AI/gemma4-qat
PY=.venv-m0/bin/python; RAW=docs/Plan/m3/raw; C=docs/Plan/m3/tools/m3_compare_tensors.py; CV=tools/convert_jax_to_hf.py
O=weights/orbax/checkpoints
{
echo "=== convert pt bf16 $(date)"; rm -rf out/m3/pt-bf16
$PY $CV --orbax $O/gemma4-e2b-pt --out out/m3/pt-bf16 2>&1 | tail -1
$PY $C out/m3/pt-bf16 weights/gemma-4-E2B $RAW/compare_pt_bf16.json 2>&1 | grep -v Redirect
echo "=== convert it f32 $(date)"; rm -rf out/m3/it-f32
$PY $CV --orbax $O/gemma4-e2b-it --out out/m3/it-f32 --dtype f32 2>&1 | tail -1
$PY $C out/m3/it-f32 weights/gemma-4-E2B-it $RAW/compare_it_f32_vs_bf16ref_cast.json --cast_bf16 2>&1 | grep -v Redirect
echo "=== convert it text_only $(date)"; rm -rf out/m3/it-text-bf16
$PY $CV --orbax $O/gemma4-e2b-it --out out/m3/it-text-bf16 --text_only 2>&1 | tail -1
$PY $C out/m3/it-text-bf16 weights/gemma-4-E2B-it $RAW/compare_it_text_only.json --text_only 2>&1 | grep -v Redirect
echo "=== convert it drop_shared_kv $(date)"; rm -rf out/m3/it-dropkv-bf16
$PY $CV --orbax $O/gemma4-e2b-it --out out/m3/it-dropkv-bf16 --drop_shared_kv 2>&1 | tail -1
$PY - <<'PYEOF'
import json
from safetensors import safe_open
a=set(safe_open("out/m3/it-dropkv-bf16/model.safetensors","pt").keys()); q=set(safe_open("weights/gemma-4-E2B-it-qat-q4_0-unquantized/model.safetensors","pt").keys())
r={"n_dropkv":len(a),"n_qat":len(q),"only_dropkv":sorted(a-q),"only_qat":sorted(q-a),"name_set_equal":a==q}
json.dump(r,open("docs/Plan/m3/raw/dropkv_vs_qat_names.json","w"),indent=1); print({k:(v if not isinstance(v,list) else len(v)) for k,v in r.items()})
PYEOF
echo "=== file sha1 $(date)"
shasum out/m3/it-bf16/model.safetensors weights/gemma-4-E2B-it/model.safetensors out/m3/pt-bf16/model.safetensors weights/gemma-4-E2B/model.safetensors
echo "M3_REST_DONE $(date)"
} > $RAW/run_rest.log 2>&1
