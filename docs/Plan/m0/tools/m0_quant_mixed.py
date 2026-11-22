"""M0 fixed copy of ../../../../quantize_mlx_mixed.py.
Fix: mlx-lm 0.32 calls quant_predicate(path, module) with 2 args (original script defines 3 -> TypeError).
MLX-internal paths: language_model.model.{embed_tokens,embed_tokens_per_layer,layers.N.mlp.*,...}
mlx-lm gemma4 sanitize() already drops vision/audio towers (text-only)."""
import argparse, re, json
p = argparse.ArgumentParser()
p.add_argument("--hf", required=True); p.add_argument("--out", required=True)
p.add_argument("--profile", choices=["safe", "mobile"], default="safe")
p.add_argument("--group", type=int, default=64)
a = p.parse_args()
stats = {}
def bits_for(path):
    if a.profile == "safe": return 4
    if re.search(r"embed_tokens_per_layer$", path): return 4
    if re.search(r"embed_tokens$|lm_head$", path): return 2
    if re.search(r"per_layer_(input_gate|projection)$", path): return 8
    m = re.search(r"layers\.(\d+)\.mlp\.", path)
    if m: return 4 if int(m.group(1)) < 15 else 2
    return 4
def predicate(path, module):
    b = bits_for(path); stats[path] = b
    # Google mobile-ct uses PLE group 256; MLX supports max 128 -> use 128
    g = 128 if (a.profile == "mobile" and path.endswith("embed_tokens_per_layer")) else a.group
    return {"bits": b, "group_size": g, "mode": "affine"}
from mlx_lm import convert
convert(hf_path=a.hf, mlx_path=a.out, quantize=True, q_bits=4, q_group_size=a.group, quant_predicate=predicate)
json.dump(stats, open(a.out + "/m0_bits_map.json", "w"), indent=0)
from collections import Counter; print("bits histogram:", Counter(stats.values()))
