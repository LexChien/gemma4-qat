"""PRIMARY <3GB route on Mac: MLX mixed-precision quantization modelled on Google's
gemma-4-E2B-it-qat-mobile-* recipe (verified from their config.json, see README).
Uses mlx_lm.convert(quant_predicate=...) Python API (exists in recent mlx-lm; Gemma 4 support in
mlx-lm UNVERIFIED). Module names below follow HF checkpoint names; MLX-internal names may differ
-> run with --dry to print what each module gets.
  --profile safe   : all 4-bit g64 (embeddings + PLE incl.)      est. ~2.6GB text-only
  --profile mobile : Google-mobile-like: embed/lm_head 2-bit, MLP layers>=15 2-bit,
                     PLE 4-bit, attn + MLP 0-14 4-bit, per-layer gate/proj 8-bit   est. ~2.0-2.3GB
"""
import argparse, re
p = argparse.ArgumentParser()
p.add_argument("--hf", required=True); p.add_argument("--out", default="out/mlx_mixed")
p.add_argument("--profile", choices=["safe", "mobile"], default="safe")
p.add_argument("--group", type=int, default=64); p.add_argument("--dry", action="store_true")
a = p.parse_args()

def bits_for(path: str):
    if re.search(r"vision|audio|embed_vision|embed_audio", path): return None  # strip towers (text-only)
    if a.profile == "safe": return 4
    if re.search(r"embed_tokens_per_layer", path): return 4
    if re.search(r"embed_tokens$|lm_head$", path): return 2
    if re.search(r"per_layer_(input_gate|projection)$", path): return 8
    m = re.search(r"layers\.(\d+)\.mlp\.", path)
    if m: return 4 if int(m.group(1)) < 15 else 2
    return 4

def predicate(path, module, config):
    if not hasattr(module, "to_quantized"): return False
    b = bits_for(path)
    if a.dry: print(path, b)
    return False if b is None else {"bits": b, "group_size": a.group}

if __name__ == "__main__":
    from mlx_lm import convert
    convert(hf_path=a.hf, mlx_path=a.out, quantize=True, q_bits=4, q_group_size=a.group,
            quant_predicate=predicate)
