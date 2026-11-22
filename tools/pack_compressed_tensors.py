#!/usr/bin/env python
"""Pack a Gemma 4 E2B text-only HF dir (bf16, weights on the symmetric Q4 block-32 grid, e.g. official QAT or our
QAT/LoRA-QAT export) as compressed-tensors `pack-quantized` W4A16 g32 (Gemma4ForCausalLM, keys model.*).
Same tensor layout as the official gemma-4-E2B-it-qat-w4a16-ct: weight_packed int32 [out, in/8] (nibble i = bits 4i,
value = level+8), weight_scale bf16 [out, in/32], weight_shape int64 [2]; same 276 Linear targets; embeddings/PLE
table/norms stay bf16. Unlike the official ct (scale = absmax/7.5, a re-quantization), the scale here is the QAT
block scale d, so the dequantized weights reproduce the QAT grid exactly (up to bf16 rounding of d).
--snap: source is off-grid -> Q4_0_MINMAX PTQ.  Usage: python tools/pack_compressed_tensors.py --hf DIR --out DIR [--snap]"""
import argparse, json, os, shutil
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
T = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
     "mlp.down_proj", "per_layer_input_gate", "per_layer_projection")
def fit(w, snap):
    b = w.float().reshape(-1, 32)
    d0 = torch.maximum(-b.min(-1, keepdim=True).values / 8, b.max(-1, keepdim=True).values / 7)
    L = torch.clamp(torch.round(b / torch.where(d0 == 0, torch.ones_like(d0), d0)), -8, 7)
    if snap:
        d = d0.to(torch.bfloat16).float(); L = torch.clamp(torch.round(b / torch.where(d == 0, torch.ones_like(d), d)), -8, 7)
    else:
        d = (b * L).sum(-1, keepdim=True) / torch.clamp((L * L).sum(-1, keepdim=True), min=1e-30)
    res = (b / torch.where(d == 0, torch.ones_like(d), d) - L).abs().max(-1).values
    return L, d, float((res < 0.05).float().mean())
def pack(L, shape):
    q = (L + 8).to(torch.int64).reshape(shape[0], -1, 8)
    v = torch.zeros(q.shape[:2], dtype=torch.int64)
    for i in range(8): v |= q[..., i] << (4 * i)
    v = torch.where(v >= 2**31, v - 2**32, v)
    return v.to(torch.int32)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hf", required=True); ap.add_argument("--out", required=True); ap.add_argument("--snap", action="store_true")
    a = ap.parse_args(); f = safe_open(os.path.join(a.hf, "model.safetensors"), "pt"); keys = list(f.keys())
    pre = "model.language_model." if any(k.startswith("model.language_model.") for k in keys) else "model."
    out = {}; rep = {"snap": a.snap, "packed": 0, "min_on_grid": 1.0, "max_abs_err": 0.0}
    for k in keys:
        if not k.startswith(pre): continue
        nk = "model." + k[len(pre):]; t = f.get_tensor(k); base = nk[:-len(".weight")]
        if nk.endswith(".weight") and (any(base.endswith(x) for x in T) or base.endswith("per_layer_model_projection")) and t.ndim == 2:
            L, d, og = fit(t, a.snap); s = d.to(torch.bfloat16)
            out[base + ".weight_packed"] = pack(L, t.shape); out[base + ".weight_scale"] = s.reshape(t.shape[0], -1)
            out[base + ".weight_shape"] = torch.tensor(list(t.shape), dtype=torch.int64)
            rec = (L.reshape(t.shape[0], -1, 32) * s.float().reshape(t.shape[0], -1, 1)).reshape(t.shape)
            rep["max_abs_err"] = max(rep["max_abs_err"], float((rec - t.float()).abs().max())); rep["min_on_grid"] = min(rep["min_on_grid"], og); rep["packed"] += 1
        else:
            out[nk] = t
    os.makedirs(a.out, exist_ok=True); save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(a.hf, "config.json"))); cfg = dict(cfg.get("text_config", cfg)); cfg["architectures"] = ["Gemma4ForCausalLM"]
    cfg["quantization_config"] = {"config_groups": {"group_0": {"format": "pack-quantized", "input_activations": None, "output_activations": None,
        "targets": ["Linear"], "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": 32, "num_bits": 4,
        "observer": "memoryless_minmax", "observer_kwargs": {}, "strategy": "group", "symmetric": True, "type": "int"}}},
        "format": "pack-quantized", "ignore": ["lm_head"], "quant_method": "compressed-tensors", "quantization_status": "compressed",
        "kv_cache_scheme": None, "sparsity_config": {}, "global_compression_ratio": None}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for x in os.listdir(a.hf):
        if x.endswith((".json", ".jinja")) and x not in ("config.json", "model.safetensors.index.json", "processor_config.json", "convert_report.json", "exact_grid_report.json"):
            shutil.copy(os.path.join(a.hf, x), os.path.join(a.out, x))
    rep["size_gb"] = round(os.path.getsize(os.path.join(a.out, "model.safetensors")) / 1e9, 3)
    json.dump(rep, open(os.path.join(a.out, "pack_report.json"), "w"), indent=1); print(json.dumps(rep))
if __name__ == "__main__":
    main()
