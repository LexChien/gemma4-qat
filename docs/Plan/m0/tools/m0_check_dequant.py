"""M0 sanity: compare dequantized official CT weights against bf16 qat-q4_0-unquantized / E2B-it."""
import sys, json, torch
from safetensors import safe_open
a = safe_open(sys.argv[1] + "/model.safetensors", "pt")
refs = {n: safe_open(f"weights/{n}/model.safetensors", "pt") for n in ["gemma-4-E2B-it-qat-q4_0-unquantized", "gemma-4-E2B-it"]}
keys = ["model.language_model.layers.0.self_attn.q_proj.weight", "model.language_model.layers.20.mlp.down_proj.weight",
        "model.language_model.layers.34.mlp.up_proj.weight", "model.language_model.embed_tokens.weight",
        "model.language_model.embed_tokens_per_layer.weight", "model.language_model.layers.3.per_layer_input_gate.weight"]
res = {}
for k in keys:
    x = a.get_tensor(k).float()
    for n, r in refs.items():
        if k not in r.keys(): continue
        y = r.get_tensor(k).float()
        d = (x - y).abs()
        res[f"{k} vs {n}"] = {"exact_frac": (d == 0).float().mean().item(), "max_abs": d.max().item(),
                              "rel_rms": (d.pow(2).mean().sqrt() / y.pow(2).mean().sqrt()).item()}
for k, v in res.items(): print(k, {kk: round(vv, 5) for kk, vv in v.items()})
json.dump(res, open(sys.argv[2], "w"), indent=1)
