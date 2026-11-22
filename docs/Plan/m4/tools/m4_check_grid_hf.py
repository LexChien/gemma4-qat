"""Check an exported HF dir: quantized tensors lie on a 4-bit block-32 grid (absmax/8 or minmax), HF input axis."""
import sys, json, numpy as np
from safetensors import safe_open
d = sys.argv[1]; F = safe_open(d + "/model.safetensors", "pt")
Q = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "per_layer_input_gate.",
     "per_layer_projection.", "per_layer_model_projection", "embed_tokens.", "embed_tokens_per_layer")
def f32(a): return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32) if a.dtype == np.uint16 else a.astype(np.float32)
res = {}
for k in F.keys():
    if not any(s in k for s in Q): continue
    x = F.get_tensor(k).view(__import__("torch").int16).numpy().view(np.uint16)
    if "embed_tokens" in k: x = x[::64]
    b = f32(np.asarray(x)).reshape(-1, 32)
    best = 0
    for name, dd in (("absmax", np.abs(b).max(-1, keepdims=True) / 8), ("absmax7", np.abs(b).max(-1, keepdims=True) / 7),
                     ("minmax", np.maximum(-b.min(-1, keepdims=True) / 8, b.max(-1, keepdims=True) / 7))):
        L = np.round(b / np.where(dd == 0, 1, dd))
        dls = (b * L).sum(-1, keepdims=True) / np.maximum((L * L).sum(-1, keepdims=True), 1e-30)
        ok = (np.abs(b / np.where(dls == 0, 1, dls) - L).max(-1) < 0.05) & (L.min(-1) >= -8) & (L.max(-1) <= 7)
        best = max(best, float(ok.mean()))
    res[k] = best
vals = np.array(list(res.values()))
print(json.dumps({"dir": d, "n": len(res), "min_on_grid": float(vals.min()), "mean_on_grid": float(vals.mean()),
                  "worst": sorted(res.items(), key=lambda kv: kv[1])[:3]}, indent=1))
