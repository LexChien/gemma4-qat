"""M4: test candidate Q4 block formulas against official qat-q4_0-unquantized blocks (exact bf16 reconstruction)."""
import json, torch
from safetensors import safe_open
f = safe_open("weights/gemma-4-E2B-it-qat-q4_0-unquantized/model.safetensors", "pt"); L = "model.language_model."
names = ["layers.0.self_attn.q_proj.weight", "layers.0.mlp.gate_proj.weight", "layers.0.mlp.down_proj.weight", "layers.4.self_attn.o_proj.weight",
         "layers.3.per_layer_input_gate.weight", "per_layer_model_projection.weight", "embed_tokens.weight", "embed_tokens_per_layer.weight"]
def cands(b):
    mn = b.min(1, keepdim=True).values; mx = b.max(1, keepdim=True).values
    am = b.abs().max(1, keepdim=True).values
    signed = torch.gather(b, 1, b.abs().argmax(1, keepdim=True))
    out = {"repo_q4_0 (d=-m/8, fp16 d)": (-signed / 8).half().float(),
           "d=max(-min/8,max/7) f32": torch.maximum(-mn / 8, mx / 7),
           "d=max(-min/8,max/7) fp16": torch.maximum(-mn / 8, mx / 7).half().float(),
           "d=max(-min/8,max/7) bf16": torch.maximum(-mn / 8, mx / 7).bfloat16().float(),
           "d=absmax/8": am / 8, "d=absmax/7": am / 7}
    return out
res = {}
for n in names:
    sl = f.get_slice(L + n); shp = sl.get_shape()
    w = (sl[0:64] if shp[0] > 20000 else f.get_tensor(L + n)).float()
    b = w.reshape(-1, 32)[:20000]
    r = {}
    for cn, d in cands(b).items():
        d = torch.where(d == 0, torch.ones_like(d), d)
        q = torch.clamp(torch.round(b / d), -8, 7) if "repo" not in cn else torch.clamp(torch.trunc(b / d + 8.5), 0, 15) - 8
        rec = (q * d).bfloat16().float()
        r[cn] = round((rec == b).all(1).float().mean().item(), 4)
    res[n] = r; print(n, r, flush=True)
json.dump(res, open("docs/Plan/m4/raw/qat_grid_formula.json", "w"), indent=1)
