"""M4: refined grid test. Per block: d0=min positive gap of unique values; q=round(b/d0); refine d=LS fit; check
exact bf16 reconstruction bf16(q*d)==b, level range, and whether block hits -8 or +7 (minimal covering scale)."""
import json, torch
from safetensors import safe_open
f = safe_open("weights/gemma-4-E2B-it-qat-q4_0-unquantized/model.safetensors", "pt"); L = "model.language_model."
names = ["layers.0.self_attn.q_proj.weight", "layers.0.mlp.gate_proj.weight", "layers.0.mlp.down_proj.weight", "layers.4.self_attn.o_proj.weight",
         "layers.17.self_attn.q_proj.weight", "layers.3.per_layer_input_gate.weight", "per_layer_model_projection.weight", "embed_tokens.weight", "embed_tokens_per_layer.weight"]
res = {}
for n in names:
    sl = f.get_slice(L + n); shp = sl.get_shape()
    w = (sl[0:64] if shp[0] > 20000 else f.get_tensor(L + n)).float(); b = w.reshape(-1, 32)[:8192]
    ex = hit8 = hit7 = neither = inrange = 0; ratio = []
    for blk in b:
        u = torch.unique(blk); d0 = torch.diff(u).min()
        q = torch.round(blk / d0)
        d = (q * blk).sum() / (q * q).sum()
        q = torch.round(blk / d)
        rec = (q * d).bfloat16().float()
        # try neighbouring d values (bf16 rounding of products tolerates a band of d) -> exactness test
        ok = bool((rec == blk).all())
        if not ok:
            for eps in torch.linspace(-0.004, 0.004, 41):
                dd = d * (1 + eps)
                if bool(((q * dd).bfloat16().float() == blk).all()): ok = True; break
        ex += ok; lo, hi = int(q.min()), int(q.max())
        inrange += (lo >= -8 and hi <= 7); hit8 += lo == -8; hit7 += hi == 7; neither += (lo > -8 and hi < 7)
        ratio.append((d / blk.abs().max()).item())
    N = len(b); r = {"blocks": N, "exact_bf16_recon": ex / N, "levels_in_[-8,7]": inrange / N, "hits_-8": hit8 / N, "hits_+7": hit7 / N,
                     "neither_(not_minimal_scale)": neither / N, "d/absmax_median": float(torch.tensor(ratio).median())}
    res[n] = r; print(n, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
json.dump(res, open("docs/Plan/m4/raw/qat_grid_refine.json", "w"), indent=1)
