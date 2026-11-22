"""M4: probe the quantization grid of official qat-q4_0-unquantized weights (blocks of 32 along HF input axis).
For each block: unique values, inferred step d (from min nonzero |diff| of sorted uniques), integer levels,
and which level range is used ([-8,7] like Q4_0, [-7,7] symmetric, ...); whether absmax maps to -8 / +8 / 7."""
import sys, json, torch, numpy as np
from safetensors import safe_open
f = safe_open("weights/gemma-4-E2B-it-qat-q4_0-unquantized/model.safetensors", "pt")
L = "model.language_model."
names = [L+"layers.0.self_attn.q_proj.weight", L+"layers.0.mlp.gate_proj.weight", L+"layers.0.mlp.down_proj.weight",
         L+"layers.20.mlp.up_proj.weight", L+"layers.4.self_attn.o_proj.weight", L+"layers.3.per_layer_input_gate.weight",
         L+"per_layer_model_projection.weight", L+"embed_tokens.weight", L+"embed_tokens_per_layer.weight"]
g = torch.Generator().manual_seed(0); res = {}
for n in names:
    sl = f.get_slice(n); shp = sl.get_shape()
    rows = torch.randint(0, shp[0], (256,), generator=g).sort().values.tolist() if shp[0] > 20000 else None
    w = (torch.stack([sl[r:r+1][0] for r in rows]) if rows else f.get_tensor(n)).float()
    b = w.reshape(-1, 32)[:4096] if rows is None else w.reshape(-1, 32)
    b = b[torch.randperm(b.shape[0], generator=g)[:4096]]
    stats = {"lv_min": [], "lv_max": [], "int_ok": 0, "absmax_is_neg8": 0, "absmax_is_pos8": 0, "absmax_lv": {}, "d_vs_absmax": []}
    for blk in b:
        u = torch.unique(blk)
        if len(u) < 2: continue
        dif = torch.diff(u); d = dif.min().item()
        lv = blk / d; lvr = lv.round()
        if (lv - lvr).abs().max().item() < 0.05 * max(1, lvr.abs().max().item()) :
            stats["int_ok"] += 1
        stats["lv_min"].append(int(lvr.min())); stats["lv_max"].append(int(lvr.max()))
        am = blk.abs().argmax(); a = int(lvr[am]); stats["absmax_lv"][a] = stats["absmax_lv"].get(a, 0) + 1
        stats["d_vs_absmax"].append(d / blk.abs().max().item())
    n_ = len(stats["lv_min"])
    out = {"blocks": n_, "int_ok_frac": stats["int_ok"] / n_,
           "lv_min_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(stats["lv_min"], return_counts=True))},
           "lv_max_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(stats["lv_max"], return_counts=True))},
           "absmax_level_hist": dict(sorted(stats["absmax_lv"].items(), key=lambda kv: -kv[1])[:6]),
           "d_over_absmax_median": float(np.median(stats["d_vs_absmax"])),
           "d_over_absmax_p10_p90": [float(np.percentile(stats["d_vs_absmax"], 10)), float(np.percentile(stats["d_vs_absmax"], 90))]}
    res[n.replace(L, "")] = out; print(n.replace(L, ""), out, flush=True)
json.dump(res, open("docs/Plan/m4/raw/qat_grid_probe.json", "w"), indent=1)
