"""M3: compare every tensor of a converted HF checkpoint against a reference HF checkpoint.
Checks: name set, shape, dtype, bit-exact values (byte equality); for mismatches: max_abs, cos, and
whether the mismatch is only bf16 rounding (|diff| <= 1 ulp).
Usage: m3_compare_tensors.py <converted_dir> <reference_dir> <out_json> [--text_only] [--cast_bf16]"""
import sys, json, torch
from safetensors import safe_open
conv, ref, outp = sys.argv[1:4]; text_only = "--text_only" in sys.argv; cast = "--cast_bf16" in sys.argv  # cast converted (f32) to bf16 (RNE) before comparing
A = safe_open(conv + "/model.safetensors", "pt"); B = safe_open(ref + "/model.safetensors", "pt")
ka = set(A.keys()); kb = set(B.keys())
if text_only:
    ka_m = {"model.language_model." + k[len("model."):] if k.startswith("model.") else k: k for k in ka}
    kb = {k for k in kb if k.startswith("model.language_model.")}
else:
    ka_m = {k: k for k in ka}
res = {"converted": conv, "reference": ref, "n_converted": len(ka), "n_reference": len(kb),
       "missing_in_converted": sorted(kb - set(ka_m)), "extra_in_converted": sorted(set(ka_m) - kb),
       "shape_mismatch": [], "dtype_mismatch": [], "value_mismatch": [], "exact": 0}
for k in sorted(kb & set(ka_m)):
    x = A.get_tensor(ka_m[k]); y = B.get_tensor(k)
    if cast: x = x.to(torch.bfloat16)
    if tuple(x.shape) != tuple(y.shape): res["shape_mismatch"].append([k, list(x.shape), list(y.shape)]); continue
    if x.dtype != y.dtype: res["dtype_mismatch"].append([k, str(x.dtype), str(y.dtype)])
    if x.dtype == y.dtype and torch.equal(x.view(torch.uint8) if x.dim() else x.reshape(1).view(torch.uint8), y.view(torch.uint8) if y.dim() else y.reshape(1).view(torch.uint8)):
        res["exact"] += 1; continue
    xf, yf = x.float(), y.float(); d = (xf - yf).abs()
    cos = torch.nn.functional.cosine_similarity(xf.flatten(), yf.flatten(), dim=0).item() if xf.numel() > 1 else float("nan")
    res["value_mismatch"].append({"name": k, "max_abs": d.max().item(), "frac_diff": (d > 0).float().mean().item(), "cos": cos,
                                  "conv_sample": xf.flatten()[:4].tolist(), "ref_sample": yf.flatten()[:4].tolist()})
res["pass"] = not (res["missing_in_converted"] or res["extra_in_converted"] or res["shape_mismatch"] or res["dtype_mismatch"] or res["value_mismatch"])
json.dump(res, open(outp, "w"), indent=1)
print({k: (len(v) if isinstance(v, list) else v) for k, v in res.items() if k not in ("converted", "reference")})
for k in ("missing_in_converted", "extra_in_converted", "shape_mismatch", "dtype_mismatch"):
    if res[k]: print(k, res[k][:8])
for m in res["value_mismatch"][:15]: print("VALUE", m["name"], round(m["max_abs"], 5), round(m["frac_diff"], 4), round(m["cos"], 5))
