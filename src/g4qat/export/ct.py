"""G3: MLX exact-grid pack (affine 4-bit g32, bias = -8d) -> compressed-tensors `pack-quantized` HF text checkpoint.
  decoders + per_layer_* + per_layer_model_projection : int4 symmetric group 32  (q = L-8, scale = d  -> LOSSLESS when bias == -8d)
  embed_tokens + lm_head (untied copy, same data)     : int4 symmetric group 32  (lossless, same rule)
  embed_tokens_per_layer (PLE)                         : int4 symmetric group 256 (requantized, like official qat-mobile-ct; MSE scale search)
Usage: g4qat export-ct SRC_MLX_DIR OUT_DIR [--ple-group 256|32]
       python -m g4qat.export_ct SRC_MLX_DIR OUT_DIR [--ple-group 256|32]"""
import argparse, json, os, shutil, sys, time
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32
from g4qat.grid import unpack_mlx, requant_sym
HF_TEMPLATE = "out/m5/qat-text-hf"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--ple-group", type=int, default=256); a = ap.parse_args()
    t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    f = safe_open(os.path.join(a.src, "model.safetensors"), "pt"); keys = list(f.keys())
    tmpl = set(safe_open(os.path.join(HF_TEMPLATE, "model.safetensors"), "np").keys())
    bases = sorted({k[:-len(".scales")] for k in keys if k.endswith(".scales")})
    out, rep = {}, dict(src=a.src, ple_group=a.ple_group, tensors={}, nonsym_groups=0)
    def hf(k): return k.replace("language_model.model.", "model.", 1)
    g32, ple = [], None
    for b in bases:
        L = unpack_mlx(f.get_tensor(b + ".weight")); s = f.get_tensor(b + ".scales"); bi = f.get_tensor(b + ".biases")
        rows, cols = L.shape; name = hf(b)
        sym = ((-8 * s.float()).to(torch.bfloat16) == bi)
        if name.endswith("embed_tokens_per_layer") and a.ple_group != 32:
            q = torch.empty(rows, cols, dtype=torch.int8); sc = torch.empty(rows, cols // a.ple_group, dtype=torch.bfloat16); err = 0.0; ref = 0.0
            for r0 in range(0, rows, 8192):
                wf = (L[r0:r0 + 8192].float().reshape(-1, cols // 32, 32) * s[r0:r0 + 8192].float()[..., None] + bi[r0:r0 + 8192].float()[..., None]).reshape(-1, cols)
                qq, ss = requant_sym(wf, a.ple_group); q[r0:r0 + 8192] = qq; sc[r0:r0 + 8192] = ss
                err += ((qq.float().reshape(-1, cols // a.ple_group, a.ple_group) * ss.float()[..., None]).reshape(-1, cols) - wf).pow(2).sum().item(); ref += wf.pow(2).sum().item()
            ple = name; gs = a.ple_group; rep["tensors"][name] = dict(group=gs, lossless=False, rel_rmse=round((err / ref) ** 0.5, 5))
        else:
            ns = int((~sym).sum()); rep["nonsym_groups"] += ns
            if ns == 0: q = (L.to(torch.int16) - 8).to(torch.int8); sc = s
            else:
                wf = (L.float().reshape(rows, cols // 32, 32) * s.float()[..., None] + bi.float()[..., None]).reshape(rows, cols); q, sc = requant_sym(wf, 32)
            gs = 32; rep["tensors"][name] = dict(group=32, lossless=ns == 0, nonsym=ns)
            if name.endswith("embed_tokens_per_layer"): ple = name
            else: g32.append(name)
        pk = pack_to_int32(q, 4)
        assert torch.equal(unpack_from_int32(pk, 4, torch.Size([rows, cols])), q), name
        out[name + ".weight_packed"] = pk; out[name + ".weight_scale"] = sc.contiguous(); out[name + ".weight_shape"] = torch.tensor([rows, cols], dtype=torch.int64)
        assert name + ".weight" in tmpl, name
        if name == "model.embed_tokens":  # untied lm_head copy: transformers cannot tie lm_head.weight to a packed embedding
            out["lm_head.weight_packed"] = pk.clone(); out["lm_head.weight_scale"] = sc.contiguous().clone(); out["lm_head.weight_shape"] = torch.tensor([rows, cols], dtype=torch.int64)
        print(f"{name} {rows}x{cols} g{gs} {rep['tensors'][name]} {time.time() - t0:.0f}s", flush=True)
    for k in keys:
        if k.endswith((".scales", ".biases")) or k[:-len(".weight")] in bases: continue
        out[hf(k)] = f.get_tensor(k); assert hf(k) in tmpl, k
    missing = sorted(t for t in tmpl if t not in out and t[:-len(".weight")] not in set(map(hf, bases)))
    assert not missing, missing[:5]
    W = lambda gsz: dict(num_bits=4, type="int", symmetric=True, strategy="group", group_size=gsz, dynamic=False, observer="memoryless_minmax", actorder=None, block_structure=None, observer_kwargs={})
    emb = [n for n in g32 if n.endswith("embed_tokens")] + ["lm_head"]; lin = [n for n in g32 if n not in emb]
    qc = dict(quant_method="compressed-tensors", format="pack-quantized", quantization_status="compressed", ignore=[], kv_cache_scheme=None, global_compression_ratio=None, version="0.19.0",
              config_groups=dict(group_0=dict(targets=emb, weights=W(32), input_activations=None, output_activations=None, format="pack-quantized"),
                                 group_1=dict(targets=[ple], weights=W(rep["tensors"][ple]["group"]), input_activations=None, output_activations=None, format="pack-quantized"),
                                 group_2=dict(targets=lin, weights=W(32), input_activations=None, output_activations=None, format="pack-quantized")))
    cfg = json.load(open(os.path.join(HF_TEMPLATE, "config.json"))); cfg["quantization_config"] = qc; cfg["tie_word_embeddings"] = False  # lm_head stored as its own (identical) int4 g32 tensor, like official qat-mobile-ct
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        p = os.path.join(a.src, fn) if os.path.exists(os.path.join(a.src, fn)) else os.path.join(HF_TEMPLATE, fn)
        if os.path.exists(p): shutil.copy(p, a.out)
    rep["size_bytes"] = sum(os.path.getsize(os.path.join(a.out, x)) for x in os.listdir(a.out)); rep["n_g32"] = len(g32); rep["elapsed_s"] = round(time.time() - t0, 1)
    rep["lossless_except_ple"] = all(v["lossless"] for k, v in rep["tensors"].items() if k != ple)
    json.dump(rep, open(os.path.join(a.out, "g4qat_ct_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k != "tensors"}), "PLE", rep["tensors"][ple], flush=True); print("EXPORT_CT_DONE")
if __name__ == "__main__": main()
