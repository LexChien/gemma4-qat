"""M0: weight-only dequantization of official compressed-tensors checkpoints -> dense bf16 HF dir,
so they can be evaluated with the SAME MLX protocol (transformers 5.20.1 + compressed-tensors 0.19.0
fail to load them: Gemma4TextScaledWordEmbedding has no attribute 'weight' in _init_weights).
Supports symmetric int pack-quantized (channel / group) and int-quantized int8 (channel).
NOTE: static int8 activation quant (input_scale/output_scale, mobile-ct) is DROPPED -> weight-only emulation.
Usage: m0_ct_dequant.py <ct_dir> <out_dir>"""
import sys, os, json, glob, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
files = sorted(glob.glob(src + "/*.safetensors"))
T = {}
for f in files:
    h = safe_open(f, "pt")
    for k in h.keys(): T[k] = (f, k)
get = lambda k: safe_open(T[k][0], "pt").get_tensor(k)
out = {}; stats = {}
handles = {f: safe_open(f, "pt") for f in files}
get = lambda k: handles[T[k][0]].get_tensor(k)
for k in sorted(T):
    if k.endswith((".input_scale", ".output_scale", ".weight_shape", ".weight_scale", ".weight_zero_point")): continue
    if k.endswith(".weight_packed"):
        p = k[:-len(".weight_packed")]
        packed = get(k); shape = get(p + ".weight_shape").tolist(); scale = get(p + ".weight_scale").float()
        assert p + ".weight_zero_point" not in T
        bits = packed.shape[-1] * 32 // shape[-1]
        while packed.shape[-1] * 32 // bits * 0 + (-(-shape[-1] * bits // 32)) != packed.shape[-1]: bits += 1
        q = unpack_from_int32(packed, bits, torch.Size(shape)).float()
        ng = scale.shape[-1]
        if ng == 1: w = q * scale
        else: w = (q.view(shape[0], ng, shape[1] // ng) * scale[:, :, None]).view(shape)
        out[p + ".weight"] = w.to(torch.bfloat16)
        stats.setdefault(f"int{bits}_{'channel' if ng == 1 else 'g%d' % (shape[1] // ng)}", []).append(p)
    elif k.endswith(".weight") and (k[:-7] + ".weight_scale") in T and get(k).dtype == torch.int8:
        w = get(k).float() * get(k[:-7] + ".weight_scale").float()
        out[k] = w.to(torch.bfloat16); stats.setdefault("int8_channel", []).append(k[:-7])
    else:
        out[k] = get(k)
# lm_head.weight is stored but bit-identical to embed_tokens (verified) -> drop it and mark tied (MLX/HF tie)
tied = "lm_head.weight" in out and torch.equal(out["lm_head.weight"], out["model.language_model.embed_tokens.weight"])
if tied: out.pop("lm_head.weight")
save_file(out, dst + "/model.safetensors", metadata={"format": "pt"})
cfg = json.load(open(src + "/config.json")); cfg.pop("quantization_config", None)
if tied:
    cfg["tie_word_embeddings"] = True
    if "text_config" in cfg: cfg["text_config"]["tie_word_embeddings"] = True
json.dump(cfg, open(dst + "/config.json", "w"), indent=2)
for f in os.listdir(src):
    if f.endswith((".json", ".jinja")) and f not in ("config.json", "model.safetensors.index.json"):
        shutil.copy(src + "/" + f, dst + "/" + f)
summary = {k: len(v) for k, v in stats.items()}
json.dump({"src": src, "summary": summary, "modules": stats}, open(dst + "/m0_dequant_report.json", "w"), indent=0)
summary["lm_head_dropped_as_tied"] = tied
print("dequantized:", summary, "tensors out:", len(out))
