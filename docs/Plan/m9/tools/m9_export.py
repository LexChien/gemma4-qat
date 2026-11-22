"""M8 export: merge the QAT-LoRA into the official exact grid and write an MLX g32 package.
Only the packed level tensors of the 205 target linears change; scales (= official d) and biases (= -8d) are reused,
every other tensor is copied from out/m5/mlx-qat-g32exact. Verifies losslessness: dequant(new pack) == the weight the
training forward used, bit-exact (bf16), for every target. Usage: m8_export.py CKPT_TAG OUT_DIR"""
import glob, json, os, shutil, sys
import mlx.core as mx
sys.path.insert(0, "docs/Plan/m8/tools"); sys.path.insert(0, "tools")
from m8_qat_lora import apply_qat_lora, load_pack, load_deployed_dense, levels_fixed, QATLoRALinear, EXACT
from mlx_exact_grid import pack
from mlx_lm import load
tag, out = sys.argv[1], sys.argv[2]
model, tok = load("out/m5/qat-text-hf"); P = load_pack(); load_deployed_dense(model, P); apply_qat_lora(model, rank=8, P=P)
model.load_weights(list(mx.load(tag + "_lora.safetensors").items()), strict=False)
new, rep = {}, dict(ckpt=tag, targets=0, blocks=0, blocks_changed_vs_official=0, levels_changed_vs_official=0, max_abs_dequant_vs_train=0.0, clipped_levels=0)
for name, mod in model.named_modules():
    if not isinstance(mod, QATLoRALinear): continue
    k = "language_model." + name; wm = mod.merged_fp32(); L = levels_fixed(wm, mod.d)
    pk = pack(L, L.shape); deq = mx.dequantize(pk, P[k + ".scales"], P[k + ".biases"], group_size=32, bits=4)
    tw = mod.quantized_weight()
    err = float(mx.abs(deq.astype(mx.float32) - tw.astype(mx.float32)).max())
    L0 = levels_fixed(mod.weight.astype(mx.float32), mod.d); ch = (L != L0)
    d32 = mod.d.astype(mx.float32)[..., None]
    raw = mx.round(wm.reshape(L.shape[0], -1, 32) / mx.where(d32 == 0, 1, d32)).reshape(L.shape)
    rep["clipped_levels"] += int(((raw < -8) | (raw > 7)).sum())
    rep["targets"] += 1; rep["blocks"] += L.size // 32; rep["levels_changed_vs_official"] += int(ch.sum())
    rep["blocks_changed_vs_official"] += int(ch.reshape(-1, 32).any(-1).sum()); rep["max_abs_dequant_vs_train"] = max(rep["max_abs_dequant_vs_train"], err)
    new[k + ".weight"] = pk; mx.eval(pk)
os.makedirs(out, exist_ok=True)
for f in os.listdir(EXACT):
    src = os.path.join(EXACT, f)
    if f.endswith(".safetensors"):
        W = dict(mx.load(src)); hit = [x for x in new if x in W]
        for x in hit:
            assert W[x].shape == new[x].shape and W[x].dtype == new[x].dtype, x; W[x] = new[x]
        mx.save_safetensors(os.path.join(out, f), W, metadata={"format": "mlx"})
    elif f != "exact_grid_report.json": shutil.copy(src, os.path.join(out, f))
rep["lossless"] = rep["max_abs_dequant_vs_train"] == 0.0
rep["size_bytes"] = sum(os.path.getsize(p) for p in glob.glob(out + "/*.safetensors"))
json.dump(rep, open(os.path.join(out, "m9_export_report.json"), "w"), indent=1); print(json.dumps(rep))
