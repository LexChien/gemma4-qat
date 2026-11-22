"""M4 (b): quantize official IT orbax weights with our Q4_0_MINMAX and the repo's Q4_0, compare with the
official gemma-4-E2B-it-qat-q4_0-unquantized (HF layout, blocks of 32 along the HF input axis = einsum
contraction axes). Also checks Q(qat) idempotence and that gemma.peft.block_quantize (JAX) == numpy ref.
Writes docs/Plan/m4/raw/q4_vs_official.json"""
import json, sys, time, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT + "/tools")
from convert_jax_to_hf import Orbax, text_map
from safetensors import safe_open
import jax, jax.numpy as jnp
from gemma.peft import _block_quantization as bq
from gemma.gm.nn.gemma4 import _qat

B = 32
cfg = json.load(open(ROOT + "/weights/gemma-4-E2B-it/config.json"))
O = Orbax(ROOT + "/weights/orbax/checkpoints/gemma4-e2b-it")
F = safe_open(ROOT + "/weights/gemma-4-E2B-it-qat-q4_0-unquantized/model.safetensors", "np")
QKEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "per_layer_input_gate",
         "per_layer_projection", "per_layer_model_projection", "embed_tokens.", "embed_tokens_per_layer")
ROW_STRIDE = int(os.environ.get("ROW_STRIDE", "32"))  # sample rows of the 262144-row tables

def bf16_to_f32(a):
    if a.dtype == np.uint16 or a.dtype.name == "bfloat16":
        return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)

def f32_to_bf16_f32(a):  # RNE to bf16, back to f32
    u = a.astype(np.float32).view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)

def minmax_levels(x):
    mn, mx = x.min(-1, keepdims=True), x.max(-1, keepdims=True)
    d = np.maximum(-mn / 8, mx / 7); sd = np.where(d == 0, 1, d)
    return np.clip(np.round(x / sd), -8, 7), d

def absmax8_levels(x):  # symmetric absmax/8 scale, clip to [-8,7] (AQT-style int4)
    d = np.abs(x).max(-1, keepdims=True) / 8; sd = np.where(d == 0, 1, d)
    return np.clip(np.round(x / sd), -8, 7), d

def q4_0_levels(x):  # repo Q4_0, normalized to a positive scale
    idx = np.abs(x).argmax(-1)[..., None]; m = np.take_along_axis(x, idx, -1)
    d = -m / 8; sd = np.where(d == 0, 1, d)
    q = np.clip(np.trunc(x / sd + 8.5), 0, 15) - 8
    d16 = d.astype(np.float16).astype(np.float32)
    return q * np.sign(np.where(d16 == 0, 1, d16)), np.abs(d16)

out = {"row_stride_tables": ROW_STRIDE, "tensors": {}}
agg = {}
t0 = time.time(); jax_checked = False
for k, fn in text_map(O, cfg, drop_shared_kv=True):
    if not any(s in k for s in QKEYS): continue
    off_full = F.get_slice(k)
    it = fn()
    if "embed_tokens" in k:
        rows = np.arange(0, it.shape[0], ROW_STRIDE)
        it = it[rows]; off = bf16_to_f32(F.get_tensor(k)[rows]) if False else None
        sl = off_full  # safetensors slice supports contiguous ranges only -> read full then sample
        off = F.get_tensor(k)[rows]
    else:
        off = F.get_tensor(k)
    off = bf16_to_f32(np.asarray(off)) if off.dtype != np.float32 else off
    it = np.asarray(it, np.float32)
    assert it.shape == off.shape, (k, it.shape, off.shape)
    itb, ofb = it.reshape(-1, B), off.reshape(-1, B)
    if not jax_checked:  # numpy ref == repo JAX implementation
        yj = np.asarray(bq.block_quantize(jnp.asarray(it), (it.ndim - 1,), block_size=B, formula="minmax"))
        Lr, dr = minmax_levels(itb); out["jax_vs_numpy_maxabs"] = float(np.abs(yj.reshape(-1, B) - Lr * dr).max())
        jax_checked = True
    Loff, doff = minmax_levels(ofb)
    dls = (ofb * Loff).sum(-1, keepdims=True) / np.maximum((Loff * Loff).sum(-1, keepdims=True), 1e-30)
    resid = np.abs(ofb / np.where(dls == 0, 1, dls) - Loff).max(-1)
    on_grid = float((resid < 0.05).mean())
    idem_exact = float((f32_to_bf16_f32(Loff * doff) == ofb).mean())
    Lmm, _ = minmax_levels(itb); Lq, _ = q4_0_levels(itb); La, _ = absmax8_levels(itb)
    rec = dict(shape=list(off.shape), off_on_grid_blocks=on_grid, off_idempotent_elems_bf16=idem_exact,
               minmax_elem_level_agree=float((Lmm == Loff).mean()), minmax_block_exact=float((Lmm == Loff).all(-1).mean()),
               absmax8_elem_level_agree=float((La == Loff).mean()), absmax8_block_exact=float((La == Loff).all(-1).mean()),
               q4_0_elem_level_agree=float((Lq == Loff).mean()), q4_0_block_exact=float((Lq == Loff).all(-1).mean()),
               it_vs_off_rel_l2=float(np.linalg.norm(it - off) / np.linalg.norm(off)))
    out["tensors"][k] = rec
    kind = next(s for s in QKEYS if s in k).strip(".")
    a = agg.setdefault(kind, {"n": 0, **{m: 0.0 for m in rec if m != "shape"}})
    a["n"] += 1
    for m in rec:
        if m != "shape": a[m] += rec[m]
    print(f"{time.time()-t0:7.1f}s {k} " + " ".join(f"{m}={v:.4f}" for m, v in rec.items() if m != "shape"), flush=True)
for kind, a in agg.items():
    n = a.pop("n"); out.setdefault("by_kind", {})[kind] = {m: v / n for m, v in a.items()} | {"n": n}
out["elapsed_s"] = time.time() - t0
json.dump(out, open(ROOT + "/docs/Plan/m4/raw/q4_vs_official.json", "w"), indent=1)
print(json.dumps(out["by_kind"], indent=1)); print("jax_vs_numpy_maxabs", out["jax_vs_numpy_maxabs"]); print("DONE")
