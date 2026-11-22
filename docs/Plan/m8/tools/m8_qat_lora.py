"""M8: MLX port of M5's mergeable on-grid LoRA-QAT.   y = x · Q(W + s·(A@B)ᵀ)ᵀ
Q = Q4_0 per 32-block along the input (contraction) axis on the OFFICIAL exact grid: the per-block scale d is the
official one (scales of out/m5/mlx-qat-g32exact, bf16), levels L = clip(round(W'/d), -8, 7); STE gradient.
Why fixed d: no closed-form scale formula reproduces the official grid (Q4_0_MINMAX changes 55% of official blocks,
absmax/8 74%; see docs/Plan/08 §2) - the official scale must be read from the checkpoint. With fixed d the base is a
fixed point (Q(W)=W) and the g32 pack (weight=L+8, scales=d, biases=-8d) is lossless by construction.
`ref_formula` = 'minmax' reproduces the M5 JAX recipe (bq.block_quantize minmax, bf16 scale) for the parity test."""
import math, glob
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
EXACT = "out/m5/mlx-qat-g32exact"

def q4_minmax_bf16(w):
    """M5 JAX formula (Q4_0_MINMAX, bf16 scale) - used only for the JAX parity test."""
    shp = w.shape; b = w.astype(mx.float32).reshape(-1, 32)
    d = mx.maximum(-b.min(-1, keepdims=True) / 8, b.max(-1, keepdims=True) / 7).astype(mx.bfloat16).astype(mx.float32)
    return (mx.clip(mx.round(b / mx.where(d == 0, 1, d)), -8, 7) * d).reshape(shp)

def levels_fixed(w, d):
    """w (out,in) fp32, d (out,in/32) -> integer levels in [-8,7] on the fixed grid."""
    o, i = w.shape; d32 = d.astype(mx.float32)[..., None]
    b = w.astype(mx.float32).reshape(o, i // 32, 32)
    return mx.clip(mx.round(b / mx.where(d32 == 0, 1, d32)), -8, 7).reshape(o, i)

def q4_fixed(w, d):
    o, i = w.shape; L = levels_fixed(w, d).reshape(o, i // 32, 32)
    return (L * d.astype(mx.float32)[..., None]).reshape(o, i)

class QATLoRALinear(nn.Module):
    def __init__(self, lin, d, rank=16, scale=2.0, seed=0):
        super().__init__()
        out_d, in_d = lin.weight.shape
        self.weight = lin.weight; self.d = d
        if "bias" in lin: self.bias = lin.bias
        k = mx.random.key(seed)
        self.lora_a = mx.random.uniform(-1 / math.sqrt(in_d), 1 / math.sqrt(in_d), (in_d, rank), key=k)
        self.lora_b = mx.zeros((rank, out_d)); self.scale = scale
    def merged_fp32(self):
        return self.weight.astype(mx.float32) + self.scale * (self.lora_a @ self.lora_b).T
    def quantized_weight(self):
        return q4_fixed(self.merged_fp32(), self.d).astype(self.weight.dtype)
    def __call__(self, x):
        wm = self.merged_fp32()
        w = (wm + mx.stop_gradient(q4_fixed(wm, self.d) - wm)).astype(self.weight.dtype).astype(x.dtype)  # STE
        y = x @ w.T
        return y + self.bias if "bias" in self else y

def load_pack(exact=EXACT):
    W = {}
    for f in glob.glob(f"{exact}/*.safetensors"): W.update(mx.load(f))
    return W

def load_deployed_dense(model, P):
    """Overwrite every weight that the g32-exact package quantizes with its dequantized value, so the dense bf16
    training model equals the deployed 4-bit model exactly (HF bf16 differs by bf16 scale rounding, <=0.0039)."""
    n = 0
    for name, mod in model.named_modules():
        k = "language_model." + name
        if k + ".scales" in P and hasattr(mod, "weight"):
            deq = mx.dequantize(P[k + ".weight"], P[k + ".scales"], P[k + ".biases"], group_size=32, bits=4).astype(mod.weight.dtype)
            assert deq.shape == mod.weight.shape, (name, deq.shape, mod.weight.shape)
            mod.weight = deq; n += 1
    mx.eval(model.parameters())
    return n

def apply_qat_lora(model, rank=16, scale=2.0, exact=EXACT, P=None):
    P = P if P is not None else load_pack(exact); S = P; model.freeze(); repl = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.endswith(TARGETS):
            d = S["language_model." + name + ".scales"]
            assert d.shape == (mod.weight.shape[0], mod.weight.shape[1] // 32), name
            q = QATLoRALinear(mod, d, rank, scale, seed=len(repl)); q.freeze(); q.unfreeze(keys=["lora_a", "lora_b"], strict=False)
            repl.append((name, q))
    model.update_modules(tree_unflatten(repl))
    return len(repl)

def lora_tree(model):
    return dict(tree_flatten(model.trainable_parameters()))
