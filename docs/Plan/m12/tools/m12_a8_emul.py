"""M12 root-cause test: emulate gemma.cpp's Q4_0 small-M kernel (ops/matmul-inl.h:445-463, 884-930) in MLX:
before every Q4_0 matmul the bf16 activation row is quantized to int8 with ONE scale per row per KC chunk
(s = max|x| / 127, kMaxKC = 6144, ops/matmul.h:65), i.e. W4A8 instead of MLX's W4A16.
mode: a16 (plain MLX) | a8 (all quantized linears, embedding/logits head bf16 = M11 SBS layout)
      | a8all (also the tied logits head + ple_model_proj, = M12 SBS layout where c_embedding is Q4_0)
usage: m12_a8_emul.py <mlx_dir> <mode> <prompt> [steps]"""
import sys, mlx.core as mx, mlx.nn as nn
from mlx_lm import load
d, mode, prompt = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 12
KC = 6144
def a8(x):
    K = x.shape[-1]; n = -(-K // KC); c = -(-K // n); c = -(-c // 32) * 32
    parts = []
    for i in range(0, K, c):
        p = x[..., i:i + c].astype(mx.float32)
        s = mx.max(mx.abs(p), axis=-1, keepdims=True) / 127.0
        s = mx.where(s == 0, 1.0, s)
        parts.append((mx.round(p / s) * s))
    return mx.concatenate(parts, axis=-1).astype(x.dtype)
if mode != 'a16':
    ql = nn.QuantizedLinear.__call__
    model0 = None
    def qcall(self, x):
        if mode == 'a8' and getattr(self, '_m12_bf16', False): return ql(self, x)
        return ql(self, a8(x))
    nn.QuantizedLinear.__call__ = qcall
    if mode == 'a8all':
        qe = nn.QuantizedEmbedding.as_linear
        nn.QuantizedEmbedding.as_linear = lambda self, x: qe(self, a8(x))
model, tok = load(d)
if mode == 'a8':  # M11 SBS stored ple_model_proj as bf16 (no int8 path)
    for n, m in model.named_modules():
        if n.endswith('per_layer_model_projection'): m._m12_bf16 = True
text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
seq = tok.encode(text, add_special_tokens=False); out = []
for s in range(N):
    logits = model(mx.array([seq]))[0, -1].astype(mx.float32)
    top = mx.argsort(-logits)[:3].tolist(); vals = [round(float(logits[t]), 3) for t in top]
    out.append(top[0]); seq.append(top[0])
    print(mode, s, [(tok.decode([t]), v) for t, v in zip(top, vals)], flush=True)
    if top[0] in (1, 106): break
print(mode, 'ANSWER', repr(tok.decode(out)))
