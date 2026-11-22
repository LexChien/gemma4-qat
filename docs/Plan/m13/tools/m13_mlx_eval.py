"""M13 diagnosis: run the M13 zh40+en40 set on an MLX model, greedy, thinking off, max 256 tokens.
mode a16 = plain MLX (W4A16); a8all = emulate gemma.cpp's Q4_0 small-M int8 activation quantization on every quantized linear
incl. the tied logits head and ple_model_proj (= stock-binary A-layout numerics, see m12_a8_emul.py / ops/matmul-inl.h).
Also records the first-answer-token logit margin (top1 - top2) for every item.
usage: m13_mlx_eval.py <mlx_dir> <tag> <mode>"""
import sys, json, time, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, 'docs/Plan/m13/tools'); import m13_sets as S13
from mlx_lm import load
from mlx_lm.generate import generate_step
d, tag, mode = sys.argv[1:4]; KC = 6144
def a8(x):
    K = x.shape[-1]; n = -(-K // KC); c = -(-K // n); c = -(-c // 32) * 32; parts = []
    for i in range(0, K, c):
        p = x[..., i:i + c].astype(mx.float32); s = mx.max(mx.abs(p), axis=-1, keepdims=True) / 127.0; s = mx.where(s == 0, 1.0, s)
        parts.append(mx.round(p / s) * s)
    return mx.concatenate(parts, axis=-1).astype(x.dtype)
if mode == 'a8all':
    ql = nn.QuantizedLinear.__call__; nn.QuantizedLinear.__call__ = lambda self, x: ql(self, a8(x))
    qe = nn.QuantizedEmbedding.as_linear; nn.QuantizedEmbedding.as_linear = lambda self, x: qe(self, a8(x))
model, tok = load(d); out = open(f'docs/Plan/m13/raw/mlx_{tag}_{mode}.jsonl', 'w'); nc = {'z': 0, 'e': 0}
for it in S13.ALL:
    text = tok.apply_chat_template([{"role": "user", "content": it[1]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = mx.array(tok.encode(text, add_special_tokens=False)); toks = []; margin = None; t0 = time.time()
    for n, (t, lp) in zip(range(256), generate_step(ids, model, sampler=lambda l: mx.argmax(l, axis=-1))):
        t = int(t)
        if margin is None:
            v = mx.sort(lp.astype(mx.float32).reshape(-1))[-2:].tolist(); margin = round(v[1] - v[0], 3)
        if t in (1, 106): break
        toks.append(t)
    ans = tok.decode(toks); c, l, w = S13.grade(it, ans); nc[it[0][0]] += c
    out.write(json.dumps(dict(id=it[0], answer=ans, correct=c, lang_ok=l, wrong_kw=w, margin0=margin, s=round(time.time() - t0, 2)), ensure_ascii=False) + '\n'); out.flush()
    print(it[0], c, margin, ans[:70].replace('\n', ' '), flush=True)
print('MLX_DONE', tag, mode, 'zh', nc['z'], '/40 en', nc['e'], '/40')
