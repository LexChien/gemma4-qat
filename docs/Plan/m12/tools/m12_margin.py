"""M12: MLX greedy on a prompt (thinking off) in bf16 (as shipped) and float32 activations; print top-5 logits per step
for the first N answer tokens, to measure how close the decision is (e.g. 孫中山 vs 蔣介石).
usage: m12_margin.py <mlx_dir> <prompt> [steps]"""
import sys, json, mlx.core as mx
from mlx_lm import load
d, prompt = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
for dt in ('native', 'float32'):
    model, tok = load(d)
    if dt == 'float32': model.set_dtype(mx.float32)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(text, add_special_tokens=False)
    print(dt, 'prompt ids', ids)
    seq = list(ids); out = []
    for s in range(N):
        logits = model(mx.array([seq]))[0, -1].astype(mx.float32)
        top = mx.argsort(-logits)[:5].tolist(); vals = [round(float(logits[t]), 3) for t in top]
        out.append(top[0]); seq.append(top[0])
        print(dt, s, [(tok.decode([t]), t, v) for t, v in zip(top, vals)], 'margin', round(vals[0] - vals[1], 3), flush=True)
        if top[0] in (1, 106): break
    print(dt, 'ANSWER', tok.decode(out))
