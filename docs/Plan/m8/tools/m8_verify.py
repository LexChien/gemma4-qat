"""Logit check: exported 4-bit package vs the dense training-time model (deployed dense + QAT-LoRA). Usage: m8_verify.py CKPT_TAG PKG"""
import json, os, sys
import mlx.core as mx
sys.path.insert(0, os.path.dirname(__file__))
from m8_qat_lora import apply_qat_lora, load_pack, load_deployed_dense
from mlx_lm import load
tag, pkg = sys.argv[1], sys.argv[2]
q, tok = load(pkg)
text = tok.apply_chat_template([{"role": "user", "content": "請簡單介紹台灣的夜市文化。"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
x = mx.array([tok.encode(text, add_special_tokens=False) + tok.encode("台灣的夜市是", add_special_tokens=False)])
lq = q(x).astype(mx.float32); mx.eval(lq); del q
d, _ = load("out/m5/qat-text-hf"); P = load_pack(); load_deployed_dense(d, P); apply_qat_lora(d, P=P)
if tag != "none": d.load_weights(list(mx.load(tag + "_lora.safetensors").items()), strict=False)
ld = d(x).astype(mx.float32)
pq, pd = mx.softmax(lq, -1), mx.softmax(ld, -1)
r = dict(tokens=x.shape[1], max_abs_logit_diff=float(mx.abs(lq - ld).max()), max_abs_prob_diff=float(mx.abs(pq - pd).max()),
         argmax_agree=float((lq.argmax(-1) == ld.argmax(-1)).mean()))
print(json.dumps(r)); json.dump(r, open(os.path.join("docs/Plan/m8/raw", "logit_check_" + os.path.basename(pkg) + ".json"), "w"))
