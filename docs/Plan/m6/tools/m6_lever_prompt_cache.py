"""M6 lever: prompt/KV cache reuse. A ~long fixed system prompt + 10 questions (thinking off, 64 new tokens).
Compare TTFT: (A) full prefill each time vs (B) prefix KV cache computed once and deep-copied per question.
Also (C) mx.compile is NOT applied by mlx_lm generate; we only record that here. Usage: m6_lever_prompt_cache.py MODEL TAG"""
import copy, json, statistics as st, sys, time, os
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
sys.path.insert(0, os.path.dirname(__file__)); from m6_chat_prompts import P
model, tok = load(sys.argv[1]); tag = sys.argv[2]
SYS = ("你是一位專業、友善的台灣在地助理，請一律使用繁體中文（臺灣用語）回答。回答要正確、簡潔、有條理；"
       "若問題資訊不足或無法回答，請直接說明並請使用者補充。不要編造事實，不要使用簡體字。") * 6
qs = [p[2] for p in P[:10]]
def ids_for(q):
    t = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return tok.encode(t, add_special_tokens=False)
full = [ids_for(q) for q in qs]
# common prefix length across all prompts
pl = min(len(x) for x in full)
for i in range(pl):
    if len({x[i] for x in full}) > 1: pl = i; break
def ttft(ids, cache=None):
    t0 = time.perf_counter()
    for y, _ in generate_step(mx.array(ids), model, max_tokens=64, sampler=make_sampler(temp=0.0), prompt_cache=cache):
        return time.perf_counter() - t0
ttft(full[0])  # warm kernels
A = [ttft(x) for x in full]
t0 = time.perf_counter(); base = make_prompt_cache(model)
for _ in generate_step(mx.array(full[0][:pl]), model, max_tokens=1, prompt_cache=base): break
# generate_step consumed one extra token into cache; rebuild cleanly by trimming
from mlx_lm.models.cache import trim_prompt_cache, can_trim_prompt_cache
extra = base[0].offset - pl if hasattr(base[0], "offset") else 0
if extra > 0 and can_trim_prompt_cache(base): trim_prompt_cache(base, extra)
prefix_s = time.perf_counter() - t0
B = [ttft(x[pl:], copy.deepcopy(base)) for x in full]
r = dict(model=sys.argv[1], tag=tag, prompt_tokens_median=st.median(len(x) for x in full), shared_prefix_tokens=pl,
         prefix_build_s=round(prefix_s, 3), can_trim=bool(can_trim_prompt_cache(base)),
         ttft_full_prefill_median_s=round(st.median(A), 4), ttft_full_prefill_max_s=round(max(A), 4),
         ttft_cached_prefix_median_s=round(st.median(B), 4), ttft_cached_prefix_max_s=round(max(B), 4))
json.dump(r, open(f"docs/Plan/m6/raw/lever_prompt_cache_{tag}.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(r, ensure_ascii=False))
