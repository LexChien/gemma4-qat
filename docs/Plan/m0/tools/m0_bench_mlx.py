"""M0 benchmark (MLX): load time, peak memory, generation tok/s (128 tokens, greedy), perplexity.
PPL: ctx=1024 (BOS + 1023 tokens), stride=512, only new tokens scored, fp32 log-softmax.
Usage: m0_bench_mlx.py <name> <model_dir> [--no-ppl] [--limit N] [--tag SUFFIX]"""
import sys, time, json, math, resource, argparse, os
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("path")
ap.add_argument("--no-ppl", action="store_true"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--tag", default="")
ap.add_argument("--ctx", type=int, default=1024); ap.add_argument("--stride", type=int, default=512)
a = ap.parse_args()
R = "docs/Plan/m0/raw/"
out = {"name": a.name, "path": a.path, "framework": "mlx", "mlx": mx.__version__}
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
t = time.perf_counter(); model, tok = load(a.path); out["load_lazy_s"] = time.perf_counter() - t
mx.eval(model.parameters()); out["load_s"] = time.perf_counter() - t
out["mem_after_load_GB"] = mx.get_active_memory() / 1e9
out["n_params_bytes_GB"] = sum(v.nbytes for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters())) / 1e9
PROMPT = "請用繁體中文詳細介紹台灣的地理、歷史與文化，至少寫五段。"
msgs = [{"role": "user", "content": PROMPT}]
try:
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
except Exception:  # base PT model has no chat template -> raw prompt
    prompt = tok.encode(PROMPT); out["prompt_mode"] = "raw (no chat template)"
sampler = make_sampler(temp=0.0)
for r in stream_generate(model, tok, prompt, max_tokens=8, sampler=sampler): pass  # warmup
mx.reset_peak_memory()
text = ""; last = None
for r in stream_generate(model, tok, prompt, max_tokens=128, sampler=sampler):
    text += r.text; last = r
out.update(gen_tokens=last.generation_tokens, gen_tps=last.generation_tps, prompt_tokens=last.prompt_tokens,
           prompt_tps=last.prompt_tps, gen_peak_mem_GB=last.peak_memory, gen_text=text)
data = json.load(open(R + "eval_tokens.json")); bos = data["bos_id"]
def ppl(ids):
    ids = ids[:a.limit] if a.limit else ids
    N = len(ids); nll = 0.0; cnt = 0; prev = 0; L = a.ctx - 1
    for b in range(0, N, a.stride):
        e = min(b + L, N)
        x = mx.array([[bos] + ids[b:e]])
        logits = model(x).astype(mx.float32)[0, :-1]          # predicts ids[b:e]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = mx.array(ids[b:e])
        tl = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
        s = max(prev, b) - b                                     # score only unseen targets
        v = -tl[s:].sum(); mx.eval(v)
        nll += v.item(); cnt += (e - b - s); prev = e
        if e == N: break
    return math.exp(nll / cnt), cnt
def finish():
    out["max_rss_GB"] = rss()
    json.dump(out, open(R + f"bench_{a.name}{a.tag}.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "gen_text"}, ensure_ascii=False)); print(text[:300])
if a.no_ppl: finish(); raise SystemExit
mx.reset_peak_memory(); t = time.perf_counter()
out["ppl_wikitext"], out["ppl_wikitext_tokens"] = ppl(data["wikitext"])
out["ppl_zh_tw"], out["ppl_zh_tw_tokens"] = ppl(data["zh_tw"])
out["ppl_time_s"] = time.perf_counter() - t; out["ppl_peak_mem_GB"] = mx.get_peak_memory() / 1e9
finish()
