"""M6 end-to-end chat latency (MLX, greedy). Usage: m6_latency.py MODEL_DIR TAG [--think on|off] [--runs 2] [--kv-bits N]
[--max-tokens N] [--think-budget N] [--n N]
Per prompt: prompt tokens, prefill tok/s, TTFT, thinking duration/tokens (thinking on: <|channel>(100) ... <channel|>(101)),
time-to-first-answer-token (TTFA), answer duration/tokens, decode tok/s, total. Run 1 prompt 1 = cold (first Metal kernels after load).
--think-budget: if thinking exceeds N tokens, force-close it by injecting <channel|> (lever: limit thinking)."""
import argparse, json, os, sys, time, statistics as st
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
sys.path.insert(0, os.path.dirname(__file__)); from m6_chat_prompts import P
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag")
ap.add_argument("--think", default="off"); ap.add_argument("--runs", type=int, default=2); ap.add_argument("--kv-bits", type=int, default=None)
ap.add_argument("--max-tokens", type=int, default=None); ap.add_argument("--think-budget", type=int, default=None); ap.add_argument("--n", type=int, default=len(P)); ap.add_argument("--stride", type=int, default=1)
a = ap.parse_args(); think = a.think == "on"; MAXT = a.max_tokens or (1024 if think else 256)
CH_OPEN, CH_CLOSE, EOT = 100, 101, 106
t = time.time(); model, tok = load(a.model); load_s = time.time() - t
eos = set(tok.eos_token_ids) | {EOT, 1}
out = f"docs/Plan/m6/raw/lat_{a.tag}"; rows = []
def one(prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=think)
    if not think: text += "<|channel>thought\n<channel|>"  # E2B thinks even without <|think|>; prefill an empty thought block = thinking off
    ids = mx.array(tok.encode(text, add_special_tokens=False))
    t0 = time.perf_counter(); toks, times = [], []; forced = False
    gen = generate_step(ids, model, max_tokens=MAXT, sampler=make_sampler(temp=0.0), kv_bits=a.kv_bits)
    in_think = False; nthink = 0
    for y, _ in gen:
        y = int(y); now = time.perf_counter()
        toks.append(y); times.append(now)
        if y in eos: break
        if y == CH_OPEN: in_think = True
        elif y == CH_CLOSE: in_think = False
        elif in_think:
            nthink += 1
            if a.think_budget and nthink >= a.think_budget:  # restart decoding with forced close (simple, costs one re-prefill)
                forced = True; break
    if forced:
        ids2 = mx.concatenate([ids, mx.array(toks + [CH_CLOSE])])
        toks.append(CH_CLOSE); times.append(time.perf_counter())
        for y, _ in generate_step(ids2, model, max_tokens=MAXT - len(toks), sampler=make_sampler(temp=0.0), kv_bits=a.kv_bits):
            y = int(y); toks.append(y); times.append(time.perf_counter())
            if y in eos: break
    t_end = time.perf_counter(); n_prompt = ids.size
    ttft = times[0] - t0
    if CH_OPEN in toks and CH_CLOSE in toks:
        i_close = toks.index(CH_CLOSE); think_s = times[i_close] - times[0]; think_n = i_close + 1
        ans_idx = i_close + 1
    elif CH_OPEN in toks:  # never closed: all thinking
        i_close = None; think_s = t_end - times[0]; think_n = len(toks); ans_idx = len(toks)
    else:
        think_s, think_n, ans_idx = 0.0, 0, 0
    ttfa = (times[ans_idx] - t0) if ans_idx < len(toks) else None
    ans_toks = [x for x in toks[ans_idx:] if x not in eos]
    ans_s = (t_end - times[ans_idx]) if ans_idx < len(toks) else 0.0
    ntot = len(toks); dec_tps = (ntot - 1) / (times[-1] - times[0]) if ntot > 1 else None
    return dict(prompt_tokens=n_prompt, prefill_tps=round(n_prompt / ttft, 1), ttft_s=round(ttft, 4), think_s=round(think_s, 3),
                think_tokens=think_n, ttfa_s=None if ttfa is None else round(ttfa, 4), answer_s=round(ans_s, 3),
                answer_tokens=len(ans_toks), decode_tps=None if dec_tps is None else round(dec_tps, 1), total_s=round(t_end - t0, 3),
                gen_tokens=ntot, hit_max=ntot >= MAXT, forced_close=forced, answer=tok.decode(ans_toks))
def pct(v, p):
    v = sorted(x for x in v if x is not None)
    return None if not v else round(v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))], 4)
summ = dict(model=a.model, tag=a.tag, think=a.think, stride=a.stride, kv_bits=a.kv_bits, max_tokens=MAXT, think_budget=a.think_budget, load_s=round(load_s, 2), runs={})
with open(out + ".jsonl", "w") as f:
    for run in range(1, a.runs + 1):
        rr = []
        for k, (pid, cat, prompt, _) in enumerate(P[::a.stride][:a.n]):
            r = dict(run=run, id=pid, cat=cat, cold=(run == 1 and k == 0), **one(prompt)); rr.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(run, pid, r["ttft_s"], r["think_tokens"], r["ttfa_s"], r["total_s"], r["decode_tps"], repr(r["answer"][:40]), flush=True)
        warm = [r for r in rr if not r["cold"]]
        s = {}
        for key in ["prefill_tps", "ttft_s", "think_s", "think_tokens", "ttfa_s", "answer_s", "answer_tokens", "decode_tps", "total_s"]:
            s[key] = dict(median=pct([r[key] for r in warm], 50), p90=pct([r[key] for r in warm], 90))
        s["n_hit_max"] = sum(r["hit_max"] for r in rr); s["n_forced_close"] = sum(r["forced_close"] for r in rr)
        if run == 1: s["cold_first_prompt"] = {k: rr[0][k] for k in ["ttft_s", "ttfa_s", "total_s", "prompt_tokens"]}
        summ["runs"][run] = s; rows += rr
summ["peak_mem_gb"] = round(mx.get_peak_memory() / 1e9, 2)
json.dump(summ, open(out + "_summary.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(summ, ensure_ascii=False))
