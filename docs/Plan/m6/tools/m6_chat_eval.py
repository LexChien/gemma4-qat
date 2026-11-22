"""M6 chat-template zh-TW Q&A eval (greedy, chat template, no thinking). Usage: m6_chat_eval.py MODEL_DIR TAG
Writes docs/Plan/m6/raw/chat_<TAG>.jsonl (generations) and chat_<TAG>_summary.json (metrics)."""
import json, os, re, sys, time
import mlx.core as mx, opencc
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
sys.path.insert(0, os.path.dirname(__file__)); from m6_chat_prompts import P
model_dir, tag = sys.argv[1], sys.argv[2]; MAXTOK = int(os.environ.get("MAXTOK", 256))
out = f"docs/Plan/m6/raw/chat_{tag}"; t2s, s2t = opencc.OpenCC("t2s"), opencc.OpenCC("s2t")
REFUSE = ["無法","不能","抱歉","不清楚","不確定","看不懂","不明白","不太明白","沒有意義","請提供","請問您","請再","無從","不知道","沒辦法","不會提供","不適合","預測","澄清","具體","未來","違法","隱私","危險","無意義","不存在","不太理解","不理解","沒有足夠"]
def script(s):
    tr = sum(1 for c in s if "\u4e00" <= c <= "\u9fff" and t2s.convert(c) != c)
    si = sum(1 for c in s if "\u4e00" <= c <= "\u9fff" and s2t.convert(c) != c)
    return tr, si
def rep4(s):
    s = re.sub(r"\s+", "", s); g = [s[i:i+4] for i in range(len(s) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)
def check(kind, arg, a):
    a2 = a.strip()
    if kind == "kw": return any(k in a2 for k in arg)
    if kind == "items": return len([l for l in a2.splitlines() if re.match(r"^\s*[-*•]\s+", l)]) == arg
    if kind == "lines": return len([l for l in a2.splitlines() if l.strip()]) == arg
    if kind == "maxlen": return len(re.sub(r"\s+", "", a2)) <= arg
    if kind == "digits": return re.sub(r"[\s。.*]", "", a2) == arg
    if kind == "eng": return len(a2) > 0 and sum(c.isascii() for c in a2) / len(a2) > 0.9
    if kind == "json":
        try: json.loads(re.sub(r"^```(json)?|```$", "", a2, flags=re.M).strip()); return True
        except Exception: return False
    if kind == "refuse": return any(k in a2 for k in REFUSE)
    return None
model, tok = load(model_dir); sampler = make_sampler(temp=0.0)
rows, t0 = [], time.time()
with open(out + ".jsonl", "w") as f:
    for pid, cat, prompt, (kind, arg) in P:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ts = time.time(); ans = generate(model, tok, prompt=text, max_tokens=MAXTOK, sampler=sampler, verbose=False); dt = time.time() - ts
        ntok = len(tok.encode(ans, add_special_tokens=False)); tr, si = script(ans)
        r = dict(id=pid, cat=cat, prompt=prompt, answer=ans, ok=check(kind, arg, ans), check=kind, chars=len(ans), tokens=ntok,
                 hit_max=ntok >= MAXTOK - 1, trad=tr, simp=si, rep4=round(rep4(ans), 4), sec=round(dt, 2))
        rows.append(r); f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
        print(pid, r["ok"], tr, si, r["rep4"], ntok, repr(ans[:60]), flush=True)
summ = {"model": model_dir, "tag": tag, "max_tokens": MAXTOK, "wall_s": round(time.time() - t0, 1)}
for cat in ["knowledge", "instruction", "garbage"]:
    rr = [r for r in rows if r["cat"] == cat]; summ[f"{cat}_pass"] = f"{sum(bool(r['ok']) for r in rr)}/{len(rr)}"
T, S = sum(r["trad"] for r in rows), sum(r["simp"] for r in rows)
summ.update(trad_chars=T, simp_chars=S, trad_ratio=round(T / max(1, T + S), 4),
            answers_with_simplified=sum(r["simp"] > r["trad"] for r in rows),
            mean_rep4=round(sum(r["rep4"] for r in rows) / len(rows), 4), n_rep4_gt_0_3=sum(r["rep4"] > 0.3 for r in rows),
            mean_chars=round(sum(r["chars"] for r in rows) / len(rows), 1), n_hit_max=sum(r["hit_max"] for r in rows),
            gen_tok_s=round(sum(r["tokens"] for r in rows) / sum(r["sec"] for r in rows), 1),
            peak_mem_gb=round(mx.get_peak_memory() / 1e9, 2))
json.dump(summ, open(out + "_summary.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(summ, ensure_ascii=False))
