"""Build the M8 SFT set from raw/sft_prompts.jsonl (+targets) and raw/distill.jsonl (teacher, thinking on, zh-TW guide).
Student format = official E2B template WITHOUT the guidance system prompt:
  thinking off: <bos><|turn>user\\n{p}<turn|>\\n<|turn>model\\n  ->  {answer}<turn|>
  thinking on : <bos><|turn>system\\n<|think|>\\n<turn|>\\n<|turn>user\\n{p}<turn|>\\n<|turn>model\\n -> <|channel>thought\\n{t}<channel|>{answer}<turn|>
Loss mask = target tokens only. Filters: closed thought, no hit_max, OpenCC s2twp (Taiwan usage), no simplified chars,
no English reasoning leak, English ratio, repetition, format checks, refusal keywords for gib/unans, eval-overlap."""
import json, re, random, os, sys, collections
import opencc
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "docs/Plan/m7/tools")
from m6_chat_prompts import P as EVALP
from transformers import AutoTokenizer
R = "docs/Plan/m8/raw/"; random.seed(8)
tok = AutoTokenizer.from_pretrained("out/m5/mlx-qat-g32exact")
s2twp, t2s, s2tw = opencc.OpenCC("s2twp"), opencc.OpenCC("t2s"), opencc.OpenCC("s2tw")
OKCH = set("里干郁秘群峰吃台粽")
LEAK = re.compile(r"Thinking Process|The user (is|wants|asks)|\*\*Analy[sz]e|Identify the|Drafting|Let me (think|analy|check|start|break|recall)|I need to|^\s*\d\.\s+\*\*[A-Z][a-z]+ the", re.M)
REFUSE = ["無法", "不能", "抱歉", "不清楚", "不確定", "看不懂", "不明白", "沒有意義", "請提供", "請問", "請再", "無從", "不知道", "沒辦法", "婉拒", "不會提供", "澄清", "補充", "看不出", "不太懂"]
def simp(s): return sum(1 for c in s if "\u4e00" <= c <= "\u9fff" and s2tw.convert(c) != c and c not in OKCH)
def eng_ratio(s):
    s = re.sub(r"```.*?```|\$\$.*?\$\$|\$[^$]*\$|\([^)]*\)|（[^）]*）", "", s, flags=re.S); s = re.sub(r"[\s\d\W_]", "", s)
    return 0.0 if not s else sum(c.isascii() for c in s) / len(s)
def rep4(s):
    s = re.sub(r"\s+", "", s); g = [s[i:i + 4] for i in range(len(s) - 3)]; return 0.0 if not g else 1 - len(set(g)) / len(g)
def grams(s): s = re.sub(r"\s+", "", s); return {s[i:i + 4] for i in range(len(s) - 3)}
EG = [grams(p[2]) for p in EVALP]
def overlap(p):
    g = grams(p); return max((len(g & e) / max(1, len(g | e)) for e in EG), default=0)
NUMW = {"一": 1, "兩": 2, "二": 2, "三": 3, "四": 4, "五": 5}
def fmt_ok(prompt, ans):
    m = re.search(r"剛好([一兩二三四五])個條列", prompt)
    if m: return len([l for l in ans.splitlines() if re.match(r"^\s*-\s+", l)]) == NUMW[m.group(1)]
    m = re.search(r"不超過 (\d+) 個字|(\d+) 個字以內|不超過 (\d+) 個字", prompt)
    if m: return len(re.sub(r"\s+", "", ans)) <= int(next(x for x in m.groups() if x)) * 1.2
    m = re.search(r"([三四])句的短詩", prompt)
    if m: return len([l for l in ans.splitlines() if l.strip()]) == NUMW[m.group(1)]
    if "只輸出英文" in prompt or "只輸出那一句" in prompt: return eng_ratio(ans) > 0.9 and len(ans) < 300
    return True
def prompt_ids(p, think):
    t = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=think)
    return tok.encode(t, add_special_tokens=False)
EX, rej = [], collections.Counter(); nconv = 0
prom = [json.loads(l) for l in open(R + "sft_prompts.jsonl")]
for o in prom:  # programmatic / self-written targets (thinking off)
    if o["target"] is not None:
        EX.append(dict(id=o["id"], cat=o["cat"], think=False, prompt=o["prompt"], answer=o["target"], thought_ids=[], src="programmatic/self-written"))
for l in open(R + "distill.jsonl"):
    r = json.loads(l); ids = r["gen_ids"]
    if r["hit_max"]: rej["hit_max"] += 1; continue
    if not ids or ids[0] != 100 or 101 not in ids: rej["no_closed_thought"] += 1; continue
    i = ids.index(101); ans = tok.decode(ids[i + 1:], skip_special_tokens=False).strip()
    if "<|channel>" in ans or "<channel|>" in ans or "<turn|>" in ans: rej["channel_in_answer"] += 1; continue
    a2 = s2twp.convert(ans); nconv += a2 != ans; ans = a2
    p = r["prompt"]; transl = "只輸出英文" in p or "只輸出那一句" in p
    if not ans: rej["empty"] += 1; continue
    if simp(ans) > 0: rej["simplified"] += 1; continue
    if LEAK.search(ans): rej["reasoning_leak"] += 1; continue
    if not transl and eng_ratio(ans) > 0.25: rej["english"] += 1; continue
    if rep4(ans) > 0.3: rej["repetition"] += 1; continue
    if len(ans) > 1600: rej["too_long"] += 1; continue
    if r["cat"] in ("unans", "gib") and not any(k in ans for k in REFUSE): rej["no_refusal"] += 1; continue
    if not fmt_ok(p, ans): rej["format"] += 1; continue
    EX.append(dict(id=r["id"], cat=r["cat"], think=r["think"], prompt=p, answer=ans, thought_ids=ids[:i + 1] if r["think"] else [], src="distilled"))
out, dropped_overlap, mx_ov = [], 0, 0.0
for e in EX:
    ov = overlap(e["prompt"]); mx_ov = max(mx_ov, ov)
    if ov > 0.5: dropped_overlap += 1; continue
    pi = prompt_ids(e["prompt"], e["think"]); ti = e["thought_ids"] + tok.encode(e["answer"], add_special_tokens=False) + [106]
    if len(pi) + len(ti) > 1024: rej["len>1024"] += 1; continue
    out.append(dict(e, ids=pi + ti, n_prompt=len(pi), n_target=len(ti)))
random.shuffle(out); nv = max(30, len(out) // 20)
for name, part in [("sft_valid.jsonl", out[:nv]), ("sft_train.jsonl", out[nv:])]:
    with open(R + name, "w") as f:
        for e in part: f.write(json.dumps({k: v for k, v in e.items() if k != "thought_ids"}, ensure_ascii=False) + "\n")
meta = dict(n_train=len(out) - nv, n_valid=nv, rejected=dict(rej), dropped_eval_overlap=dropped_overlap, max_eval_4gram_jaccard=round(mx_ov, 3),
            s2twp_changed_answers=nconv, by_cat=dict(collections.Counter((e["cat"], e["think"]) for e in out).most_common()) if False else
            {f"{k[0]}{'_think' if k[1] else ''}": v for k, v in collections.Counter((e["cat"], e["think"]) for e in out).items()},
            target_tokens=sum(e["n_target"] for e in out), total_tokens=sum(len(e["ids"]) for e in out),
            sources={"prompts": "self-written topics/templates/programmatic (project-owned); zh-TW Wikipedia leads via zetavg/zh-tw-wikipedia (CC BY-SA 3.0/4.0)",
                     "answers": "distilled from official Gemma 4 E2B QAT (thinking on, zh-TW guide system prompt; Gemma Terms of Use) + programmatic/self-written",
                     "postprocess": "OpenCC s2twp (Apache-2.0)"})
json.dump(meta, open(R + "sft_meta.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(meta, ensure_ascii=False))
