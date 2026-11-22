"""Grade chat answers saved by m6_latency.py (run 1, answer text after the thought block) with the M6 chat metrics.
Usage: m6_grade.py TAG...  -> raw/chat_<TAG>_summary.json"""
import json, re, sys, os
import opencc
sys.path.insert(0, os.path.dirname(__file__)); from m6_chat_prompts import P
t2s, s2t = opencc.OpenCC("t2s"), opencc.OpenCC("s2tw")  # s2tw: Taiwan standard (台/吃/粽/峰 are not simplified)
OKCH = set("里干郁秘群峰吃台粽")  # valid in Taiwan usage; OpenCC maps them for context reasons
SPEC = {p[0]: p[3] for p in P}
REFUSE = ["無法","不能","抱歉","不清楚","不確定","看不懂","不明白","不太明白","沒有意義","請提供","請問您","請再","無從","不知道","沒辦法","不會提供","不適合","預測","澄清","具體","未來","違法","隱私","危險","無意義","不存在","不太理解","不理解","沒有足夠"]
def script(s):
    return (sum(1 for c in s if "\u4e00" <= c <= "\u9fff" and t2s.convert(c) != c), sum(1 for c in s if "\u4e00" <= c <= "\u9fff" and s2t.convert(c) != c and c not in OKCH))
def rep4(s):
    s = re.sub(r"\s+", "", s); g = [s[i:i+4] for i in range(len(s) - 3)]; return 0.0 if not g else 1 - len(set(g)) / len(g)
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
RAW = os.environ.get("RAW", "docs/Plan/m7/raw")
LEAK = re.compile(r"Thinking Process|The user (is|wants|asks)|\*\*Analy[sz]e|Identify the|Recall|Drafting|Constraint|Let me (?:think|analy|check|start|break|recall)|I need to|^\s*\d\.\s+\*\*[A-Z]", re.M)
ENG_OK = {"i04", "i10", "i05"}  # translation / English greeting / JSON
def eng_ratio(s):
    s = re.sub(r"```.*?```|\$\$.*?\$\$|\$[^$]*\$", "", s, flags=re.S)  # drop code / LaTeX
    s = re.sub(r"[\s\d\W_]", "", s); return 0.0 if not s else sum(c.isascii() for c in s) / len(s)
for tag in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"{RAW}/lat_{tag}.jsonl")]; rows = [r for r in rows if r["run"] == 1]
    g = []
    for r in rows:
        k, a = SPEC[r["id"]]; ans = r["answer"]; tr, si = script(ans)
        raw = r.get("raw", ans); thought = r.get("n_ch_open", 1 if "<|channel>" in raw else 0) > 0 or "<channel|>" in raw
        leak = bool(LEAK.search(ans))  # reasoning leak
        eng_reply = r["id"] not in ENG_OK and eng_ratio(ans) > 0.5  # answered in English (language issue, not thinking)
        g.append(dict(id=r["id"], cat=r["cat"], ok=check(k, a, ans), thought_block=thought, english_leak=leak, english_reply=eng_reply, compliant=(not thought and not leak), eng_ratio=round(eng_ratio(ans), 3), trad=tr, simp=si, rep4=rep4(ans), chars=len(ans), hit_max=r["hit_max"]))
    s = {"tag": tag, "n": len(g)}
    for cat in ["knowledge", "instruction", "garbage"]:
        rr = [x for x in g if x["cat"] == cat]; s[f"{cat}_pass"] = f"{sum(bool(x['ok']) for x in rr)}/{len(rr)}"
    T, S = sum(x["trad"] for x in g), sum(x["simp"] for x in g)
    s.update(thinking_off_compliance=f"{sum(x['compliant'] for x in g)}/{len(g)}", n_thought_block=sum(x["thought_block"] for x in g), n_reasoning_leak=sum(x["english_leak"] for x in g), n_english_reply=sum(x["english_reply"] for x in g), trad_chars=T, simp_chars=S, trad_ratio=round(T / max(1, T + S), 4), answers_mostly_simplified=sum(x["simp"] > x["trad"] for x in g),
             answers_with_any_simplified=sum(x["simp"] > 0 for x in g), mean_rep4=round(sum(x["rep4"] for x in g) / len(g), 4),
             n_rep4_gt_0_3=sum(x["rep4"] > 0.3 for x in g), mean_chars=round(sum(x["chars"] for x in g) / len(g), 1),
             n_hit_max=sum(x["hit_max"] for x in g), per_prompt=g)
    json.dump(s, open(f"{RAW}/chat_{tag}_summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != "per_prompt"}, ensure_ascii=False))
