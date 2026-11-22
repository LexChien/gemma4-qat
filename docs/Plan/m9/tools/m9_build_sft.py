"""M9: cleaned rebuild of the M8 SFT set (fact-check applied, guide mentions stripped from thinking traces,
OpenCC s2tw + small safe Taiwan-term map instead of s2twp phrase conversion) + general text chunks for KL. Based on: Build the M8 SFT set from raw/sft_prompts.jsonl (+targets) and raw/distill.jsonl (teacher, thinking on, zh-TW guide).
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
R = "docs/Plan/m9/raw/"; R8 = "docs/Plan/m8/raw/"; random.seed(9)
tok = AutoTokenizer.from_pretrained("out/m5/mlx-qat-g32exact")
s2twp, t2s, s2tw = opencc.OpenCC("s2twp"), opencc.OpenCC("t2s"), opencc.OpenCC("s2tw")
OKCH = set("里干郁秘群峰吃台粽游岩污夸采祕雇托后")
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
# s2twp's phrase table damages already-Traditional text (保存->儲存, 擴展->擴充套件, 支持->支援...). Use char-level s2tw + safe terms.
TWMAP = [("軟件", "軟體"), ("硬件", "硬體"), ("網絡", "網路"), ("信息", "資訊"), ("視頻", "影片"), ("程序員", "程式設計師"), ("編程", "程式設計"),
         ("數據庫", "資料庫"), ("服務器", "伺服器"), ("鼠標", "滑鼠"), ("打印", "列印"), ("短信", "簡訊"), ("U盤", "隨身碟"), ("內存", "記憶體"),
         ("默認", "預設"), ("屏幕", "螢幕"), ("激光", "雷射"), ("出租車", "計程車"), ("土豆", "馬鈴薯"), ("質量", "品質")]
KEEP = set("台里游岩郁污夸干采祕雇托秘群峰吃粽后")  # valid Traditional chars that char-level s2tw would rewrite (公里->公裡, 游泳->遊泳, 濃郁->濃鬱 ...)
def tw(x):
    y = s2tw.convert(x)
    y = "".join(a_ if a_ in KEEP else b_ for a_, b_ in zip(x, y)) if len(y) == len(x) else x
    for a_, b_ in TWMAP: y = y.replace(a_, b_)
    return y
GUIDE_LINE = re.compile(r"(?i)persona|system prompt|system instruction|the instructions? (state|say|require)|adher\w* to (the )?(instruction|persona|guideline)|as per the (persona|instruction|guideline)|guideline|based on (the )?instructions?|assistant for taiwan|taiwan(ese)? assistant|如果輸入是|請一律使用|臺灣的繁體中文助理|不要編造答案|嚴格遵守使用者")
PAREN = re.compile(r"\s*\((?:[^()]*?)(?:persona|instruction|guideline|Taiwan|Traditional Chinese)[^()]*\)", re.I)
STYLE = re.compile(r"(?i)traditional chinese|taiwan(ese)? (style|usage|tone|terminology|colloquial\w*|traditional)|colloquialism|繁體中文|臺灣慣用語|台灣慣用語")
SUBS = [re.compile(r"\s*\((?:[^()]*?)(?:Traditional|Taiwan)[^()]*\)", re.I), re.compile(r",?\s*(?:in|using|with) (?:concise |natural |formal |fluent |accurate )*(?:Taiwanese )?Traditional Chinese[^,.;:*]*", re.I)]
def strip_thought(t, prompt=""):
    out = []; ask_style = any(k in prompt for k in ("繁體", "臺灣用語", "台灣用語", "慣用語"))
    for ln in t.split("\n"):
        if STYLE.search(ln) and not ask_style:
            st = ln.strip()
            if st.startswith(("*", "-")) or len(st) < 100: continue
            for rx in SUBS: ln = rx.sub("", ln)
        if GUIDE_LINE.search(ln):
            ln2 = PAREN.sub("", ln)
            if GUIDE_LINE.search(ln2) or not ln2.strip(): continue
            ln = ln2
        out.append(ln)
    return "\n".join(out)
FC = {}
if os.path.exists(R + "factcheck.jsonl"):
    for l_ in open(R + "factcheck.jsonl"): o_ = json.loads(l_); FC[o_["id"]] = o_
fcs = collections.Counter(); nstrip = 0
prom = [json.loads(l) for l in open(R8 + "sft_prompts.jsonl")]
for o in prom:  # programmatic / self-written targets (thinking off)
    if o["target"] is not None:
        EX.append(dict(id=o["id"], cat=o["cat"], think=False, prompt=o["prompt"], answer=o["target"], thought_ids=[], src="programmatic/self-written"))
for l in open(R8 + "distill.jsonl"):
    r = json.loads(l); ids = r["gen_ids"]
    if r["hit_max"]: rej["hit_max"] += 1; continue
    if not ids or ids[0] != 100 or 101 not in ids: rej["no_closed_thought"] += 1; continue
    i = ids.index(101); ans = tok.decode(ids[i + 1:], skip_special_tokens=False).strip()
    if ans.endswith("<turn|>"): ans = ans[:-7].strip()
    fc = FC.get(r["id"])
    if r["cat"] == "know" and fc:
        if fc["verdict"] == "錯誤":
            if fc.get("fixed_answer") and fc.get("fixed_verdict") == "正確": ans = fc["fixed_answer"].strip(); fcs["fixed"] += 1
            else: fcs["dropped_wrong"] += 1; rej["factcheck_wrong"] += 1; continue
        else: fcs[fc["verdict"]] += 1
    tids = ids[:i + 1]
    if r["think"]:
        th = tok.decode(ids[1:i], skip_special_tokens=False); th2 = strip_thought(th, r["prompt"]); nstrip += th2 != th
        if GUIDE_LINE.search(th2): rej["guide_in_thought"] += 1; continue
        tids = [100] + tok.encode(th2, add_special_tokens=False) + [101]
    if "<|channel>" in ans or "<channel|>" in ans or "<turn|>" in ans: rej["channel_in_answer"] += 1; continue
    a2 = tw(ans); nconv += a2 != ans; ans = a2
    p = r["prompt"]; transl = "只輸出英文" in p or "只輸出那一句" in p
    if not ans: rej["empty"] += 1; continue
    if simp(ans) > 0: rej["simplified"] += 1; continue
    if LEAK.search(ans): rej["reasoning_leak"] += 1; continue
    if not transl and eng_ratio(ans) > 0.25: rej["english"] += 1; continue
    if rep4(ans) > 0.3: rej["repetition"] += 1; continue
    if len(ans) > 1600: rej["too_long"] += 1; continue
    if r["cat"] in ("unans", "gib") and not any(k in ans for k in REFUSE): rej["no_refusal"] += 1; continue
    if not fmt_ok(p, ans): rej["format"] += 1; continue
    EX.append(dict(id=r["id"], cat=r["cat"], think=r["think"], prompt=p, answer=ans, thought_ids=tids if r["think"] else [], src="distilled"))
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
            s2tw_changed_answers=nconv, factcheck=dict(fcs), thoughts_stripped=nstrip, by_cat=dict(collections.Counter((e["cat"], e["think"]) for e in out).most_common()) if False else
            {f"{k[0]}{'_think' if k[1] else ''}": v for k, v in collections.Counter((e["cat"], e["think"]) for e in out).items()},
            target_tokens=sum(e["n_target"] for e in out), total_tokens=sum(len(e["ids"]) for e in out),
            sources={"prompts": "self-written topics/templates/programmatic (project-owned); zh-TW Wikipedia leads via zetavg/zh-tw-wikipedia (CC BY-SA 3.0/4.0)",
                     "answers": "distilled from official Gemma 4 E2B QAT (thinking on, zh-TW guide system prompt; Gemma Terms of Use) + programmatic/self-written",
                     "postprocess": "OpenCC s2tw + safe Taiwan-term map (Apache-2.0)", "factcheck": "zh.wikipedia.org extracts (CC BY-SA 4.0) + official QAT judge", "general": "wikitext-2-raw-v1 train (CC BY-SA 3.0) + zh.wikipedia topic extracts excluding the M0 PPL titles"})
json.dump(meta, open(R + "sft_meta.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(meta, ensure_ascii=False))

# ---- general text chunks for the KL-to-teacher term (never the PPL eval texts: wikitext TEST and the 8 M0 zh titles) ----
from datasets import load_dataset
EVAL_TITLES = {"臺灣", "台灣", "臺北市", "玉山", "日月潭", "台灣積體電路製造", "臺灣積體電路製造", "國立故宮博物院", "臺灣高速鐵路", "阿里山"}
CH = 512; gen = []
wt = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
wi = tok.encode(wt[:2_000_000], add_special_tokens=False)
en = [[tok.bos_token_id] + wi[k:k + CH - 1] for k in range(0, len(wi) - CH, CH - 1)]
wk = json.load(open(R + "wiki_topics.json"))
zt = "\n\n".join(v["text"] for k_, v in wk.items() if v.get("text") and k_ not in EVAL_TITLES and v.get("title") not in EVAL_TITLES)
zi = tok.encode(zt, add_special_tokens=False)
zh = [[tok.bos_token_id] + zi[k:k + CH - 1] for k in range(0, len(zi) - CH, CH - 1)]
rng = random.Random(9); rng.shuffle(en); rng.shuffle(zh); n = min(len(en), len(zh), 400)
with open(R + "general.jsonl", "w") as f:
    for k in range(n):
        f.write(json.dumps(dict(src="wikitext2-train", ids=en[k])) + "\n"); f.write(json.dumps(dict(src="zhwiki-topics", ids=zh[k])) + "\n")
print("general chunks", 2 * n, "en avail", len(en), "zh avail", len(zh), "zh chars", len(zt), flush=True)
