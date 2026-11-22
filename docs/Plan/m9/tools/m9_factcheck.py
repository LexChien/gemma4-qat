"""M9 step 1a: fact-check the distilled KNOWLEDGE answers (raw M8 distill.jsonl, cat=know, thinking on/off).
1) Wikipedia (zh-tw variant) plain-text extract for each M8 topic -> raw/wiki_topics.json (cached; public API).
2) Judge: official QAT (thinking OFF for speed, no guide, reference <=1500 chars) compares answer vs reference -> 正確 / 錯誤 / 無法判斷.
3) Fix: for 錯誤 non-thinking items, regenerate with GUIDE + reference (teacher thinking on, answer only) and re-judge.
Output raw/factcheck.jsonl: {id, topic, verdict, reason, fixed_answer|null, fixed_verdict}. Resumable."""
import json, os, re, sys, time, urllib.parse, urllib.request
sys.path.insert(0, "docs/Plan/m8/tools")
R = "docs/Plan/m9/raw/"; R8 = "docs/Plan/m8/raw/"
src = open("docs/Plan/m8/tools/m8_make_prompts.py").read()
TOPICS = re.search(r'TOPICS = """(.*?)"""', src, re.S).group(1).split()
GUIDE = re.search(r'GUIDE = \((.*?)\)\n', open("docs/Plan/m8/tools/m8_distill.py").read(), re.S).group(1)
GUIDE = "".join(re.findall(r'"([^"]*)"', GUIDE))
WT = R + "wiki_topics.json"
wiki = json.load(open(WT)) if os.path.exists(WT) else {}
def fetch(title):
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "format": "json", "titles": title, "variant": "zh-tw", "redirects": 1})
    for k in range(6):
        try:
            req = urllib.request.Request("https://zh.wikipedia.org/w/api.php?" + q, headers={"User-Agent": "gemma4-qat-M9/0.1 (research; lex@xsoin.com)"})
            pg = list(json.load(urllib.request.urlopen(req, timeout=60))["query"]["pages"].values())[0]
            time.sleep(1.5); return dict(title=pg.get("title", title), text=pg.get("extract", ""))
        except Exception as e:
            err = str(e)[:100]; time.sleep(10 * (k + 1))
    return dict(title=title, text="", err=err)
for t in TOPICS:
    if t in wiki and not wiki[t].get("err") and len(wiki[t]["text"]) >= 300: continue
    best = None
    for cand in [t, t + "縣", t + "市", t + "島", t + "節"]:  # short pages are disambiguations (花蓮 -> 花蓮縣)
        r = fetch(cand)
        if r.get("err"): best = best or r; break
        if len(r["text"]) >= 300: best = r; break
        best = best if best and len(best["text"]) >= len(r["text"]) else r
    wiki[t] = best; json.dump(wiki, open(WT, "w"), ensure_ascii=False)
print("wiki topics", len(wiki), "with text", sum(bool(v["text"]) for v in wiki.values()), flush=True)
def topic_of(p):
    hits = [t for t in TOPICS if t in p]; return max(hits, key=len) if hits else None
def ref_of(t):
    s = wiki.get(t, {}).get("text", "") if t else ""; return s[:1500] if s else "（無參考資料）"
from mlx_lm import load
from mlx_lm.generate import batch_generate
model, tok = load("out/m5/mlx-qat-g32exact")
def gen(msgs_list, max_tokens=1200, think=True):
    ids = [tok.apply_chat_template(m, add_generation_prompt=True, enable_thinking=think) for m in msgs_list]
    r = batch_generate(model, tok, ids, max_tokens=max_tokens, return_token_ids=True)
    out = []
    for toks in (r.token_ids if hasattr(r, "token_ids") else r.tokens):
        s = tok.decode([int(x) for x in toks])
        if think: out.append(s.split("<channel|>")[-1].replace("<turn|>", "").strip() if "<channel|>" in s else "")
        else: out.append(s.replace("<turn|>", "").strip())
    return out
def judge_msgs(q, a, ref):
    return [{"role": "user", "content": "你是事實查核員。請根據下面的維基百科參考資料與確定的常識，檢查「回答」裡是否有明確的事實錯誤"
             "（例如地點、行政區、年代、人物、數字、定義、因果）。用字風格、詳略、主觀建議不算錯誤；參考資料沒提到且你也不確定的內容，不要判為錯誤。\n\n"
             f"參考資料：\n{ref}\n\n問題：{q}\n回答：{a}\n\n只輸出一行：「判定：正確」、「判定：錯誤｜<錯誤內容與正確事實>」或「判定：無法判斷」。"}]
def parse(s):
    m = re.search(r"判定[:：]\s*(正確|錯誤|無法判斷)\s*[｜|]?\s*(.*)", s)
    return (m.group(1), m.group(2).strip()[:300]) if m else ("無法判斷", "unparsed:" + s[:120])
FC = R + "factcheck.jsonl"
done = {json.loads(l)["id"] for l in open(FC)} if os.path.exists(FC) else set()
items = []
for l in open(R8 + "distill.jsonl"):
    r = json.loads(l); ids = r["gen_ids"]
    if r["cat"] != "know" or r["id"] in done or r["hit_max"] or not ids or ids[0] != 100 or 101 not in ids: continue
    ans = tok.decode(ids[ids.index(101) + 1:]).replace("<turn|>", "").strip()
    if ans: items.append(dict(id=r["id"], think=r["think"], prompt=r["prompt"], answer=ans, topic=topic_of(r["prompt"])))
print("to check", len(items), "done", len(done), flush=True)
B = 16; t0 = time.time(); cnt = {}
with open(FC, "a") as f:
    for i in range(0, len(items), B):
        ch = items[i:i + B]
        v = [parse(s) for s in gen([judge_msgs(e["prompt"], e["answer"], ref_of(e["topic"])) for e in ch], max_tokens=160, think=False)]
        fix_idx = [k for k, (vv, _) in enumerate(v) if vv == "錯誤" and not ch[k]["think"] and ch[k]["topic"]]
        fixed = {}
        if fix_idx:
            fa = gen([[{"role": "system", "content": GUIDE + "\n\n參考資料（維基百科）：\n" + ref_of(ch[k]["topic"]) + "\n請根據參考資料正確回答；參考資料沒有的細節不要編造，也不要提到「參考資料」。"},
                       {"role": "user", "content": ch[k]["prompt"]}] for k in fix_idx], max_tokens=1400)
            fv = [parse(s) for s in gen([judge_msgs(ch[k]["prompt"], a2, ref_of(ch[k]["topic"])) for k, a2 in zip(fix_idx, fa)], max_tokens=160, think=False)]
            for k, a2, (vv, rr) in zip(fix_idx, fa, fv): fixed[k] = (a2, vv)
        for k, (e, (vv, rr)) in enumerate(zip(ch, v)):
            cnt[vv] = cnt.get(vv, 0) + 1
            f.write(json.dumps(dict(id=e["id"], think=e["think"], topic=e["topic"], verdict=vv, reason=rr, answer=e["answer"],
                                    fixed_answer=fixed.get(k, (None, None))[0], fixed_verdict=fixed.get(k, (None, None))[1]), ensure_ascii=False) + "\n")
        f.flush(); el = time.time() - t0
        print(f"factcheck {min(i + B, len(items))}/{len(items)} {cnt} fixed_in_batch {len(fixed)} elapsed {el / 60:.1f} min", flush=True)
print("FACTCHECK_DONE", flush=True)
