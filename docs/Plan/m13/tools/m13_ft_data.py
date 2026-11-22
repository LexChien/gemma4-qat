"""M13 fine-tune data ("website-envelope distillation"): teach the model to answer the same way inside the website's xsoin-chat-v1
envelope as it does on the bare question. Inputs = website envelope (built by the website's own code, m13_envelopes.mjs) around
(a) the M9 cleaned zh-TW SFT prompts (thinking off) with their existing curated answers, (b) hand-written English prompts with the M8
model's own bare-prompt greedy answers. 25% of examples are kept bare (no envelope) to preserve the bare behaviour.
Decontamination vs the M13 eval (120 items) and the 50 prompts: drop if 4-gram Jaccard(prompt, any eval prompt) > 0.2, or if the prompt or
answer contains any accepted keyword of a factual eval item (keywords >= 2 CJK chars or >= 4 latin letters). Counts -> raw/ft_decontam.json
step 1 (gen): python m13_ft_data.py gen   -> out/m13/work/ft/en_answers.jsonl (MLX M8 greedy, bare)
step 2 (src): python m13_ft_data.py src   -> out/m13/work/ft/env_in.json  (+ decontam report)
step 3: node m13_envelopes.mjs out/m13/work/ft/env_in.json out/m13/work/ft/env_out.json
step 4 (ids): python m13_ft_data.py ids  -> out/m13/work/ft/{train,valid}.jsonl (token ids + loss mask counts)"""
import sys, json, re, random
sys.path.insert(0, 'docs/Plan/m13/tools'); sys.path.insert(0, 'docs/Plan/m7/tools')
import m13_sets as S; from m6_chat_prompts import P as P50; from m13_en_prompts import EN_PROMPTS
W = 'out/m13/work/ft/'; R = 'docs/Plan/m13/raw/'; step = sys.argv[1]; random.seed(13)
def grams(s): s = re.sub(r'\s+', '', s.lower()); return {s[i:i + 4] for i in range(len(s) - 3)}
EVP = [it[1] for it in S.ALL] + [p[2] for p in P50]; EG = [grams(p) for p in EVP]
KW = sorted({k for it in S.ALL if isinstance(it[2], list) for k in it[2]
             if len(re.findall(r'[\u4e00-\u9fff]', k)) >= 2 or len(re.findall(r'[A-Za-z]', k)) >= 4})
def contaminated(p, a):
    g = grams(p); j = max((len(g & e) / max(1, len(g | e)) for e in EG), default=0)
    if j > 0.2: return 'prompt_overlap'
    t = (p + ' ' + a).lower()
    for k in KW:
        if k.lower() in t: return 'answer_keyword:' + k
    return None
if step == 'gen':
    import mlx.core as mx; from mlx_lm import load; from mlx_lm.generate import generate_step
    model, tok = load('out/m8/mlx-m8-g32exact'); out = open(W + 'en_answers.jsonl', 'w')
    for p in EN_PROMPTS:
        ids = mx.array(tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False))
        toks = []
        for _, (t, _) in zip(range(300), generate_step(ids, model, sampler=lambda l: mx.argmax(l, axis=-1))):
            if int(t) in (1, 106): break
            toks.append(int(t))
        out.write(json.dumps(dict(prompt=p, answer=tok.decode(toks).strip(), n=len(toks)), ensure_ascii=False) + '\n'); out.flush()
if step == 'src':
    zh = [json.loads(l) for l in open('docs/Plan/m9/raw/sft_train.jsonl')] + [json.loads(l) for l in open('docs/Plan/m9/raw/sft_valid.jsonl')]
    zh = [dict(prompt=x['prompt'], answer=x['answer'], cat=x['cat'], loc='zh-tw') for x in zh if not x['think']]
    en = [json.loads(l) for l in open(W + 'en_answers.jsonl')]
    en = [dict(prompt=x['prompt'], answer=x['answer'], cat='en', loc='en') for x in en if x['n'] < 290]
    rep = dict(zh_in=len(zh), en_in=len(en), dropped={}, keywords_checked=len(KW)); keep = []
    for x in zh + en:
        c = contaminated(x['prompt'], x['answer'])
        if c: rep['dropped'].setdefault(c.split(':')[0], []).append([x['prompt'][:60], c]); continue
        keep.append(x)
    random.shuffle(keep)
    for i, x in enumerate(keep): x['id'] = f'ft{i:04d}'; x['bare'] = random.random() < 0.25
    rep['kept'] = len(keep); rep['kept_zh'] = sum(x['loc'] == 'zh-tw' for x in keep); rep['kept_en'] = sum(x['loc'] == 'en' for x in keep)
    rep['n_dropped'] = {k: len(v) for k, v in rep['dropped'].items()}
    json.dump(keep, open(W + 'src.json', 'w'), ensure_ascii=False)
    json.dump({'ft': [[x['id'], x['prompt'], x['loc']] for x in keep]}, open(W + 'env_in.json', 'w'), ensure_ascii=False)
    json.dump(rep, open(R + 'ft_decontam.json', 'w'), ensure_ascii=False, indent=1); print({k: v for k, v in rep.items() if k != 'dropped'})
if step == 'ids':
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('out/m5/mlx-qat-g32exact')
    src = {x['id']: x for x in json.load(open(W + 'src.json'))}; env = {e['id']: e for e in json.load(open(W + 'env_out.json'))['ft']}
    rows = []; skip = {}
    for i, x in src.items():
        e = env[i]
        if 'error' in e: skip['env_error'] = skip.get('env_error', 0) + 1; continue
        if e.get('task'): skip['website_task'] = skip.get('website_task', 0) + 1; continue  # website computes these itself
        user = x['prompt'] if x['bare'] else e['prompt']
        t = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pi = tok.encode(t, add_special_tokens=False); ai = tok.encode(x['answer'] + '<turn|>', add_special_tokens=False)
        if len(pi) + len(ai) > 1400: skip['too_long'] = skip.get('too_long', 0) + 1; continue
        rows.append(dict(id=i, cat=x['cat'], loc=x['loc'], bare=x['bare'], ids=pi + ai, n_prompt=len(pi), n_target=len(ai)))
    random.shuffle(rows); va = rows[:40]; tr = rows[40:]
    with open(W + 'train.jsonl', 'w') as f: f.writelines(json.dumps(r) + '\n' for r in tr)
    with open(W + 'valid.jsonl', 'w') as f: f.writelines(json.dumps(r) + '\n' for r in va)
    import statistics as st
    print(dict(train=len(tr), valid=len(va), skip=skip, en=sum(r['loc'] == 'en' for r in tr), bare=sum(r['bare'] for r in tr), median_len=st.median(len(r['ids']) for r in tr)))
