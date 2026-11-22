"""M12 step: choose the kept vocabulary for a <2GB SBS and write a remapped tokenizer.json + keep.npy.
Rules (all verified by m12_vocab_check.py):
 1. ids 0..4368 always kept with unchanged ids (gemma.cpp hard-codes 2,105,106,107,2364,4368 in tokenizer.cc:75-79).
 2. all added/special tokens and all 256 <0xHH> byte-fallback tokens kept; <unusedN> dropped.
 3. a token containing any letter/mark outside {ASCII, CJK, Bopomofo, Fullwidth/Halfwidth, Ideographic, Hiragana, Katakana} is dropped
    (other scripts / accented Latin fall back to UTF-8 byte tokens, which stay).
 4. remaining candidates: tokens seen in the frequency corpus (wikitext-2 train+valid + XsOin site/docs text + gemma4-qat docs/src; eval prompts, factual set and wikitext test are held out) kept first,
    then unseen ones by ascending old id (BPE id order ~ training frequency) until the target size.
 5. BPE closure: for every kept token t and every merge (l, r) with l+r == t, l and r are kept (so text made only of kept
    tokens tokenizes identically); iterate 4-5 until |keep| == TARGET.
New id = rank of the old id among kept ids (order preserved).
usage: m12_vocab.py <target_vocab> <out_dir>"""
import sys, os, json, re, unicodedata, glob, numpy as np, collections
from tokenizers import Tokenizer
TOK = '/Users/lexchien/Documents/AI/ALife-Platform/Google-Gemma4/originals/models/gemma4-e2b-qat/tokenizer.json'
TARGET, OUT = int(sys.argv[1]), sys.argv[2]; os.makedirs(OUT, exist_ok=True)
j = json.load(open(TOK)); vocab = j['model']['vocab']; merges = j['model']['merges']
id2t = {i: t for t, i in vocab.items()}; V = len(vocab); assert V == 262144
added = {a['id'] for a in j['added_tokens']}
def bad_char(ch):
    if ord(ch) < 128 or ch == '\u2581': return False
    c = unicodedata.category(ch)
    if not (c.startswith('L') or c.startswith('M')): return False
    return not unicodedata.name(ch, '').startswith(('CJK', 'BOPOMOFO', 'FULLWIDTH', 'HALFWIDTH', 'IDEOGRAPHIC', 'HIRAGANA', 'KATAKANA'))
must = set(range(4369)) | added | {vocab[f'<0x{b:02X}>'] for b in range(256)}
unused = {i for t, i in vocab.items() if re.fullmatch(r'<unused\d+>', t)}
script_drop = {i for t, i in vocab.items() if i not in must and any(bad_char(c) for c in t)}
# frequency corpus
tk = Tokenizer.from_file(TOK); cnt = collections.Counter(); chars = 0
def feed(text):
    global chars
    for k in range(0, len(text), 200000):
        ids = tk.encode(text[k:k+200000], add_special_tokens=False).ids; cnt.update(ids)
    chars += len(text)
import pyarrow.parquet as pq
for f in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/*.parquet')):
    if '/test-' in f: continue  # held out for m12_vocab_check
    feed('\n'.join(pq.read_table(f).column('text').to_pylist()))
src = []
for root in ['/Users/lexchien/Documents/ChatGPT/XsOin/site', '/Users/lexchien/Documents/ChatGPT/XsOin/docs', 'docs', 'src']:
    for f in glob.glob(root + '/**/*', recursive=True):
        if f.endswith(('.html', '.md', '.txt', '.py', '.json', '.mjs', '.js', '.css')) and os.path.getsize(f) < 5_000_000:
            try: src.append(open(f, encoding='utf-8').read())
            except Exception: pass
feed(re.sub(r'<[^>]{0,200}>', ' ', '\n'.join(src)))
seen = set(cnt)
cands = [i for i in range(V) if i not in must and i not in unused and i not in script_drop]
order = sorted(cands, key=lambda i: (i not in seen, i))
# merges index: result token -> list of (l, r) ids
prod = collections.defaultdict(list)
for m in merges:
    l, r = m if isinstance(m, list) else m.split(' ')
    if l in vocab and r in vocab and (l + r) in vocab: prod[vocab[l + r]].append((vocab[l], vocab[r]))
def closure(keep):
    stack = list(keep)
    while stack:
        t = stack.pop()
        for l, r in prod.get(t, ()):
            for x in (l, r):
                if x not in keep: keep.add(x); stack.append(x)
    return keep
lo, hi = 0, len(order)
while lo < hi:  # largest prefix n of `order` whose closure fits TARGET
    mid = (lo + hi + 1) // 2
    if len(closure(set(must) | set(order[:mid]))) <= TARGET: lo = mid
    else: hi = mid - 1
keep = closure(set(must) | set(order[:lo]))
# fill any remaining slack with the next candidates whose closure still fits
for i in order[lo:]:
    if len(keep) >= TARGET: break
    k2 = closure(keep | {i})
    if len(k2) <= TARGET: keep = k2
keep = np.array(sorted(keep), dtype=np.int64)
assert all(keep[:4369] == np.arange(4369))
new = {int(o): n for n, o in enumerate(keep)}
nv = {id2t[o]: new[o] for o in keep}
nm = []
for m in merges:
    l, r = m if isinstance(m, list) else m.split(' ')
    if l in nv and r in nv and (l + r) in nv: nm.append(m)
j2 = json.loads(json.dumps(j)); j2['model']['vocab'] = nv; j2['model']['merges'] = nm
j2['added_tokens'] = [dict(a, id=new[a['id']]) for a in j['added_tokens']]
json.dump(j2, open(f'{OUT}/tokenizer.json', 'w'), ensure_ascii=False)
np.save(f'{OUT}/keep.npy', keep)
tot = sum(cnt.values()); cov = sum(c for i, c in cnt.items() if i in new)
rep = {'target': TARGET, 'kept': int(keep.size), 'dropped': V - int(keep.size), 'must': len(must), 'unused_dropped': len(unused),
       'script_dropped': len(script_drop), 'corpus_chars': chars, 'corpus_tokens': tot, 'corpus_distinct_ids': len(seen),
       'corpus_distinct_kept': len(seen & set(new)), 'corpus_token_coverage': cov / tot, 'seen_prefix_used': lo,
       'merges_kept': len(nm), 'merges_total': len(merges), 'max_old_id_kept': int(keep.max())}
json.dump(rep, open(f'{OUT}/vocab_report.json', 'w'), indent=1); print(json.dumps(rep, indent=1))
