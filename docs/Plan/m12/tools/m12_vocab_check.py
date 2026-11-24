"""M12: compare tokenization of the original vs trimmed tokenizer on (a) the 50 eval prompts + factual set, (b) wikitext-2 test lines,
(c) zh text from the XsOin Inc. site. Reports identical-sequence rate (after id remap) and token-count inflation.
usage: m12_vocab_check.py <vocab_dir>"""
import sys, json, glob, os, re, numpy as np
from tokenizers import Tokenizer
import pyarrow.parquet as pq
sys.path.insert(0, 'docs/Plan/m7/tools'); sys.path.insert(0, 'docs/Plan/m12/tools')
from m6_chat_prompts import P; from m12_facts import FACTS
D = sys.argv[1]; keep = np.load(D + '/keep.npy'); new = {int(o): n for n, o in enumerate(keep)}
A = Tokenizer.from_file('~/Documents/AI/ALife-Platform/Google-Gemma4/originals/models/gemma4-e2b-qat/tokenizer.json')
B = Tokenizer.from_file(D + '/tokenizer.json')
def cmp(name, texts):
    same = n = ta = tb = 0; diffs = []
    for t in texts:
        a = A.encode(t, add_special_tokens=False).ids; b = B.encode(t, add_special_tokens=False).ids
        ok = all(x in new for x in a) and [new[x] for x in a] == b
        same += ok; n += 1; ta += len(a); tb += len(b)
        if not ok and len(diffs) < 5: diffs.append(t[:80])
        assert B.decode(b) == A.decode(a), t[:80]  # round-trip text must be identical
    return {'set': name, 'texts': n, 'identical_token_seq': same, 'tokens_orig': ta, 'tokens_trim': tb, 'inflation': tb / max(ta, 1), 'examples_changed': diffs}
f = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet'))[0]
wt = [x for x in pq.read_table(f).column('text').to_pylist() if x.strip()]
site = []
for g in glob.glob('$WEBSITE_ROOT/site/**/*.html', recursive=True)[:300]:
    site += [l.strip() for l in re.sub(r'<[^>]{0,300}>', '\n', open(g, encoding='utf-8', errors='ignore').read()).split('\n') if re.search(r'[\u4e00-\u9fff]', l)]
r = [cmp('eval50+facts', [p[2] for p in P] + [q for _, q, _ in FACTS]), cmp('wikitext2-test', wt), cmp('site-zh-lines', site[:20000])]
json.dump(r, open(D + '/vocab_check.json', 'w'), ensure_ascii=False, indent=1); print(json.dumps(r, ensure_ascii=False, indent=1))
