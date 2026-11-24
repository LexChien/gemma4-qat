"""M12 summary: grades (m7_grade chat_<tag>_summary.json), factual set, 國父 answers, latency, and rebuilt-binary equivalence
(orig SBS on new binary vs M11 native_orig on shipped binary). -> raw/summary.json + markdown on stdout.
Factual grading: keyword match after removing LaTeX markup ($ \\text { } _), so '$\\text{H}_2\\text{O}$' counts as H2O."""
import json, re, os, statistics as st, sys
sys.path.insert(0, 'docs/Plan/m12/tools'); from m12_facts import FACTS
R = 'docs/Plan/m12/raw'; KW = {i: k for i, _, k in FACTS}
def norm(s): return re.sub(r'\\text|[${}_\s]', '', s)
SIZES = {'Cm8': 'out/m12/gemma4-e2b-m8-v114176-embbf16-q4_0.sbs', 'Bm8': 'out/m12/gemma4-e2b-m8-v154624-q4_0.sbs', 'Bm9': 'out/m12/gemma4-e2b-m9-v154624-q4_0.sbs', 'Am9': 'out/m12/gemma4-e2b-m9-q4emb.sbs',
         'orig_newbin': '$WEBSITE_ROOT/packages/Gemma4-Portable/models/gemma4-e2b-q4_0.sbs',
         'orig_oldbin': '$WEBSITE_ROOT/packages/Gemma4-Portable/models/gemma4-e2b-q4_0.sbs',
         'm11m9_oldbin': 'out/m11/gemma4-e2b-m9-q4_0.sbs', 'm11m8_oldbin': 'out/m11/gemma4-e2b-m8-q4_0.sbs'}
out = {}
for t, path in SIZES.items():
    o = {'bytes': os.path.getsize(path), 'GiB': round(os.path.getsize(path) / 2**30, 4)}
    fp = f'{R}/facts_{t}.jsonl'
    if os.path.exists(fp):
        fr = [json.loads(l) for l in open(fp)]
        ok = [r['id'] for r in fr if any(norm(k) in norm(r['answer']) for k in KW[r['id']])]
        o['facts'] = f'{len(ok)}/{len(fr)}'; o['facts_wrong'] = [r['id'] for r in fr if r['id'] not in ok]
        o['guofu'] = {r['id']: r['answer'][:40] for r in fr if r['id'] in ('f01', 'f21', 'f22')}
    lp = f'{R}/lat_{t}.jsonl'; gp = f'{R}/chat_{t}_summary.json'
    if os.path.exists(lp) and os.path.exists(gp):
        rows = [json.loads(l) for l in open(lp)]; g = json.load(open(gp)); c = [r for r in rows if r['status'] == 'complete']
        o.update(knowledge=g['knowledge_pass'], instruction=g['instruction_pass'], garbage=g['garbage_pass'], trad_ratio=g['trad_ratio'],
                 english_reply=g['n_english_reply'], status={s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})},
                 k08=[r['answer'][:30] for r in rows if r['id'] == 'k08'][0],
                 ttft_median_s=round(st.median([r['ttft_s'] for r in c]), 3), total_median_s=round(st.median([r['total_s'] for r in rows if 'total_s' in r]), 3),
                 decode_tps_median=round(st.median([r['decode_tps'] for r in c]), 2), peak_rss_gb_max=round(max(r.get('peak_rss_bytes') or 0 for r in rows) / 1e9, 3))
    out[t] = o
# rebuilt binary equivalence on the original SBS
a = {json.loads(l)['id']: json.loads(l)['answer'] for l in open('docs/Plan/m11/raw/lat_native_orig.jsonl')}
if os.path.exists(f'{R}/lat_orig_newbin.jsonl'):
    b = {json.loads(l)['id']: json.loads(l)['answer'] for l in open(f'{R}/lat_orig_newbin.jsonl')}
    out['_equiv_orig_newbin_vs_m11_shipped_bin'] = f'{sum(a[k] == b.get(k) for k in a)}/{len(a)} identical answers'
if os.path.exists(f'{R}/facts_orig_newbin.jsonl') and os.path.exists(f'{R}/facts_orig_oldbin.jsonl'):
    x = {json.loads(l)['id']: json.loads(l)['answer'] for l in open(f'{R}/facts_orig_newbin.jsonl')}
    y = {json.loads(l)['id']: json.loads(l)['answer'] for l in open(f'{R}/facts_orig_oldbin.jsonl')}
    out['_equiv_facts_orig_new_vs_old_bin'] = f'{sum(x[k] == y.get(k) for k in x)}/{len(x)} identical answers'
json.dump(out, open(f'{R}/summary.json', 'w'), ensure_ascii=False, indent=1); print(json.dumps(out, ensure_ascii=False, indent=1))
