"""M11: side-by-side summary of gemma_native eval (grades + latency). -> raw/summary.json + stdout markdown"""
import json, statistics as st
R = 'docs/Plan/m11/raw'; out = {}
for t in ['native_orig', 'native_m9', 'native_m8']:
    rows = [json.loads(l) for l in open(f'{R}/lat_{t}.jsonl')]; g = json.load(open(f'{R}/chat_{t}_summary.json'))
    c = [r for r in rows if r['status'] == 'complete']
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    tt = [r['ttft_s'] for r in c]; tot = [r['total_s'] for r in rows if 'total_s' in r]
    out[t] = dict(knowledge=g['knowledge_pass'], instruction=g['instruction_pass'], garbage=g['garbage_pass'],
        english_reply=g['n_english_reply'], reasoning_leak=g['n_reasoning_leak'], thought_block=g['n_thought_block'], trad_ratio=g['trad_ratio'],
        status={s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})},
        ttft_median_s=round(st.median(tt), 3), ttft_p90_s=round(q(tt, .9), 3), total_median_s=round(st.median(tot), 3), total_mean_s=round(st.mean(tot), 3),
        total_p90_s=round(q(tot, .9), 3), decode_tps_median=round(st.median([r['decode_tps'] for r in c]), 2), gen_tokens_mean=round(st.mean([r['gen_tokens'] or 0 for r in rows if 'gen_tokens' in r]), 1),
        peak_rss_gb_max=round(max(r.get('peak_rss_bytes') or 0 for r in rows) / 1e9, 2))
json.dump(out, open(f'{R}/summary.json', 'w'), ensure_ascii=False, indent=1)
keys = list(next(iter(out.values())).keys())
print('| 指標 | ' + ' | '.join(out) + ' |'); print('|---|' + '---|' * len(out))
for k in keys: print(f'| {k} | ' + ' | '.join(str(out[t][k]) for t in out) + ' |')
