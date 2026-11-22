"""M13: speed / memory / finish-status summary of native p50 + envp50 runs (lat_<tag>.jsonl, lat_env<tag>.jsonl) and 國父 answers.
usage: m13_speed.py tag [tag ...]"""
import json, sys, statistics as st, os
R = 'docs/Plan/m13/raw'
for t in sys.argv[1:]:
    for f in (f'lat_{t}', f'lat_env{t}', f'set_{t}', f'setenv_{t}'):
        p = f'{R}/{f}.jsonl'
        if not os.path.exists(p): continue
        L = [json.loads(l) for l in open(p)]; ok = [r for r in L if r.get('total_s') is not None]
        cnt = {}; [cnt.__setitem__(r['status'], cnt.get(r['status'], 0) + 1) for r in L]
        z1 = [r['answer'][:40].replace('\n', ' ') for r in L if r['id'] in ('z01', 'k08')]
        print(f"{f:16s} n={len(L)} status={cnt} med_total={st.median(r['total_s'] for r in ok):.2f}s med_ttft={st.median(r['ttft_s'] for r in ok):.2f}s "
              f"med_tps={st.median(r['decode_tps'] for r in ok if r['decode_tps']):.2f} peak_rss_max={max(r['peak_rss_bytes'] or 0 for r in ok)/1e9:.2f}GB {z1}")
