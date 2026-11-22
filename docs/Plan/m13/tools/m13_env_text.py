"""M13 experiment (NOT a website change): render the SAME website envelope content (policy xsoin-public-v29 + runtime clock + task)
as plain labelled text instead of a JSON object, to measure how much the JSON format itself costs answer quality.
in: docs/Plan/m13/raw/envelopes.json -> out: docs/Plan/m13/raw/envelopes_txt.json (same structure)"""
import json
R = 'docs/Plan/m13/raw/'; d = json.load(open(R + 'envelopes.json'))
L = {'zh-tw': dict(svc='公司服務', unk='未提供事項', links='連結', now='現在時間', task='已驗證計算結果', q='訪客問題', sep='：'),
     'en': dict(svc='Company services', unk='Not supplied', links='Links', now='Current time', task='Verified calculation result', q='Visitor question', sep=': ')}
def render(p):
    e = json.loads(p); l = L[e['locale']]; s = l['sep']; pol = e['policy']; rt = e.get('runtime_context'); out = []
    ins = pol['instructions'].replace('task.result', l['task']).replace('task.reference', 'reference').replace('user_text', l['q'])
    out.append(ins)
    if 'services' in pol: out.append(l['svc'] + s + pol['services'] + ('。' if e['locale'] == 'zh-tw' else '. ') + pol.get('meeting', ''))
    if 'unknown' in pol: out.append(l['unk'] + s + pol['unknown'].replace('user_text', l['q']))
    if 'links' in pol: out.append(l['links'] + s + ', '.join(f'{k} {v}' for k, v in pol['links'].items()))
    if rt: out.append(l['now'] + s + f"{rt['current_date']} {rt['weekday']} {rt['current_time']} ({rt['time_zone']}). " + rt['instructions'].replace('runtime_context', l['now']))
    if e.get('task', {}).get('kind') == 'arithmetic': out.append(l['task'] + s + e['task']['result'])
    return '\n'.join(out) + '\n\n' + l['q'] + s + e['user_text']
o = {k: [dict(x, prompt=render(x['prompt'])) for x in v] for k, v in d.items()}
json.dump(o, open(R + 'envelopes_txt.json', 'w'), ensure_ascii=False, indent=1); print(o['sets'][0]['prompt']); print(o['sets'][60]['prompt'])
