"""M13 diagnosis: per-step top1-top2 logit margin on the native numerics, via the M12 diag build (same source/numerics as the stock
0.2.1 binary + top-5 print; analysis only). For each item: answer, min margin over generated tokens and the step where it occurs.
usage: m13_margins.py <tag> <weights.sbs> <id,id,...>"""
import sys, json, subprocess
sys.path.insert(0, 'docs/Plan/m13/tools'); import m13_sets as S
D = 'out/m12/bin/diag/gemma_native_diag'; tag, w, ids = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
PROF = ['--threads', '2', '--context', '2048', '--prefill', '4', '--packed-q4', '1', '--tasks', '1', '--output', '64']
items = {i[0]: i for i in S.ALL}; out = open(f'docs/Plan/m13/raw/margins_{tag}.jsonl', 'w')
for i in ids:
    p = subprocess.run([D, '--weights', w, *PROF, '--prompt', items[i][1]], capture_output=True, text=True)
    tops = [[(int(a), float(b)) for a, b in (x.split(':') for x in l.split()[1:])] for l in p.stderr.splitlines() if l.startswith('M12TOP5')]
    texts = [json.loads(l)['text'] for l in p.stdout.splitlines() if l.startswith('{') and '"delta"' in l]
    ms = [t[0][1] - t[1][1] for t in tops]; k = min(range(len(ms)), key=ms.__getitem__) if ms else -1
    ans = ''.join(texts); c = S.grade(items[i], ans)[0]
    r = dict(id=i, correct=c, min_margin=round(ms[k], 3) if ms else None, at_step=k, text_before=''.join(texts[:k]), chosen=texts[k] if 0 <= k < len(texts) else None, answer=ans)
    out.write(json.dumps(r, ensure_ascii=False) + '\n'); out.flush(); print(i, c, r['min_margin'], k, repr(r['text_before'][-12:]), repr(r['chosen']), flush=True)
