"""M13: website-profile eval on the STOCK gemma_native (one process per prompt, 30 s default deadline, --output 256):
M13 zh40+en40 set (m13_sets) and optionally the M7/M9 50-prompt set (graded by m7_grade.py).
usage: m13_native_eval.py <tag> <weights.sbs> [sets|p50|all|envsets|envp50]
env*: the prompt is the exact website site-chat-v1 envelope from docs/Plan/m13/raw/envelopes.json (m13_envelopes.mjs)   (resumable; appends to docs/Plan/m13/raw/{set,lat}_<tag>.jsonl)"""
import sys, os, json, subprocess, time
sys.path.insert(0, 'docs/Plan/m7/tools'); sys.path.insert(0, 'docs/Plan/m13/tools')
from m6_chat_prompts import P
import m13_sets as S13
BIN = '$WEBSITE_ROOT/packages/Gemma4-Portable/mac/bin/gemma_native'
tag, w = sys.argv[1:3]; mode = sys.argv[3] if len(sys.argv) > 3 else 'sets'
PROF = ['--threads', '2', '--context', '2048', '--prefill', '4', '--packed-q4', '1', '--tasks', '1', '--output', '256']
RAW = 'docs/Plan/m13/raw'; ST = 'out/m13/STATUS.txt'
def rec(pid, cat, prompt):
    t = time.time(); p = subprocess.run([BIN, '--weights', w, *PROF, '--prompt', prompt], capture_output=True, text=True); wall = time.time() - t
    ev = [json.loads(l) for l in p.stdout.splitlines() if l.startswith('{')]; dd = [e for e in ev if e.get('event') == 'done']
    if not dd: return dict(run=1, id=pid, cat=cat, status='error', rc=p.returncode, stderr=p.stderr[-500:], answer='', hit_max=False)
    d = dd[0]
    return dict(run=1, id=pid, cat=cat, status=d["status"], prompt_tokens=d.get('input_tokens'), gen_tokens=d.get('output_tokens'),
                ttft_s=round(d.get('first_text_ms', 0) / 1000, 3), prefill_s=round(d.get('prefill_ms', 0) / 1000, 3),
                decode_tps=round(d.get('decode_tps', 0), 2), total_s=round(d.get('total_ms', 0) / 1000, 3), wall_s=round(wall, 3),
                peak_rss_bytes=d.get('peak_rss_bytes'), hit_max=d['status'] == 'length' or (d.get('output_tokens') or 0) >= 256,
                answer=d.get('text', ''), ready={k: ev[0].get(k) for k in ('version', 'model_bytes', 'load_ms')} if ev else None)
def loop(items, out, kind):
    done = {json.loads(l)['id'] for l in open(out)} if os.path.exists(out) else set()
    with open(out, 'a') as f:
        for n, (pid, cat, prompt, item) in enumerate(items):
            if pid in done: continue
            r = rec(pid, cat, prompt)
            if item: r['correct'], r['lang_ok'], r['wrong_kw'] = S13.grade(item, r['answer'])
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush()
            with open(ST, 'a') as s: s.write(f"{time.strftime('%F %T')}   eval:{kind}:{tag} {n+1}/{len(items)} {pid} {r['status']} {r.get('total_s')}s {r.get('correct', '')}\n")
            print(pid, r['status'], r.get('total_s'), r.get('correct', ''), r['answer'][:70].replace('\n', ' '), flush=True)
if mode in ('sets', 'all'): loop([(it[0], it[0][0], it[1], it) for it in S13.ALL], f'{RAW}/set_{tag}.jsonl', 'set')
if mode in ('p50', 'all'): loop([(pid, cat, prompt, None) for pid, cat, prompt, _ in P], f'{RAW}/lat_{tag}.jsonl', 'p50')
if mode.startswith('env'):
    E = json.load(open(f"{RAW}/{os.environ.get('M13_ENVFILE', 'envelopes.json')}")); PFX = os.environ.get('M13_ENVPFX', 'env'); IT = {i[0]: i for i in S13.ALL}; CAT = {p[0]: p[1] for p in P}
    if mode == 'envsets': loop([(e['id'], e['id'][0], e['prompt'], IT[e['id']]) for e in E['sets']], f'{RAW}/set{PFX}_{tag}.jsonl', 'envset')
    if mode == 'envp50': loop([(e['id'], CAT[e['id']], e['prompt'], None) for e in E['p50']], f'{RAW}/lat_{PFX}{tag}.jsonl', 'envp50')
print('EVAL_DONE', tag, mode)
