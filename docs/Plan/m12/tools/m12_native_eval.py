"""M12: website-profile single-shot eval on gemma_native (any binary): the M7/M9 50-prompt set (graded later by m7_grade.py)
plus the M12 factual set. Same procedure as m11_native_eval.py (one process per prompt, default 30 s deadline, --output 256).
usage: m12_native_eval.py <tag> <weights.sbs> <binary> [facts_only]"""
import sys, os, json, subprocess, time
sys.path.insert(0, 'docs/Plan/m7/tools'); sys.path.insert(0, 'docs/Plan/m12/tools')
from m6_chat_prompts import P
from m12_facts import FACTS, WRONG
tag, w, BIN = sys.argv[1:4]; facts_only = len(sys.argv) > 4
PROF = ['--threads', '2', '--context', '2048', '--prefill', '4', '--packed-q4', '1', '--tasks', '1', '--output', '256']
RAW = 'docs/Plan/m12/raw'; S = 'out/m12/STATUS.txt'
def run(prompt):
    t = time.time(); p = subprocess.run([BIN, '--weights', w, *PROF, '--prompt', prompt], capture_output=True, text=True)
    ev = [json.loads(l) for l in p.stdout.splitlines() if l.startswith('{')]
    return ev, time.time() - t, p.returncode, p.stderr
def rec(pid, cat, prompt):
    ev, wall, rc, err = run(prompt); dd = [e for e in ev if e.get('event') == 'done']
    if not dd: return dict(run=1, id=pid, cat=cat, status='error', rc=rc, stderr=err[-500:], answer='', raw='', hit_max=False)
    d = dd[0]; ans = d.get('text', '')
    return dict(run=1, id=pid, cat=cat, status=d['status'], prompt_tokens=d.get('input_tokens'), gen_tokens=d.get('output_tokens'),
                ttft_s=round(d.get('first_text_ms', 0) / 1000, 3), prefill_s=round(d.get('prefill_ms', 0) / 1000, 3),
                decode_tps=round(d.get('decode_tps', 0), 2), total_s=round(d.get('total_ms', 0) / 1000, 3), wall_s=round(wall, 3),
                peak_rss_bytes=d.get('peak_rss_bytes'), hit_max=d['status'] == 'length' or (d.get('output_tokens') or 0) >= 256,
                answer=ans, raw=''.join(e.get('text', '') for e in ev if e.get('event') == 'delta'),
                ready={k: ev[0].get(k) for k in ('version', 'vocab_size', 'model_bytes', 'load_ms')} if ev else None)
def loop(items, out, kind):
    done = {json.loads(l)['id'] for l in open(out)} if os.path.exists(out) else set()
    with open(out, 'a') as f:
        for n, (pid, cat, prompt) in enumerate(items):
            if pid in done: continue
            r = rec(pid, cat, prompt)
            if kind == 'facts':
                acc = dict(FACTS_KW)[pid]; r['correct'] = any(k in r['answer'] for k in acc)
                r['wrong_kw'] = [k for k in WRONG.get(pid, []) if k in r['answer']]
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush()
            with open(S, 'a') as s: s.write(f"{time.strftime('%F %T')}   eval:{kind}:{tag} {n+1}/{len(items)} {pid} {r['status']} total={r.get('total_s')}s\n")
            print(pid, r['status'], r.get('total_s'), r.get('correct', ''), r['answer'][:60].replace('\n', ' '), flush=True)
FACTS_KW = [(i, kw) for i, _, kw in FACTS]
loop([(i, 'fact', q) for i, q, _ in FACTS], f'{RAW}/facts_{tag}.jsonl', 'facts')
if not facts_only: loop([(pid, cat, prompt) for pid, cat, prompt, _ in P], f'{RAW}/lat_{tag}.jsonl', 'p50')
print('EVAL_DONE', tag)
