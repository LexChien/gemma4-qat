"""M11: run the M7/M9 50-prompt thinking-off set through Gemma4-Portable gemma_native (website profile),
one single-shot process per prompt (no state/history; mmap load is ~0.2 s and excluded from timings).
Writes lat_<tag>.jsonl in the m7_latency format so docs/Plan/m7/tools/m7_grade.py can grade it.
usage: m11_native_eval.py <tag> <weights.sbs>"""
import sys, os, json, subprocess, time
sys.path.insert(0, 'docs/Plan/m7/tools'); from m6_chat_prompts import P
tag, w = sys.argv[1], sys.argv[2]
BIN = '$WEBSITE_ROOT/packages/Gemma4-Portable/mac/bin/gemma_native'
PROF = ['--threads', '2', '--context', '2048', '--prefill', '4', '--packed-q4', '1', '--tasks', '1', '--output', '256']
RAW = 'docs/Plan/m11/raw'; out = f'{RAW}/lat_{tag}.jsonl'; S = 'out/m11/STATUS.txt'
done = set()
if os.path.exists(out): done = {json.loads(l)['id'] for l in open(out)}
def run(prompt):
    t = time.time(); p = subprocess.run([BIN, '--weights', w, *PROF, '--prompt', prompt], capture_output=True, text=True)
    ev = [json.loads(l) for l in p.stdout.splitlines() if l.startswith('{')]
    return ev, time.time() - t, p.returncode, p.stderr
ev, wall, rc, err = run('17 * 23 等於多少？只回答數字。')  # warm-up + README check
d = [e for e in ev if e.get('event') == 'done'][0]
json.dump({'tag': tag, 'weights': w, 'status': d['status'], 'text': d['text'], 'rc': rc, 'ready': ev[0], 'done': d}, open(f'{RAW}/check17x23_{tag}.json', 'w'), ensure_ascii=False, indent=1)
print('17*23', d['status'], repr(d['text']), flush=True)
with open(out, 'a') as f:
    for n, (pid, cat, prompt, _) in enumerate(P):
        if pid in done: continue
        ev, wall, rc, err = run(prompt)
        dd = [e for e in ev if e.get('event') == 'done']
        if not dd:
            r = dict(run=1, id=pid, cat=cat, status='error', rc=rc, stderr=err[-500:], answer='', raw='', hit_max=False)
        else:
            d = dd[0]; ans = d.get('text', '')
            r = dict(run=1, id=pid, cat=cat, status=d['status'], prompt_tokens=d.get('input_tokens'), gen_tokens=d.get('output_tokens'),
                     ttft_s=round(d.get('first_text_ms', 0) / 1000, 3), prefill_s=round(d.get('prefill_ms', 0) / 1000, 3),
                     decode_tps=round(d.get('decode_tps', 0), 2), total_s=round(d.get('total_ms', 0) / 1000, 3), wall_s=round(wall, 3),
                     peak_rss_bytes=d.get('peak_rss_bytes'), hit_max=d['status'] == 'length' or (d.get('output_tokens') or 0) >= 256,
                     answer=ans, raw=''.join(e.get('text', '') for e in ev if e.get('event') == 'delta'))
        f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush()
        with open(S, 'a') as s: s.write(f"{time.strftime('%F %T')} stage=eval:{tag} pct={(n+1)*2} {pid} {r['status']} total={r.get('total_s')}s\n")
        print(pid, r['status'], r.get('ttft_s'), r.get('total_s'), ans[:60].replace('\n', ' '), flush=True)
print('EVAL_DONE', tag)
