import json, os, sys, time, urllib.request, urllib.parse
from huggingface_hub import snapshot_download

def log(m):
    print(m, flush=True)

def dl_hf(repo):
    dest = 'weights/' + repo.split('/')[1]
    log(f'HF start {repo} -> {dest}')
    snapshot_download(repo, local_dir=dest, resume_download=True)
    log(f'HF done {repo}')

def dl_gcs(prefix):
    tok = None
    while True:
        u = 'https://storage.googleapis.com/storage/v1/b/gemma-data/o?prefix=' + urllib.parse.quote(prefix) + ('&pageToken=' + tok if tok else '')
        d = json.load(urllib.request.urlopen(u))
        for it in d.get('items', []):
            n = it['name']; size = int(it['size']); dst = os.path.join('weights/orbax', n)
            if os.path.exists(dst) and os.path.getsize(dst) == size:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            log(f'GCS {n} ({size/1e9:.2f}GB)')
            urllib.request.urlretrieve('https://storage.googleapis.com/gemma-data/' + urllib.parse.quote(n), dst)
            log(f'GCS done {n}')
        tok = d.get('nextPageToken')
        if not tok:
            break

# HF models first (resume incomplete)
for r in ['google/gemma-4-E2B-it-qat-q4_0-unquantized', 'google/gemma-4-E2B-it', 'google/gemma-4-E2B']:
    try:
        dl_hf(r)
    except Exception as e:
        log(f'HF ERROR {r}: {e}')
        raise
# JAX orbax + tokenizer
try:
    for p in ['checkpoints/gemma4-e2b-it/', 'checkpoints/gemma4-e2b-pt/', 'tokenizers/tokenizer_gemma4.model']:
        dl_gcs(p)
except Exception as e:
    log(f'GCS ERROR: {e}')
    raise
log('ALL DONE')
