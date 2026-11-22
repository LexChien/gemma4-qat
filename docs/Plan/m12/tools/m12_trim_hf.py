"""M12: slice embed_tokens / embed_tokens_per_layer rows of an M11 HF-layout safetensors to keep.npy order. Never overwrites.
usage: m12_trim_hf.py <in.safetensors> <keep.npy> <out.safetensors>"""
import sys, os, numpy as np, torch, safetensors, safetensors.torch
src, keep, out = sys.argv[1:4]; assert not os.path.exists(out)
k = torch.from_numpy(np.load(keep)); f = safetensors.safe_open(src, 'pt'); d = {}
for n in f.keys():
    t = f.get_tensor(n)
    if n.endswith(('embed_tokens.weight', 'embed_tokens_per_layer.weight')): t = t[k].contiguous(); print('trim', n, tuple(t.shape), flush=True)
    d[n] = t
safetensors.torch.save_file(d, out + '.partial', metadata={'format': 'pt', 'm12_source': src, 'm12_keep': keep})
os.rename(out + '.partial', out); print('TRIM_DONE', out)
