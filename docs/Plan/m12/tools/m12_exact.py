"""M12: make every Q4_0 block (now incl. c_embedding and ple_model_proj) bit-exact to the MLX g32exact model,
optionally with a vocab row selection (keep.npy = old token ids in new-id order), then verify.
usage: m12_exact.py patch  <mlx_dir> <pipeline.sbs> <out.sbs> [keep.npy]
       m12_exact.py verify <mlx_dir> <out.sbs> <report.json> [keep.npy]"""
import sys, os, json, shutil, re, numpy as np, torch
sys.path.insert(0, 'docs/Plan/m11/tools'); sys.path.insert(0, 'src')
import sbs_io, m11_exact as m11
BF16 = ("pre_", "post_", "c_final_norm", "query_norm", "key_norm", "skip_scale", "ple_proj_norm") + (('c_embedding',) if os.environ.get('M12_EMB_BF16') else ())  # M12_EMB_BF16=1: candidate C keeps c_embedding bf16
def is_bf16(k): return k.startswith(BF16)
KEEP = None
def expected(m, name):
    if name == 'c_embedding': L, s = m.q('embed_tokens')
    elif name == 'ple_embeddings': L, s = m.q('embed_tokens_per_layer')
    elif name == 'ple_model_proj': L, s = m.q('per_layer_model_projection')
    else: return None
    if KEEP is not None and name != 'ple_model_proj': L, s = L[KEEP], s[KEEP]
    return [(L, s)]
def build(m, name, nb):
    p = expected(m, name)
    if p is None: return m11.build_q4(m, name, nb)
    b = m11.blocks(np.ascontiguousarray(p[0][0]), np.ascontiguousarray(p[0][1])).reshape(-1)
    assert b.size == nb, (name, b.size, nb); return b, 0
def patch(mlx, src, out, keep=None):
    global KEEP; KEEP = np.load(keep) if keep else None
    assert not os.path.exists(out), 'refuse overwrite'
    m = m11.M(mlx); d, _ = sbs_io.read_dir(src); shutil.copyfile(src, out + '.partial')
    q4 = [k for k in d if k not in ('config', 'tokenizer', 'toc') and not is_bf16(k)]
    changed = total = 0
    with open(out + '.partial', 'r+b') as f:
        for j, k in enumerate(q4):
            off, nb = d[k]; exp, nz = build(m, k, nb)
            cur = np.asarray(sbs_io.blob(src, (off, exp.size)))
            changed += int((cur.reshape(-1, 18) != exp.reshape(-1, 18)).any(1).sum()); total += exp.size // 18
            f.seek(off); f.write(exp.tobytes())
            if j % 25 == 0: print(f'patch {j}/{len(q4)} {k} changed={changed}/{total}', flush=True)
    os.rename(out + '.partial', out)
    print(json.dumps({'patched_tensors': len(q4), 'blocks_total': total, 'blocks_rewritten_vs_converter': changed})); print('PATCH_DONE')
def verify(mlx, path, rep, keep=None):
    global KEEP; KEEP = np.load(keep) if keep else None
    m = m11.M(mlx); d, size = sbs_io.read_dir(path)
    q4_bad = q4_blocks = zero_ok = bf_bad = bf_n = 0; types = {}
    for k, (off, nb) in d.items():
        if k in ('config', 'tokenizer', 'toc'): continue
        cur = np.asarray(sbs_io.blob(path, (off, nb)))
        if not is_bf16(k):
            exp, nz = build(m, k, nb); q4_blocks += nb // 18
            q4_bad += int((cur[:exp.size].reshape(-1, 18) != exp.reshape(-1, 18)).any(1).sum())
            if nz:
                Lz, sz = sbs_io.q4_decode(cur[exp.size:]); zero_ok += int(np.all(((Lz.astype(np.float32) - 8) * sz[:, None]) == 0))
            types['q4'] = types.get('q4', 0) + nb
        else:
            kind, i = re.match(r'(.*?)_?(\d+)?$', k).groups(); lay = f'layers.{i}.'
            src = {'c_final_norm': 'norm', 'ple_proj_norm': 'per_layer_projection_norm', 'c_embedding': 'embed_tokens'}.get(k) or lay + {
                'pre_att_ns': 'input_layernorm', 'post_att_ns': 'post_attention_layernorm', 'pre_ff_ns': 'pre_feedforward_layernorm',
                'post_ff_ns': 'post_feedforward_layernorm', 'query_norm': 'self_attn.q_norm', 'key_norm': 'self_attn.k_norm',
                'skip_scale': 'layer_scalar', 'post_ple_ns': 'post_per_layer_input_norm'}[kind]
            if src.endswith('k_norm') and not m.has(src): v = torch.ones(nb // 2)
            elif k == 'c_embedding': v = (m.dense(src)[torch.from_numpy(KEEP)] if KEEP is not None else m.dense(src)).reshape(-1)
            else: v = m.dense(src).reshape(-1)
            if 'norm' in k or '_ns' in k: v = v - 1.0
            exp = v.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint8)
            bf_n += nb // 2; bf_bad += int((cur.view(np.uint16) != exp.view(np.uint16)).sum())
            types['bf16'] = types.get('bf16', 0) + nb
    r = {'file': path, 'bytes': size, 'blobs': len(d), 'keep_rows': None if KEEP is None else int(KEEP.size),
         'blob_bytes': {k: v for k, v in types.items()}, 'tokenizer_bytes': d['tokenizer'][1],
         'q4': {'blocks': q4_blocks, 'blocks_not_bit_exact_vs_mlx': q4_bad, 'shared_kv_zero_tensors_ok': zero_ok},
         'bf16': {'elements': bf_n, 'elements_not_equal_bf16_of_mlx': bf_bad}}
    r['PASS'] = bool(q4_bad == 0 and bf_bad == 0 and zero_ok == 20)
    json.dump(r, open(rep, 'w'), indent=1); print(json.dumps(r, indent=1)); print('VERIFY_DONE')
if __name__ == '__main__':
    {'patch': patch, 'verify': verify}[sys.argv[1]](*sys.argv[2:])
