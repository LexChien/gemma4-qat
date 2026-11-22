"""M11 step 3/4: make every Q4_0 block in the converter's SBS bit-exact to the MLX g32exact model, then verify.

Why: gemma.cpp's Q4_0 encoder recomputes d = bf16(max_val / -8). That is lossless only for groups whose
largest-|x| element is level -8; ~40% of the MLX groups have no -8 level, so re-quantizing would move them
off the trained grid. Runtime decode is (nibble-8)*bf16_scale, identical to MLX (L*s + b, b = -8s), so we
write (s, L) directly: scale bits = MLX bf16 scale, nibble = MLX level. Layout follows convert_from_safetensors.py.

usage: m11_exact.py patch  <mlx_dir> <pipeline.sbs> <out.sbs>
       m11_exact.py verify <mlx_dir> <out.sbs> <original.sbs> <report.json>
"""
import sys, os, json, shutil, re, numpy as np, torch, safetensors
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, 'src')
import sbs_io
from g4qat.grid import unpack_mlx

BF16_PREFIX = ("pre_", "post_", "c_", "query_norm", "key_norm", "skip_scale", "ple_proj_norm", "ple_model_proj")
def is_bf16(name): return name.startswith(BF16_PREFIX)

class M:
    def __init__(s, d): s.f = safetensors.safe_open(d + '/model.safetensors', 'pt'); s.keys = set(s.f.keys())
    def has(s, b): return f'language_model.model.{b}.weight' in s.keys
    def q(s, b):  # -> L uint8 [r,c], sbits uint16 [r,c/32]
        p = f'language_model.model.{b}'
        L = unpack_mlx(s.f.get_tensor(p + '.weight')).numpy()
        sc = s.f.get_tensor(p + '.scales'); bb = s.f.get_tensor(p + '.biases')
        assert torch.equal(bb.float(), -8 * sc.float()), b
        return L, sc.view(torch.int16).numpy().view(np.uint16)
    def dense(s, b):
        p = f'language_model.model.{b}'
        if p + '.scales' in s.keys:
            L, _ = s.q(b); sc = s.f.get_tensor(p + '.scales').float(); r, c = L.shape
            return (torch.from_numpy(L).float().reshape(r, c // 32, 32) * sc[..., None] - 8 * sc[..., None]).reshape(r, c)
        return s.f.get_tensor(p + '.weight' if p + '.weight' in s.keys else p).float()

def blocks(L, sb):
    L = L.reshape(-1, 32); sb = np.ascontiguousarray(sb.reshape(-1))
    out = np.empty((L.shape[0], 18), np.uint8)
    out[:, :2] = sb.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = L[:, :16] | (L[:, 16:] << 4)
    return out

def expected_q4(m, name):
    """-> (bytes array [nblk,18] or None for converter-made zero rows, mask of zero blocks)"""
    kind, i = re.match(r'(.*?)_?(\d+)?$', name).groups()
    lay = f'layers.{i}.'
    if name == 'ple_embeddings': return [m.q('embed_tokens_per_layer')]
    if kind == 'gating_ein': return [m.q(lay + 'mlp.gate_proj'), m.q(lay + 'mlp.up_proj')]
    if kind == 'linear_w': return [m.q(lay + 'mlp.down_proj')]
    if kind == 'ple_gate': return [m.q(lay + 'per_layer_input_gate')]
    if kind == 'ple_proj': return [m.q(lay + 'per_layer_projection')]
    if kind == 'att_ein':
        L, sb = m.q(lay + 'self_attn.o_proj'); hd = m.f.get_slice(f'language_model.model.{lay}self_attn.q_norm.weight').get_shape()[0]
        H = L.shape[1] // hd
        return [(L.reshape(L.shape[0], H, hd).transpose(1, 0, 2), sb.reshape(sb.shape[0], H, hd // 32).transpose(1, 0, 2))]
    if kind == 'qkv_ein':
        hd = m.f.get_slice(f'language_model.model.{lay}self_attn.q_norm.weight').get_shape()[0]
        parts = [m.q(lay + 'self_attn.q_proj')]
        if m.has(lay + 'self_attn.k_proj'):
            (kL, ks), (vL, vs) = m.q(lay + 'self_attn.k_proj'), m.q(lay + 'self_attn.v_proj')
            for h in range(kL.shape[0] // hd):
                parts += [(kL[h*hd:(h+1)*hd], ks[h*hd:(h+1)*hd]), (vL[h*hd:(h+1)*hd], vs[h*hd:(h+1)*hd])]
        else:
            parts.append('ZEROS')
        return parts
    raise KeyError(name)

def build_q4(m, name, nbytes):
    parts = expected_q4(m, name); arrs = []
    for p in parts:
        if isinstance(p, str): continue
        arrs.append(blocks(np.ascontiguousarray(p[0]), np.ascontiguousarray(p[1])))
    b = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    nz = nbytes // 18 - b.shape[0]  # trailing converter-made zero K/V blocks (shared-KV layers)
    assert nz == 0 or parts[-1] == 'ZEROS', (name, nz)
    return b.reshape(-1), nz

def patch(mlx, src, out):
    assert not os.path.exists(out), 'refuse overwrite'
    m = M(mlx); d, _ = sbs_io.read_dir(src)
    shutil.copyfile(src, out + '.partial')
    q4 = [k for k in d if k not in ('config', 'tokenizer', 'toc') and not is_bf16(k)]
    changed = 0; total = 0
    with open(out + '.partial', 'r+b') as f:
        for j, k in enumerate(q4):
            off, nb = d[k]; exp, nz = build_q4(m, k, nb)
            cur = np.asarray(sbs_io.blob(src, (off, exp.size)))
            changed += int((cur.reshape(-1, 18) != exp.reshape(-1, 18)).any(1).sum()); total += exp.size // 18
            f.seek(off); f.write(exp.tobytes())
            if j % 25 == 0: print(f'patch {j}/{len(q4)} {k} changed_blocks_so_far={changed}/{total}', flush=True)
    os.rename(out + '.partial', out)
    print(json.dumps({'patched_tensors': len(q4), 'blocks_total': total, 'blocks_rewritten_vs_converter': changed}))
    print('PATCH_DONE')

def verify(mlx, path, orig, rep):
    m = M(mlx); d, size = sbs_io.read_dir(path); do, so = sbs_io.read_dir(orig)
    r = {'file': path, 'bytes': size, 'same_directory_as_original': d == do, 'q4': {}, 'bf16': {}, 'meta': {}}
    for k in ('config', 'tokenizer', 'toc'):
        r['meta'][k] = bool(np.array_equal(sbs_io.blob(path, d[k]), sbs_io.blob(orig, do[k])))
    q4_bad = q4_blocks = q4_diff_orig = zero_ok = 0; bf_bad = bf_n = bf_diff_orig = 0
    for k, (off, nb) in d.items():
        if k in ('config', 'tokenizer', 'toc'): continue
        cur = np.asarray(sbs_io.blob(path, (off, nb))); o = np.asarray(sbs_io.blob(orig, do[k]))
        if not is_bf16(k):
            exp, nz = build_q4(m, k, nb)
            q4_blocks += nb // 18
            q4_bad += int((cur[:exp.size].reshape(-1, 18) != exp.reshape(-1, 18)).any(1).sum())
            if nz:
                Lz, sz = sbs_io.q4_decode(cur[exp.size:]); zero_ok += int(np.all(((Lz.astype(np.float32) - 8) * sz[:, None]) == 0))
            q4_diff_orig += int((cur.reshape(-1, 18) != o.reshape(-1, 18)).any(1).sum())
        else:
            kind, i = re.match(r'(.*?)_?(\d+)?$', k).groups(); lay = f'layers.{i}.'
            src = {'c_embedding': 'embed_tokens', 'c_final_norm': 'norm', 'ple_model_proj': 'per_layer_model_projection',
                   'ple_proj_norm': 'per_layer_projection_norm'}.get(k) or lay + {
                   'pre_att_ns': 'input_layernorm', 'post_att_ns': 'post_attention_layernorm', 'pre_ff_ns': 'pre_feedforward_layernorm',
                   'post_ff_ns': 'post_feedforward_layernorm', 'query_norm': 'self_attn.q_norm', 'key_norm': 'self_attn.k_norm',
                   'skip_scale': 'layer_scalar', 'post_ple_ns': 'post_per_layer_input_norm'}[kind]
            if src.endswith('k_norm') and not m.has(src): v = torch.ones(nb // 2)
            else: v = m.dense(src).reshape(-1)
            if 'norm' in k or '_ns' in k: v = v - 1.0
            exp = v.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint8)
            n = nb // 2; bf_n += n
            bf_bad += int((cur.view(np.uint16) != exp.view(np.uint16)).sum())
            bf_diff_orig += int((cur.view(np.uint16) != o.view(np.uint16)).sum())
    r['q4'] = {'blocks': q4_blocks, 'blocks_not_bit_exact_vs_mlx': q4_bad, 'shared_kv_zero_tensors_ok': zero_ok,
               'blocks_differing_from_original_sbs': q4_diff_orig}
    r['bf16'] = {'elements': bf_n, 'elements_not_equal_bf16_of_mlx': bf_bad, 'elements_differing_from_original_sbs': bf_diff_orig}
    r['PASS'] = bool(r['same_directory_as_original'] and all(r['meta'].values()) and q4_bad == 0 and bf_bad == 0)
    json.dump(r, open(rep, 'w'), indent=1); print(json.dumps(r, indent=1)); print('VERIFY_DONE')

if __name__ == '__main__':
    {'patch': patch, 'verify': verify}[sys.argv[1]](*sys.argv[2:])
