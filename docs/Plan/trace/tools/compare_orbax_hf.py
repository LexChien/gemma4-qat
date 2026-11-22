# 數值比對：orbax (JAX, f32/bf16) vs HF safetensors (bf16)。唯讀。
# 用法：.venv/bin/python docs/Plan/trace/tools/compare_orbax_hf.py  （在專案根目錄執行）
# 驗證：JAX<->HF 轉置/reshape（含方陣）、RMSNorm 慣例（w vs 1+w）、per_layer_model_projection 縮放、Q4_0 格點（含 embed）。
import json, struct, os, sys, numpy as np, tensorstore as ts
R = 'docs/Plan/trace/raw'
ORB = {t: os.path.abspath(f'weights/orbax/checkpoints/gemma4-e2b-{t}') + '/' for t in ('it', 'pt')}
HF = {'it': 'weights/gemma-4-E2B-it/model.safetensors', 'pt': 'weights/gemma-4-E2B/model.safetensors',
      'qat': 'weights/gemma-4-E2B-it-qat-q4_0-unquantized/model.safetensors'}
lines = []; res = {}
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)
_hf = {}
def hf_open(tag):
    if tag not in _hf:
        f = open(HF[tag], 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); h.pop('__metadata__', None)
        _hf[tag] = (h, np.memmap(HF[tag], dtype=np.uint8, mode='r', offset=8 + n))
    return _hf[tag]
def hf(tag, k, rows=None):
    h, mm = hf_open(tag)
    if k not in h: return None
    v = h[k]; assert v['dtype'] == 'BF16'; a, b = v['data_offsets']; shp = v['shape']
    if rows is not None:  # 只讀選定的列（第 0 軸）
        rowb = int(np.prod(shp[1:])) * 2
        out = np.stack([np.frombuffer(mm[a + r * rowb: a + (r + 1) * rowb], dtype=np.uint16) for r in rows])
        return (out.astype(np.uint32) << 16).view(np.float32).reshape([len(rows)] + shp[1:])
    u = np.frombuffer(mm[a:b], dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(shp)
def orb(tag, path, idx=None):
    t = ts.open({'driver': 'zarr', 'kvstore': {'driver': 'ocdbt', 'base': 'file://' + ORB[tag], 'path': path + '/'}}).result()
    x = t[idx].read().result() if idx is not None else t.read().result()
    return np.asarray(x, dtype=np.float32)
def bf16(x):  # f32 -> bf16（round-to-nearest-even）-> f32
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return u.view(np.float32)
def cmp(a, b):
    if a.shape != b.shape: return dict(shape_mismatch=[list(a.shape), list(b.shape)])
    d = a - b; na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return dict(exact=bool(np.array_equal(a, b)), max_abs=float(np.abs(d).max()), rel_l2=float(np.linalg.norm(d) / max(nb, 1e-30)),
                cos=float((a * b).sum() / max(na * nb, 1e-30)), norm_ratio=float(na / max(nb, 1e-30)))
def best(name, hfx, cands):
    out = {}
    for cn, cx in cands.items():
        out[cn] = cmp(hfx, bf16(cx)) if cx.shape == hfx.shape else dict(shape_mismatch=[list(cx.shape), list(hfx.shape)])
    win = [c for c, r in out.items() if r.get('exact')]
    P(f'  {name}: ' + '; '.join(f"{c}: " + (('EXACT' if r.get('exact') else f"maxabs={r['max_abs']:.3g} cos={r['cos']:.5f} ratio={r['norm_ratio']:.5f}") if 'shape_mismatch' not in r else f"shape {r['shape_mismatch']}") for c, r in out.items()))
    res[name] = dict(cands=out, exact=win); return out
L = 'model.language_model.'
# ---------- 1. 文字主幹：每層映射（含 global 4/14/19/34、KV 共享層 15+） ----------
def text_layer(tag, hftag, i):
    p = f'layer_{i}.'; q = f'{L}layers.{i}.'
    w = orb(tag, p + 'attn.q_einsum.w'); N, D, H = w.shape
    best(f'{hftag}/L{i}/q_proj', hf(hftag, q + 'self_attn.q_proj.weight'),
         {'q.transpose(0,2,1).reshape(N*H,D)': w.transpose(0, 2, 1).reshape(N * H, D), 'q.reshape(N*D,H)': w.reshape(N * D, H), 'q.reshape(N,D,H)->(D,N*H).T?': w.transpose(1, 0, 2).reshape(D, N * H)})
    kv = orb(tag, p + 'attn.kv_einsum.w')
    for c, nm in ((0, 'k_proj'), (1, 'v_proj')):
        x = hf(hftag, q + f'self_attn.{nm}.weight')
        if x is None: P(f'  {hftag}/L{i}/{nm}: HF 無此張量（KV 共享層被移除）'); res[f'{hftag}/L{i}/{nm}'] = 'absent'; continue
        best(f'{hftag}/L{i}/{nm}', x, {f'kv[{c}].reshape(K*D,H).T (=kv[{c},0].T)': kv[c].transpose(0, 2, 1).reshape(-1, D), f'kv[{c},0] (no T)': kv[c, 0]})
    o = orb(tag, p + 'attn.attn_vec_einsum.w')
    best(f'{hftag}/L{i}/o_proj', hf(hftag, q + 'self_attn.o_proj.weight'), {'o.reshape(N*H,D).T': o.reshape(-1, o.shape[-1]).T, 'o.transpose(1,0,2).reshape.T': o.transpose(1, 0, 2).reshape(-1, o.shape[-1]).T})
    g = orb(tag, p + 'mlp.gating_einsum.w')
    best(f'{hftag}/L{i}/gate_proj', hf(hftag, q + 'mlp.gate_proj.weight'), {'g[0] (no T)': g[0], 'g[1]': g[1]})
    best(f'{hftag}/L{i}/up_proj', hf(hftag, q + 'mlp.up_proj.weight'), {'g[1] (no T)': g[1], 'g[0]': g[0]})
    lw = orb(tag, p + 'mlp.linear.w')
    best(f'{hftag}/L{i}/down_proj', hf(hftag, q + 'mlp.down_proj.weight'), {'linear.T': lw.T})
    best(f'{hftag}/L{i}/per_layer_input_gate', hf(hftag, q + 'per_layer_input_gate.weight'), {'w.T': orb(tag, p + 'per_layer_input_gate.w').T})
    best(f'{hftag}/L{i}/per_layer_projection', hf(hftag, q + 'per_layer_projection.weight'), {'w.T': orb(tag, p + 'per_layer_projection.w').T})
    for jn, hn in (('pre_attention_norm', 'input_layernorm'), ('post_attention_norm', 'post_attention_layernorm'), ('pre_ffw_norm', 'pre_feedforward_layernorm'),
                   ('post_ffw_norm', 'post_feedforward_layernorm'), ('post_per_layer_input_norm', 'post_per_layer_input_norm'), ('attn.query_norm', 'self_attn.q_norm'), ('attn.key_norm', 'self_attn.k_norm')):
        x = hf(hftag, q + hn + '.weight')
        if x is None: P(f'  {hftag}/L{i}/{hn}: absent'); continue
        s = orb(tag, p + jn + '.scale'); best(f'{hftag}/L{i}/{hn}', x, {'w (HF 乘 w)': s, 'w-1 (HF 乘 1+w)': s - 1, 'w+1': s + 1})
    best(f'{hftag}/L{i}/layer_scalar', hf(hftag, q + 'layer_scalar'), {'skip_scale': orb(tag, p + 'skip_scale')})
P('# orbax vs HF 數值比對  (由 tools/compare_orbax_hf.py 產生)')
P('## 1. 文字主幹逐層 (it: orbax gemma4-e2b-it vs HF gemma-4-E2B-it; pt: orbax pt vs HF gemma-4-E2B)')
for i in (0, 4, 14, 15, 19, 34):
    text_layer('it', 'it', i)
for i in (0, 34):
    text_layer('pt', 'pt', i)
# ---------- 2. 嵌入/PLE/投影/final norm ----------
P('## 2. 嵌入、PLE、per_layer_model_projection、final norm')
rng = np.random.default_rng(0); rows = np.sort(rng.choice(262144, 64, replace=False)).tolist()
for tag in ('it', 'pt'):
    hft = tag
    e = np.stack([orb(tag, 'embedder.input_embedding', r) for r in rows])
    best(f'{hft}/embed_tokens(64 rows)', hf(hft, L + 'embed_tokens.weight', rows), {'input_embedding': e, 'input_embedding*sqrt(1536)': e * np.sqrt(1536)})
    pe = np.stack([orb(tag, 'embedder.per_layer_embeddings', r) for r in rows])
    best(f'{hft}/embed_tokens_per_layer(64 rows)', hf(hft, L + 'embed_tokens_per_layer.weight', rows), {'ple.reshape(V,35*256)': pe.reshape(len(rows), -1), 'ple*sqrt(256)': pe.reshape(len(rows), -1) * 16})
    pm = orb(tag, 'embedder.per_layer_model_projection.w')
    best(f'{hft}/per_layer_model_projection', hf(hft, L + 'per_layer_model_projection.weight'),
         {'w.reshape(1536,8960).T': pm.reshape(1536, -1).T, 'w.reshape.T * 1536^-0.5': pm.reshape(1536, -1).T * 1536 ** -0.5})
    best(f'{hft}/per_layer_projection_norm', hf(hft, L + 'per_layer_projection_norm.weight'), {'w': orb(tag, 'embedder.per_layer_projection_norm.scale'), 'w-1': orb(tag, 'embedder.per_layer_projection_norm.scale') - 1})
    best(f'{hft}/norm(final)', hf(hft, L + 'norm.weight'), {'w': orb(tag, 'final_norm.scale'), 'w-1': orb(tag, 'final_norm.scale') - 1})
    best(f'{hft}/embed_vision.embedding_projection', hf(hft, 'model.embed_vision.embedding_projection.weight'), {'mm_input_projection.w.T': orb(tag, 'embedder.mm_input_projection.w').T})
    a = orb(tag, 'embedder.audio_input_projection.w')
    best(f'{hft}/embed_audio.embedding_projection(方陣)', hf(hft, 'model.embed_audio.embedding_projection.weight'), {'w.T': a.T, 'w (no T)': a})
# ---------- 3. 視覺/音訊方陣 ----------
P('## 3. 視覺 / 音訊塔（方陣轉置驗證，it）')
for l in (0, 15):
    vp = 'vision_encoder.transformer.stacked_layers.block.'; vh = f'model.vision_tower.encoder.layers.{l}.'
    q = orb('it', vp + 'attn.q_einsum.w', l)  # [12,768,64]
    best(f'it/vision L{l} q_proj(768x768)', hf('it', vh + 'self_attn.q_proj.linear.weight'), {'q.transpose(0,2,1).reshape(768,768)': q.transpose(0, 2, 1).reshape(768, 768), 'q.reshape(768,768) (錯誤候選)': q.reshape(768, 768).T})
    kv = orb('it', vp + 'attn.kv_einsum.w', l)
    best(f'it/vision L{l} k_proj', hf('it', vh + 'self_attn.k_proj.linear.weight'), {'kv[0].transpose(0,2,1).reshape': kv[0].transpose(0, 2, 1).reshape(768, 768), 'kv[0].reshape.T': kv[0].reshape(768, 768).T})
    o = orb('it', vp + 'attn.attn_vec_einsum.w', l)
    best(f'it/vision L{l} o_proj', hf('it', vh + 'self_attn.o_proj.linear.weight'), {'o.reshape(768,768).T': o.reshape(768, 768).T, 'o.reshape(768,768)': o.reshape(768, 768)})
    best(f'it/vision L{l} input_layernorm', hf('it', vh + 'input_layernorm.weight'), {'w': orb('it', vp + 'pre_attention_norm.scale', l), 'w-1': orb('it', vp + 'pre_attention_norm.scale', l) - 1})
    best(f'it/vision L{l} q_proj.input_max', hf('it', vh + 'self_attn.q_proj.input_max'), {'clip_input_max': orb('it', vp + 'attn.q_einsum.clip_input_max', l)})
for l in (0, 11):
    ap = f'audio_encoder.conformer/stacked_layers_{l}.'; ah = f'model.audio_tower.layers.{l}.'
    for jn, hn in (('trans_atten.self_atten.query.kernel', 'self_attn.q_proj.linear.weight'), ('trans_atten.self_atten.key.kernel', 'self_attn.k_proj.linear.weight'),
                   ('lconv.linear_end.kernel', 'lconv1d.linear_end.linear.weight')):
        k = orb('it', ap + jn); best(f'it/audio L{l} {hn}(方陣)', hf('it', ah + hn), {'kernel.T': k.T, 'kernel (no T)': k})
    k = orb('it', ap + 'trans_atten.post.kernel'); best(f'it/audio L{l} self_attn.post', hf('it', ah + 'self_attn.post.linear.weight'), {'post.reshape(1024,1024).T': k.reshape(1024, 1024).T, 'post.reshape': k.reshape(1024, 1024)})
    k = orb('it', ap + 'trans_atten.self_atten.relative_position_embedding.pos_proj.kernel'); best(f'it/audio L{l} relative_k_proj', hf('it', ah + 'self_attn.relative_k_proj.weight'), {'pos_proj.reshape(1024,1024).T': k.reshape(1024, 1024).T, 'reshape': k.reshape(1024, 1024)})
    best(f'it/audio L{l} norm_pre_attn', hf('it', ah + 'norm_pre_attn.weight'), {'w': orb('it', ap + 'trans_atten.pre_norm.scale'), 'w-1': orb('it', ap + 'trans_atten.pre_norm.scale') - 1})
    best(f'it/audio L{l} per_dim_scale', hf('it', ah + 'self_attn.per_dim_scale'), {'w': orb('it', ap + 'trans_atten.self_atten.per_dim_scale')})
# ---------- 4. QAT ckpt vs orbax it：縮放與差異 ----------
P('## 4. QAT ckpt (HF qat-q4_0-unquantized) vs orbax it — 用 norm_ratio 判斷是否預先縮放')
pm = orb('it', 'embedder.per_layer_model_projection.w').reshape(1536, -1).T
best('qat/per_layer_model_projection', hf('qat', L + 'per_layer_model_projection.weight'), {'w.T': pm, 'w.T*1536^-0.5': pm * 1536 ** -0.5})
best('it(HF)/per_layer_model_projection vs qat', hf('qat', L + 'per_layer_model_projection.weight'), {'HF it': hf('it', L + 'per_layer_model_projection.weight')})
w = orb('it', 'layer_0.attn.q_einsum.w'); best('qat/L0/q_proj', hf('qat', L + 'layers.0.self_attn.q_proj.weight'), {'q.transpose(0,2,1).reshape': w.transpose(0, 2, 1).reshape(2048, 1536)})
best('qat/L0/input_layernorm', hf('qat', L + 'layers.0.input_layernorm.weight'), {'w': orb('it', 'layer_0.pre_attention_norm.scale')})
best('qat/norm(final)', hf('qat', L + 'norm.weight'), {'w': orb('it', 'final_norm.scale')})
e = np.stack([orb('it', 'embedder.input_embedding', r) for r in rows])
best('qat/embed_tokens(64 rows)', hf('qat', L + 'embed_tokens.weight', rows), {'input_embedding': e})
# ---------- 5. Q4_0 格點：embed / PLE / 投影 ----------
P('## 5. Q4_0 格點檢查（每 32 個一組；HF 最後一軸 = 輸入軸 / 嵌入維）')
def grid(x, axis=-1, nblk=4096):
    x = np.moveaxis(x, axis, -1)
    if x.shape[-1] % 32: return None
    x = x.reshape(-1, 32); idx = np.random.default_rng(1).choice(len(x), min(nblk, len(x)), replace=False); x = x[idx]
    u = np.array([len(np.unique(r)) for r in x]); amax = np.abs(x).max(1, keepdims=True); d = np.where(amax == 0, 1, amax / 8)
    err = np.abs(x / d - np.round(x / d)).max(1)
    return dict(blocks=int(len(x)), uniq_mean=round(float(u.mean()), 3), uniq_max=int(u.max()), frac_le16=round(float((u <= 16).mean()), 4), frac_on_grid=round(float((err < 0.02).mean()), 4))
rows2 = np.sort(rng.choice(262144, 256, replace=False)).tolist()
G = {}
for tag in ('it', 'qat'):
    for k in ('embed_tokens.weight', 'embed_tokens_per_layer.weight'):
        x = hf(tag, L + k, rows2); G[f'{tag}/{k}'] = dict(last=grid(x, -1), first=grid(x, 0))
    for k in ('per_layer_model_projection.weight', 'layers.0.per_layer_projection.weight', 'layers.0.mlp.up_proj.weight', 'layers.34.self_attn.q_proj.weight', 'layers.4.self_attn.o_proj.weight'):
        x = hf(tag, L + k); G[f'{tag}/{k}'] = dict(last=grid(x, -1), first=grid(x, 0))
    for k in ('model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight', 'model.audio_tower.layers.0.self_attn.q_proj.linear.weight', 'model.embed_audio.embedding_projection.weight'):
        x = hf(tag, k); G[f'{tag}/{k}'] = dict(last=grid(x, -1), first=grid(x, 0)) if x is not None else None
for k, v in G.items(): P(f'  {k}: {json.dumps(v, ensure_ascii=False)}')
res['q4_grid'] = G
open(f'{R}/orbax_vs_hf_compare.txt', 'w').write('\n'.join(lines) + '\n')
json.dump(res, open(f'{R}/orbax_vs_hf_compare.json', 'w'), indent=1, ensure_ascii=False)
P('DONE')
