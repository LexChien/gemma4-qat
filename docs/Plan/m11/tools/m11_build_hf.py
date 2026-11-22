"""M11 step 1: MLX g32exact (M8/M9) -> text-only HF-layout safetensors for the gemma.cpp converter.
Quantized tensors are dequantized exactly (L*s+b, b=-8s). Tensors the SBS stores as bf16
(embed_tokens, per_layer_model_projection) are written f32 so the converter does the only rounding;
Q4_0-target tensors are written bf16 (their Q4_0 blocks are later replaced bit-exactly by m11_exactify.py).
usage: m11_build_hf.py <mlx_dir> <out.safetensors>"""
import sys, os, torch, safetensors, safetensors.torch
sys.path.insert(0, 'src'); from g4qat.grid import unpack_mlx
src, out = sys.argv[1], sys.argv[2]
assert not os.path.exists(out), 'refuse overwrite'
F32 = {'embed_tokens', 'per_layer_model_projection'}
f = safetensors.safe_open(src + '/model.safetensors', 'pt')
keys = [k for k in f.keys() if k.startswith('language_model.model.')]
outd = {}
bases = sorted({k.rsplit('.', 1)[0] for k in keys if k.endswith('.scales')})
for i, b in enumerate(bases):
    L = unpack_mlx(f.get_tensor(b + '.weight')); s = f.get_tensor(b + '.scales').float(); bb = f.get_tensor(b + '.biases').float()
    assert torch.equal(bb, -8 * s), b
    r, c = L.shape
    w = (L.float().reshape(r, c // 32, 32) * s[..., None] + bb[..., None]).reshape(r, c)
    name = 'model.language_model.' + b[len('language_model.model.'):] + '.weight'
    outd[name] = w if b.split('.')[-1] in F32 else w.to(torch.bfloat16)
    if i % 20 == 0: print(f'dequant {i}/{len(bases)} {name}', flush=True)
for k in keys:
    if k.rsplit('.', 1)[0] in bases: continue
    outd['model.language_model.' + k[len('language_model.model.'):]] = f.get_tensor(k)
print('tensors', len(outd), flush=True)
safetensors.torch.save_file(outd, out + '.partial', metadata={'format': 'pt', 'm11_source': src})
os.rename(out + '.partial', out); print('BUILD_HF_DONE', out)
