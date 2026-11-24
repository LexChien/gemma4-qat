# M11 probe: compare MLX g32exact dequant vs official unquantized bf16 for selected tensors.
import sys, numpy as np, torch, safetensors
sys.path.insert(0, 'src'); from g4qat.grid import unpack_mlx
OFF = '~/Documents/AI/ALife-Platform/Google-Gemma4/originals/models/gemma4-e2b-qat/model.safetensors'
M = sys.argv[1]
fo = safetensors.safe_open(OFF, 'pt'); fm = safetensors.safe_open(M + '/model.safetensors', 'pt')
names = sys.argv[2:]
for n in names:
    hf = 'model.language_model.' + n; ml = 'language_model.model.' + n.replace('.weight', '')
    o = fo.get_tensor(hf).float()
    L = unpack_mlx(fm.get_tensor(ml + '.weight')).float(); s = fm.get_tensor(ml + '.scales').float(); b = fm.get_tensor(ml + '.biases').float()
    r, c = L.shape
    w = (L.reshape(r, c // 32, 32) * s[..., None] + b[..., None]).reshape(r, c)
    wb = w.to(torch.bfloat16).float()
    bias_ok = torch.equal(b, -8 * s)
    has_m8 = (L.reshape(r, c//32, 32) == 0).any(-1)
    nz = (s != 0)
    print(n, 'shape', tuple(o.shape), 'f32_eq', (w == o).float().mean().item(), 'bf16_eq', (wb == o).float().mean().item(),
          'bias=-8s', bias_ok, 'groups_without_-8', int((~has_m8 & nz).sum()), '/', has_m8.numel(), 'maxdiff', (w - o).abs().max().item(), flush=True)
