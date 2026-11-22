"""Export QAT-trained params -> fake-quantized (Q4_0-snapped) bf16 safetensors with JAX names.
APIs: gemma.peft.simulate_quantize (gemma/peft/_quantization.py), gm.ckpts.load_params.
Snapping weights onto the Q4_0 grid means a later 4-bit converter loses ~nothing.
NOT PROVIDED BY REPO: JAX->HF transformers name/shape mapping for Gemma 4 (no safetensors/HF
code exists in the repo). You must write/obtain a converter (e.g. transformers' own
convert_gemma4 script, if present in your transformers version — UNVERIFIED) to produce an HF
folder, then: MLX  -> mlx_lm convert -q --q-bits 4 --q-group-size 32 (match Q4_0 block 32)
          compressed-tensors -> ../quantize_w4a16.py
Usage: python export_params.py --ckpt workdir/checkpoints/ckpt_2000 --out export/
"""
import argparse, os
import numpy as np, jax, jax.numpy as jnp
from flax import traverse_util
from safetensors.numpy import save_file
from gemma import peft
from orbax import checkpoint as ocp

p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True)
p.add_argument("--no_snap", action="store_true")
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
state = ocp.PyTreeCheckpointer().restore(a.ckpt)   # kauldron ckpt layout: check keys, params may be under 'params'
params = state.get("params", state)
flat = traverse_util.flatten_dict(params, sep=".")
out = {}
for k, v in flat.items():
  v = jnp.asarray(v)
  if not a.no_snap and "_SimulateQuantizedEinsum" in k and v.ndim >= 2:
    v = peft.simulate_quantize(v.astype(jnp.float32), peft.QuantizationMethod.Q4_0)
  k = k.replace("._SimulateQuantizedEinsum_0.kernel", ".w").replace("._SimulateQuantizedEinsum_", ".")
  out[k] = np.asarray(v.astype(jnp.float32)).astype(np.float16 if v.dtype != jnp.int32 else np.int32)
save_file(out, os.path.join(a.out, "gemma4_e2b_qat_jaxnames.safetensors"))
print(len(out), "tensors ->", a.out)
