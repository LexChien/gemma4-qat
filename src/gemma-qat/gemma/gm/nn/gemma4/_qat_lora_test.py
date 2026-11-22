# Copyright 2026 DeepMind Technologies Limited.
# Licensed under the Apache License, Version 2.0.
"""gemma4-qat (M5) tests: mergeable LoRA-QAT."""
import jax
import jax.numpy as jnp
import numpy as np
from gemma import peft
from gemma.gm.nn.gemma4 import _qat
from gemma.gm.nn.gemma4 import _qat_lora
from gemma.gm.nn.gemma4._qat_test import _tiny_model


def _setup(method):
  base = _tiny_model()
  tok = jax.random.randint(jax.random.PRNGKey(3), (2, 16), 1, 512)
  m = _qat_lora.QATLoRA(model=_tiny_model(), rank=4, alpha=8.0, method=method)
  p = m.init(jax.random.PRNGKey(0), tok)['params']
  p = jax.tree.map(lambda a: a.astype(jnp.float32), p)
  # make B non-zero so the adapter matters
  keys = iter(jax.random.split(jax.random.PRNGKey(7), 1000))
  p = jax.tree_util.tree_map_with_path(
      lambda path, a: 0.05 * jax.random.normal(next(keys), a.shape) if 'lora_b' in jax.tree_util.keystr(path) else a, p)
  return base, m, p, tok


def test_lora_param_tree():
  base, m, p, tok = _setup(peft.QuantizationMethod.Q4_0_MINMAX)
  lo, ba = _qat_lora.split_lora(p)
  pb = base.init(jax.random.PRNGKey(0), tok)['params']
  assert jax.tree_util.tree_structure(ba) == jax.tree_util.tree_structure(pb)
  n = len(jax.tree_util.tree_leaves(lo))
  assert n == 2 * 5 * 5, n   # 5 layers x (q, kv, attn_vec, gating, linear) x (a, b)
  ks = {jax.tree_util.keystr(k) for k, _ in jax.tree_util.tree_leaves_with_path(lo)}
  assert "['layer_0']['mlp']['gating_einsum_lora_a']" in ks and "['layer_0']['attn']['q_einsum']['lora_a']" in ks, sorted(ks)[:6]


def _forward_check(method, formula):
  if True:
    base, m, p, tok = _setup(method)
    y = m.apply({'params': p}, tok).logits
    merged = _qat_lora.merge_lora(p, rank=4, alpha=8.0)
    ref = base.apply({'params': _qat.snap_params(merged, formula=formula)}, tok).logits
    _, ba = _qat_lora.split_lora(p)
    y0 = base.apply({'params': _qat.snap_params(ba, formula=formula)}, tok).logits
    err = float(jnp.max(jnp.abs(y - ref)))
    assert err < 1e-4, (formula, err)
    assert float(jnp.max(jnp.abs(y - y0))) > 1e-3   # adapter changes the output


def test_forward_equals_merged_snapped_minmax():
  _forward_check(peft.QuantizationMethod.Q4_0_MINMAX, 'minmax')


def test_forward_equals_merged_snapped_absmax_subprocess():
  # KNOWN LIMITATION: within one process, flax re-uses the module replacement
  # of the first QAT wrapper traced for a given model config, so a second
  # wrapper with another method silently keeps the first formula. Use one QAT
  # method per process (all M4/M5 scripts do); check ABSMAX in a fresh process.
  import subprocess, sys
  code = ("from gemma import peft; from gemma.gm.nn.gemma4._qat_lora_test import _forward_check; "
          "_forward_check(peft.QuantizationMethod.Q4_0_ABSMAX, 'absmax'); print('OK')")
  r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env={**__import__('os').environ, 'JAX_PLATFORMS': 'cpu'})
  assert 'OK' in r.stdout, r.stderr[-2000:]


def test_minmax_snap_idempotent_after_training_formula():
  x = jax.random.normal(jax.random.PRNGKey(5), (64, 96))
  from gemma.peft import _block_quantization as bq
  qa = bq.block_quantize(x, (0,), formula='absmax')
  np.testing.assert_allclose(bq.block_quantize(qa, (0,), formula='minmax'), qa, atol=1e-6)


def test_grad_only_lora():
  base, m, p, tok = _setup(peft.QuantizationMethod.Q4_0_MINMAX)
  lo, ba = _qat_lora.split_lora(p)
  def merge(a, b):
    def rec(x, y):
      out = dict(y)
      for k, v in x.items():
        out[k] = rec(v, y.get(k, {})) if isinstance(v, dict) else v
      return out
    return rec(a, b)
  g = jax.grad(lambda l: m.apply({'params': merge(l, ba)}, tok).logits.std())(lo)
  for k, x in jax.tree_util.tree_leaves_with_path(g):
    ks = jax.tree_util.keystr(k)
    shared_kv = 'kv_einsum' in ks and ("'layer_3'" in ks or "'layer_4'" in ks)  # KV-shared layers: unused
    assert (float(jnp.abs(x).sum()) > 0) != shared_kv, ks


if __name__ == '__main__':
  test_lora_param_tree(); test_forward_equals_merged_snapped_minmax(); test_minmax_snap_idempotent_after_training_formula(); test_grad_only_lora()
  print('ok')
