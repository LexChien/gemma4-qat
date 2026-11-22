# Copyright 2026 DeepMind Technologies Limited.
# Licensed under the Apache License, Version 2.0.
"""gemma4-qat (M4) unit tests: QAT wrapper on Gemma 4 E2B (abstract shapes)."""
import collections
import jax
import jax.numpy as jnp
import numpy as np
from gemma import gm
from gemma import peft
from gemma.gm.nn import _quantization as gq
from gemma.peft import _block_quantization as bq

TOK = jnp.zeros((1, 8), jnp.int32)


def _abstract_params(model):
  return jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), TOK)['params'])


def test_contraction_axes():
  assert bq.contraction_axes('BTD,NDH->BTNH') == (1,)
  assert bq.contraction_axes('BSD,CKDH->CBSKH') == (2,)
  assert bq.contraction_axes('BTNH,NHD->BTD') == (0, 1)
  assert bq.contraction_axes('...F,NHF->...NH') == (2,)
  assert bq.contraction_axes('...H,HF->...F') == (0,)
  assert bq.contraction_axes('...td,dnp->...tnp') == (0,)
  assert bq.contraction_axes('...D,DP->...P') == (0,)
  assert bq.contraction_axes('...P,PD->...D') == (0,)


def test_block_quantize_minmax_levels():
  x = jax.random.normal(jax.random.PRNGKey(1), (64, 96))
  q = bq.block_quantize(x, (0,), formula='minmax')
  blocks = np.asarray(jnp.transpose(q, (1, 0))).reshape(-1, 32)
  for b in blocks:
    d = max(-b.min() / 8, b.max() / 7)
    lv = b / d
    assert np.allclose(lv, np.round(lv), atol=1e-4) and lv.min() >= -8 - 1e-4 and lv.max() <= 7 + 1e-4
  # idempotent
  np.testing.assert_allclose(bq.block_quantize(q, (0,), formula='minmax'), q, atol=1e-6)


def test_absmax_formula():
  x = jnp.array([[0.8] + [0.1] * 31, [-0.8] + [0.1] * 31])
  q = np.asarray(bq.block_quantize(x, (1,), formula='absmax'))
  np.testing.assert_allclose(q[:, 0], [0.7, -0.8], atol=1e-6)   # +absmax -> +7*d (clipped), -absmax -> -8*d
  y = peft.simulate_quantize(x, peft.QuantizationMethod.Q4_0_ABSMAX)
  np.testing.assert_allclose(y, q, atol=1e-6)


def test_ste_gradient_identity():
  x = jax.random.normal(jax.random.PRNGKey(2), (32, 32))
  g = jax.grad(lambda w: bq.simulate_block_quantize(w, (0,)).sum())(x)
  np.testing.assert_array_equal(g, jnp.ones_like(x))


def test_wrapper_replaces_and_keeps_param_tree():
  base = gm.nn.Gemma4_E2B(text_only=True)
  qat = gm.nn.QuantizationAwareWrapper(model=gm.nn.Gemma4_E2B(text_only=True),
                                       method=peft.QuantizationMethod.Q4_0_MINMAX,
                                       log_replacements=True)
  p0 = _abstract_params(base)
  gq.LAST_REPLACEMENTS.clear()
  p1 = _abstract_params(qat)
  s0 = jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), p0)
  s1 = jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), p1)
  assert jax.tree_util.tree_structure(p0) == jax.tree_util.tree_structure(p1)
  assert s0 == s1
  kinds = collections.Counter(k for _, k, _ in gq.LAST_REPLACEMENTS)
  assert kinds['Einsum'] > 0, kinds
  return kinds, len(jax.tree_util.tree_leaves(p0))


def _tiny_model():
  import dataclasses
  from gemma.gm.nn.gemma4 import _config, _modules
  base = gm.nn.Gemma4_E2B.config
  pat = (_modules.AttentionType.LOCAL_SLIDING,) * 4 + (_modules.AttentionType.GLOBAL,)
  cfg = dataclasses.replace(
      base, num_embed=512, embed_dim=128, hidden_dim=512, head_dim=64,
      global_key_size=64, num_heads=2, per_layer_input_dim=32,
      attention_types=_config.make_attention_layers_types(pat, num_layers=5),
      kv_cache_sharing_config=_config.KVCacheSharingConfig(
          frac_shared_layers=0.4, share_global=True, share_local=True),
      override_kv_shared_ffw_hidden=1024, sliding_window_size=16,
      vision_encoder=None, audio_encoder=None)
  return gm.nn.Gemma4_E2B(config=cfg, text_only=True)


def test_qat_forward_equals_fake_quantized_params():
  from gemma.gm.nn.gemma4 import _qat
  base = _tiny_model()
  tok = jax.random.randint(jax.random.PRNGKey(3), (2, 16), 1, 512)
  p = base.init(jax.random.PRNGKey(0), tok)['params']
  p = jax.tree.map(lambda a: a.astype(jnp.float32), p)
  qat = gm.nn.QuantizationAwareWrapper(model=_tiny_model(),
                                       method=peft.QuantizationMethod.Q4_0_MINMAX,
                                       log_replacements=True)
  gq.LAST_REPLACEMENTS.clear()
  y_qat = qat.apply({'params': p}, tok).logits
  y_ref = base.apply({'params': _qat.snap_params(p)}, tok).logits
  y_fp = base.apply({'params': p}, tok).logits
  err = float(jnp.max(jnp.abs(y_qat - y_ref)))
  diff_fp = float(jnp.max(jnp.abs(y_qat - y_fp)))
  assert err < 1e-4, err
  assert diff_fp > 1e-3, diff_fp   # quantization actually did something
  n = sum(1 for _, k, _ in gq.LAST_REPLACEMENTS if k == 'Einsum')
  return err, diff_fp, n


def test_int_wrapper_shapes():
  m = gm.nn.IntWrapper(model=_tiny_model(), dtype=jnp.int4)
  tok = jnp.ones((1, 8), jnp.int32)
  shapes = jax.eval_shape(lambda: m.init(jax.random.PRNGKey(0), tok))['params']
  leaves = jax.tree_util.tree_leaves_with_path(shapes)
  ks = [jax.tree_util.keystr(k) for k, _ in leaves]
  assert any('kernel' in k for k in ks) and any('scale' in k for k in ks), ks[:20]
  out = jax.eval_shape(lambda: m.apply(m.init(jax.random.PRNGKey(0), tok), tok))
  assert out.logits.shape == (1, 8, 512), out.logits.shape
  return len(ks)


def test_clipped_einsum():
  import flax.linen as nn
  from gemma.gm.nn.gemma4 import _layers

  class M(nn.Module):
    @nn.compact
    def __call__(self, x):
      return _layers.ClippedEinsum(shape=(64, 48), name='proj')('...H,HF->...F', x)

  x = 3 * jax.random.normal(jax.random.PRNGKey(4), (4, 64))
  m = M()
  p = m.init(jax.random.PRNGKey(0), x)['params']
  p = jax.tree.map(lambda a: a, p)
  p['proj'] = dict(p['proj'], clip_input_min=jnp.array(-2.), clip_input_max=jnp.array(2.),
                   clip_output_min=jnp.array(-.5), clip_output_max=jnp.array(.5))
  qm = gm.nn.QuantizationAwareWrapper(model=M(), method=peft.QuantizationMethod.Q4_0_MINMAX,
                                      quantize_towers=True)
  pq = qm.init(jax.random.PRNGKey(0), x)['params']
  assert jax.tree_util.tree_structure(pq) == jax.tree_util.tree_structure(p)
  y = qm.apply({'params': p}, x)
  wq = bq.block_quantize(p['proj']['w'], (0,))
  ref = jnp.clip(jnp.clip(x, -2, 2) @ wq, -.5, .5)
  np.testing.assert_allclose(y, ref, atol=1e-5)
  return float(jnp.max(jnp.abs(y - ref)))


if __name__ == '__main__':
  print('clipped:', test_clipped_einsum())
  print('fwd-equiv (err, |qat-fp|, n_wrapped):', test_qat_forward_equals_fake_quantized_params())
  print('intwrapper leaves:', test_int_wrapper_shapes())
  test_contraction_axes(); test_block_quantize_minmax_levels(); test_ste_gradient_identity()
  print('wrapper:', test_wrapper_replaces_and_keeps_param_tree())
