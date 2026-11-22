# Copyright 2026 DeepMind Technologies Limited.
# Licensed under the Apache License, Version 2.0.
"""gemma4-qat (M5): LoRA-QAT for Gemma 4 with a *mergeable* formulation.

Forward of an adapted Einsum:  y = einsum(x, Q(W + s * A @ B))   (STE through Q)
so that the exported merged weight `snap(W + s*A@B)` gives exactly the training
forward (unlike `gm.nn.LoRA`, whose `Q(W) + BA` cannot be merged onto a 4-bit
grid; `gm.nn.LoRA` also has no `fuse_params`).

* A: [in, r] (in = einsum contraction axes = HF input axis), B: [r, out] (zeros).
* Params live in the wrapped module scope (shared scope):
  `<scope>/{w, lora_a, lora_b}`, or `<scope>/{<name>_lora_a, <name>_lora_b}`
  for mlp `gating_einsum` / `linear` (they share the FeedForward scope).
* Non-adapted Einsums get the plain QAT wrapper; embed / PLE / lm_head are
  fake-quantized through `_qat.embed_qat` (tables usually frozen).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

from flax import linen as nn
from gemma import peft
from gemma.gm.nn import _quantization as _gq
from gemma.gm.nn.gemma4 import _layers
from gemma.gm.nn.gemma4 import _qat
from gemma.peft import _block_quantization as bq
import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_TARGETS = ('q_einsum', 'kv_einsum', 'k_einsum', 'attn_vec_einsum',
                   'gating_einsum', 'linear')


def _names(weight_name: str) -> tuple[str, str]:
  if weight_name == 'w':
    return 'lora_a', 'lora_b'
  return f'{weight_name}_lora_a', f'{weight_name}_lora_b'


def lora_delta(a, b, shape, axes, scale):
  """(A @ B) reshaped back to the kernel `shape` (contraction `axes` first)."""
  nd = len(shape)
  rest = tuple(i for i in range(nd) if i not in axes)
  perm = tuple(axes) + rest
  d = (a.astype(jnp.float32) @ b.astype(jnp.float32)) * scale
  d = d.reshape([shape[i] for i in perm])
  return jnp.transpose(d, np.argsort(perm))


class _QATLoRAEinsum(nn.Module):
  _: dataclasses.KW_ONLY
  shape: tuple[int, ...]
  weight_name: str
  formula: str
  rank: int
  alpha: float
  block_size: int = 32
  initializer: Any = nn.initializers.normal()
  param_dtype: Any = None
  w_scale: float | None = None
  wrapped: nn.Module

  def __post_init__(self):
    super().__post_init__()
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    w = self.param(self.weight_name, self.initializer, self.shape, self.param_dtype)
    if isinstance(w, dict) and 'w' in w:
      w = w['w']
    axes = bq.contraction_axes(eqn, w.ndim)
    rest = tuple(i for i in range(w.ndim) if i not in axes)
    d_in = math.prod(self.shape[i] for i in axes)
    d_out = math.prod(self.shape[i] for i in rest)
    na, nb = _names(self.weight_name)
    a = self.param(na, nn.initializers.normal(1.0 / math.sqrt(d_in)), (d_in, self.rank), jnp.float32)
    b = self.param(nb, nn.initializers.zeros, (self.rank, d_out), jnp.float32)
    w_eff = w.astype(jnp.float32) + lora_delta(a, b, self.shape, axes, self.alpha / self.rank)
    wq = bq.simulate_block_quantize(w_eff, axes, block_size=self.block_size, formula=self.formula)
    if self.w_scale is not None:
      wq = wq * self.w_scale
    return jnp.einsum(eqn, x, wq.astype(x.dtype) if x.dtype != jnp.float32 else wq)


def _replace(module, *, formula, method, rank, alpha, targets, block_size):
  if isinstance(module, _layers.Einsum):
    path = _gq._module_path(module)  # pylint: disable=protected-access
    if any(p.startswith(_gq._TOWER_PREFIXES) for p in path) or (path and path[-1] in _gq._SKIP_NAMES):  # pylint: disable=protected-access
      return module
    name = module.weight_name if module.weight_name != 'w' else (path[-1] if path else module.name)
    if name in targets:
      return _QATLoRAEinsum(
          name=module.name + 'qat', shape=module.shape, weight_name=module.weight_name,
          formula=formula, rank=rank, alpha=alpha, block_size=block_size,
          initializer=module.initializer, param_dtype=module.dtype,
          w_scale=module.w_scale, wrapped=module)
    return _gq._Gemma4SimulateQuantizedEinsum(  # pylint: disable=protected-access
        name=module.name + 'qat', method=method, block_size=block_size, shape=module.shape,
        weight_name=module.weight_name, initializer=module.initializer, param_dtype=module.dtype,
        w_scale=module.w_scale, clipped=False, wrapped=module)
  return module


class QATLoRA(nn.Module):
  """LoRA-QAT wrapper (mergeable). See module docstring."""
  _: dataclasses.KW_ONLY
  model: nn.Module
  rank: int = 16
  alpha: float = 32.0
  method: peft.QuantizationMethod = peft.QuantizationMethod.Q4_0_MINMAX
  block_size: int = 32
  targets: tuple[str, ...] = DEFAULT_TARGETS
  quantize_embeddings: bool = True
  quantize_ple: bool = True
  quantize_lm_head: bool = True

  def __post_init__(self):
    super().__post_init__()
    if self.scope is not None:
      nn.share_scope(self, self.model)

  @nn.compact
  def __call__(self, *args, **kwargs):
    formula = bq.method_to_formula(self.method)
    cfg = _qat.EmbedQATConfig(formula=formula, quantize_embed=self.quantize_embeddings,
                              quantize_ple=self.quantize_ple, quantize_lm_head=self.quantize_lm_head)
    fn = lambda m: _replace(m, formula=formula, method=self.method, rank=self.rank, alpha=self.alpha,  # pylint: disable=g-long-lambda
                            targets=self.targets, block_size=self.block_size)
    with _qat.embed_qat(cfg):
      with peft.ModuleInterceptor(fn):
        return self.model(*args, **kwargs)


def is_lora_leaf(path) -> bool:
  k = path[-1]
  k = getattr(k, 'key', k)
  return isinstance(k, str) and (k.startswith('lora_') or '_lora_' in k)


def split_lora(params):
  """-> (lora_params, base_params) as nested dicts (disjoint leaves)."""
  def rec(p):
    lo, ba = {}, {}
    for k, v in p.items():
      if isinstance(v, dict):
        l, b = rec(v)
        if l: lo[k] = l
        if b: ba[k] = b
      elif k.startswith('lora_') or '_lora_' in k:
        lo[k] = v
      else:
        ba[k] = v
    return lo, ba
  return rec(params)


def merge_lora(params, *, rank: int, alpha: float):
  """Folds every LoRA pair into its kernel (f32); returns a tree without LoRA leaves.

  Uses `_qat.CONTRACTION_AXES_BY_NAME`, so it matches `_QATLoRAEinsum`. Snap the
  result with `_qat.snap_params(..., formula=<training formula>)` to get exactly
  the training forward weights.
  """
  s = alpha / rank
  def rec(p, name=None):
    out = {}
    for k, v in p.items():
      if k.startswith('lora_') or '_lora_' in k:
        continue
      if isinstance(v, dict):
        if 'lora_a' in v and 'w' in v:  # <scope>/{w, lora_a, lora_b}
          ax = _qat.CONTRACTION_AXES_BY_NAME[k]
          w = v['w']
          out[k] = dict(rec({kk: vv for kk, vv in v.items() if kk not in ('w',)}, k),
                        w=(w.astype(jnp.float32) + lora_delta(v['lora_a'], v['lora_b'], w.shape, ax, s)))
          continue
        out[k] = rec(v, k)
      else:
        out[k] = v
    # mlp scope: gating_einsum / linear with <name>_lora_{a,b} siblings
    for nm in ('gating_einsum', 'linear'):
      if f'{nm}_lora_a' in p and nm in p:
        w = p[nm]['w'] if isinstance(p[nm], dict) else p[nm]
        merged = w.astype(jnp.float32) + lora_delta(p[f'{nm}_lora_a'], p[f'{nm}_lora_b'], w.shape,
                                                     _qat.CONTRACTION_AXES_BY_NAME[nm], s)
        out[nm] = {'w': merged} if isinstance(p[nm], dict) else merged
    return out
  return rec(params)
