# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Axis-aware block quantization (gemma4-qat patch, M4).

The upstream `_q4_0` always blocks along the *last* kernel axis
(`_quantization._pack`). For Gemma 4 the official QAT checkpoint is blocked
along the HF *input* axis, i.e. the einsum *contraction* axes of the kernel
(trace docs: docs/Plan/trace/04_weights_map.md §5, 05_quantization_internals.md
§3-4). This module blocks along arbitrary axes:

  kernel --transpose--> [non-contracted..., contracted...] --reshape-->
  [-1, prod(contracted)] --reshape--> [-1, block_size] -> quantize -> back.

For multiple contracted axes (e.g. attn_vec `NHD`, contracted N,H) the
flattening order (N major, H minor) equals the HF input index n*H + h, so
blocks match the HF layout as long as the minor axis is a multiple of
`block_size`.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import jax
import jax.numpy as jnp

Array = jax.Array


def contraction_axes(eqn: str, kernel_ndim: int | None = None) -> tuple[int, ...]:
  """Kernel axes that are contracted in a 2-operand einsum `x,kernel->out`."""
  eqn = eqn.replace(' ', '')
  lhs, out = eqn.split('->')
  x_spec, k_spec = lhs.split(',')
  k_spec_clean = k_spec.replace('...', '')
  x_letters = set(x_spec.replace('...', ''))
  out_letters = set(out.replace('...', ''))
  if '...' in k_spec:
    raise ValueError(f'Ellipsis in kernel spec is not supported: {eqn}')
  axes = tuple(
      i for i, c in enumerate(k_spec_clean)
      if c in x_letters and c not in out_letters
  )
  if kernel_ndim is not None and len(k_spec_clean) != kernel_ndim:
    raise ValueError(f'Kernel ndim {kernel_ndim} does not match {eqn}')
  if not axes:
    raise ValueError(f'No contraction axis found in {eqn}')
  return axes


def _quantize_blocks(blocks: Array, formula: str, scale_dtype) -> Array:
  """blocks: [n, block] float32 -> fake-quantized float32."""
  if formula == 'minmax':
    mn = jnp.min(blocks, axis=-1, keepdims=True)
    mx = jnp.max(blocks, axis=-1, keepdims=True)
    d = jnp.maximum(-mn / 8.0, mx / 7.0)
    if scale_dtype is not None:
      d = d.astype(scale_dtype).astype(jnp.float32)
    safe_d = jnp.where(d == 0.0, 1.0, d)
    q = jnp.clip(jnp.round(blocks / safe_d), -8.0, 7.0)
    return jnp.where(d == 0.0, 0.0, q * d)
  if formula == 'absmax':  # d = absmax / 8, clip to [-8, 7]
    d = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True) / 8.0
    if scale_dtype is not None:
      d = d.astype(scale_dtype).astype(jnp.float32)
    safe_d = jnp.where(d == 0.0, 1.0, d)
    q = jnp.clip(jnp.round(blocks / safe_d), -8.0, 7.0)
    return jnp.where(d == 0.0, 0.0, q * d)
  if formula == 'q4_0':  # upstream `_q4_0` math (signed absmax -> level -8)
    idx = jnp.argmax(jnp.abs(blocks), axis=-1, keepdims=True)
    m = jnp.take_along_axis(blocks, idx, axis=-1)
    d = -m / 8.0
    safe_d = jnp.where(d == 0.0, 1.0, d)
    inv = jnp.where(d == 0.0, 0.0, 1.0 / safe_d)
    q = jnp.clip(jnp.trunc(blocks * inv + 8.5), 0.0, 15.0)
    d = d.astype(scale_dtype or jnp.float16).astype(jnp.float32)
    return d * (q - 8.0)
  raise ValueError(f'Unknown formula {formula!r}')


def block_quantize(
    x: Array,
    axes: Sequence[int],
    *,
    block_size: int = 32,
    formula: str = 'minmax',
    scale_dtype=None,
) -> Array:
  """Fake-quantizes `x` in blocks of `block_size` along `axes` (no STE)."""
  nd = x.ndim
  axes = tuple(sorted(a % nd for a in axes))
  rest = tuple(i for i in range(nd) if i not in axes)
  perm = rest + axes
  xt = jnp.transpose(x, perm)
  inner = math.prod(x.shape[a] for a in axes)
  if inner % block_size != 0 or x.shape[axes[-1]] % block_size != 0:
    raise ValueError(
        f'Block size {block_size} must divide the minor contracted axis '
        f'(shape {x.shape}, axes {axes}).'
    )
  blocks = xt.reshape(-1, block_size).astype(jnp.float32)
  q = _quantize_blocks(blocks, formula, scale_dtype)
  q = q.reshape(xt.shape).astype(x.dtype)
  inv = [0] * nd
  for i, p in enumerate(perm):
    inv[p] = i
  return jnp.transpose(q, inv)


def simulate_block_quantize(x: Array, axes: Sequence[int], **kwargs) -> Array:
  """Straight-through estimator version: forward = Q(x), d/dx = identity."""
  return x + jax.lax.stop_gradient(block_quantize(x, axes, **kwargs) - x)


def method_to_formula(method) -> str | None:
  """Maps a `QuantizationMethod` to a block formula (None = not block)."""
  m = str(method).upper().split('.')[-1]
  return {'Q4_0': 'q4_0', 'Q4_0_TRANSPOSE': 'q4_0', 'Q4_0_MINMAX': 'minmax',
          'Q4_0_ABSMAX': 'absmax'}.get(m)
