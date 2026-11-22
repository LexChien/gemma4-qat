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

"""Quantization Aware Training (QAT) wrapper around Gemma models."""

import dataclasses
import functools
from typing import Any

from flax import linen as nn
from flax.linen import dtypes as flax_dtypes
from gemma import peft
from gemma.gm.nn import _layers
from gemma.gm.nn.gemma4 import _layers as _g4_layers
from gemma.gm.nn.gemma4 import _qat as _g4_qat
from gemma.peft import _block_quantization
from gemma.peft import _quantization
import jax
from jax import numpy as jnp
from kauldron import kontext


class QuantizationAwareWrapper(nn.Module):
  """Wrapper around a Gemma model to enable quantization aware training.

  The model wrapped will have all it's `nn.Dense`, `nn.Einsum`,... layers
  replaced by their quantization aware training versions. See `gemma.peft`
  documentation for more details.

  Attributes:
    method: The quantization method to use.
    model: The model to wrap.
  """

  _: dataclasses.KW_ONLY

  method: peft.QuantizationMethod = peft.QuantizationMethod.NONE
  model: nn.Module
  # gemma4-qat (M4) options. Only used for Gemma 4 layers / embeddings.
  block_size: int = 32
  quantize_embeddings: bool = True   # input_embedding (+ tied lm_head)
  quantize_ple: bool = True          # per_layer_embeddings table
  quantize_lm_head: bool = True
  embed_block_size: int = 32
  ple_block_size: int = 32
  quantize_towers: bool = False      # vision/audio ClippedEinsum (official QAT: not quantized)
  log_replacements: bool = False     # record replaced modules in LAST_REPLACEMENTS

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': model_params}}` rather than
    # `{'params': {'model': model_params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.model)

  @nn.compact
  def __call__(self, *args, **kwargs):
    """Calls the model."""
    # NOTE: LAST_REPLACEMENTS is appended to (not cleared): flax only calls the
    # replace fn when a module is first intercepted in the process, so later
    # applies/traces (jit, grad) log nothing although quantization is active.
    # Clear it manually before the call you want to inspect.
    replace_module_fn = functools.partial(
        _replace_by_simulated_quantization,
        method=self.method,
        block_size=self.block_size,
        quantize_towers=self.quantize_towers,
        log=self.log_replacements,
    )
    formula = _block_quantization.method_to_formula(self.method)
    embed_cfg = None
    if formula is not None and (self.quantize_embeddings or self.quantize_ple):
      embed_cfg = _g4_qat.EmbedQATConfig(
          formula=formula,
          embed_block_size=self.embed_block_size,
          ple_block_size=self.ple_block_size,
          quantize_embed=self.quantize_embeddings,
          quantize_ple=self.quantize_ple,
          quantize_lm_head=self.quantize_lm_head,
      )
    with _g4_qat.embed_qat(embed_cfg):
      with peft.ModuleInterceptor(replace_module_fn):
        return self.model(*args, **kwargs)

  def __kontext_keys__(self) -> dict[str, str]:
    """Kauldron keys when calling `kontext.get_from_keys_obj`."""
    # Forward the keys from the wrapped model.
    # This allow to define the config as:
    # gm.nn.QuantizationAwareWrapper(
    #   model=MyModel(
    #     input='batch.input',  # propagated to wrapper
    #   ),
    # )
    return kontext.get_keypaths(self.model)  # pyrefly: ignore[bad-return]

  def __getattr__(self, name: str) -> Any:
    # Forward attribute accesses to the wrapped model.
    return getattr(self.model, name)


class IntWrapper(nn.Module):
  """Wrapper around a Gemma model to enable int4 inference.

  The model wrapped will have all it's `nn.Dense`, `nn.Einsum`,... layers
  replaced by their int4 versions. See `gemma.peft` documentation for more
  details.

  Attributes:
    model: The model to wrap.
  """

  _: dataclasses.KW_ONLY

  model: nn.Module
  dtype: jnp.dtype = jnp.int4  # pyrefly: ignore[bad-assignment]

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': model_params}}` rather than
    # `{'params': {'model': model_params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.model)

  @nn.compact
  def __call__(self, *args, **kwargs):
    """Calls the model."""
    replace_module_fn = functools.partial(_replace_by_int, dtype=self.dtype)
    with peft.ModuleInterceptor(replace_module_fn):
      return self.model(*args, **kwargs)

  def __kontext_keys__(self) -> dict[str, str]:
    """Kauldron keys when calling `kontext.get_from_keys_obj`."""
    # Forward the keys from the wrapped model.
    # This allow to define the config as:
    # gm.nn.QuantizationAwaretrainingWrapper(
    #   model=MyModel(
    #     input='batch.input',  # keys propagated to the `IntWrapper`
    #   ),
    # )
    return kontext.get_keypaths(self.model)  # pyrefly: ignore[bad-return]

  def __getattr__(self, name: str) -> Any:
    # Forward attribute accesses to the wrapped model.
    return getattr(self.model, name)


LAST_REPLACEMENTS: list[tuple[str, str, tuple[int, ...] | None]] = []

_TOWER_PREFIXES = ('vision_encoder', 'audio_encoder')
# Official QAT leaves the multimodal soft-token projections in bf16
# (trace 04 §5: embed_vision / embed_audio not on the Q4 grid).
_SKIP_NAMES = ('mm_input_projection', 'audio_input_projection')


def _module_path(module: nn.Module) -> tuple[str, ...]:
  try:
    return tuple(module.path)
  except Exception:  # pylint: disable=broad-except
    return ()


def _replace_by_simulated_quantization(
    module: nn.Module,
    *,
    method: peft.QuantizationMethod,
    block_size: int = 32,
    quantize_towers: bool = False,
    log: bool = False,
):
  if isinstance(module, (_g4_layers.Einsum, _g4_layers.ClippedEinsum)):
    path = _module_path(module)
    skip = any(p.startswith(_TOWER_PREFIXES) for p in path) and not quantize_towers
    skip = skip or (path and path[-1] in _SKIP_NAMES)
    if log:
      LAST_REPLACEMENTS.append(('/'.join(path), type(module).__name__ + (' (skipped)' if skip else ''), None))
    if skip:
      return module
    return _Gemma4SimulateQuantizedEinsum(
        name=module.name + 'qat',  # pyrefly: ignore[unsupported-operation]
        method=method,
        block_size=block_size,
        shape=module.shape,
        weight_name=module.weight_name,
        initializer=module.initializer,
        param_dtype=module.dtype,
        w_scale=module.w_scale,
        clipped=isinstance(module, _g4_layers.ClippedEinsum),
        wrapped=module,
    )
  match module:
    case nn.Dense():
      return peft.SimulateQuantizedDense(wrapped=module, method=method)
    case nn.Einsum():
      return peft.SimulateQuantizedEinsum(wrapped=module, method=method)
    case _layers.Einsum():
      # This hack is required because the FeedForward layer call two different
      # Einsum with using `nn.share_scope`, so the two wrappers need a different
      # name.
      # This seems to be a bug in flax interceptor.
      return _SimulateQuantizedEinsum(
          name=module.name + 'qat',  # pyrefly: ignore[unsupported-operation]
          method=method,
          shape=module.shape,
          weight_name=module.weight_name,
          wrapped=module,
      )
    case _:
      return module


def _replace_by_int(module: nn.Module, dtype: jnp.dtype):
  if isinstance(module, (_g4_layers.Einsum, _g4_layers.ClippedEinsum)):
    # gemma4-qat: Gemma 4 layers (same param-name hack as below).
    return _Gemma4IntEinsum(
        name=f'_IntEinsum_{module.weight_name}' if module.weight_name != 'w' else None,
        shape=module.shape,
        dtype=dtype,
        w_scale=module.w_scale,
        clipped=isinstance(module, _g4_layers.ClippedEinsum),
    )
  match module:
    case nn.Dense():
      return peft.IntDense(wrapped=module, dtype=dtype)
    case nn.Einsum():
      return peft.IntEinsum(wrapped=module, dtype=dtype)
    case _layers.Einsum():
      # This hack is required because the FeedForward layer call two different
      # Einsum with using `nn.share_scope`, so the two wrappers need a different
      # name.
      # This seems to be a bug in flax interceptor.
      if module.weight_name != 'w':
        name = f'_IntEinsum_{module.weight_name}'
      else:
        name = None
      return _IntEinsum(name=name, shape=module.shape, dtype=dtype)
    case _:
      return module


class _SimulateQuantizedEinsum(nn.Module):
  """LoRA wrapper around a Gemma Einsum."""

  _: dataclasses.KW_ONLY
  shape: tuple[int, ...]
  weight_name: str
  method: peft.QuantizationMethod
  kernel_init: nn.initializers.Initializer = nn.initializers.normal()
  wrapped: _layers.Einsum

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': {'wrapped': params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  def process_einsum_str(self, einsum_str: str) -> str:
    """Processes the einsum string."""

    einsum_str = einsum_str.replace(' ', '')
    if '->' not in einsum_str:
      raise ValueError(
          '`einsum_str` equation must be explicit and include "->".'
      )
    if einsum_str.count(',') != 1:
      raise ValueError(
          '`einsum_str` equation must have exactly two operands and '
          'therefore, exactly one comma character, instead of '
          f'{einsum_str.count(",")}'
      )
    return einsum_str

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    eqn = self.process_einsum_str(eqn)
    kernel = self.param(self.weight_name, self.kernel_init, self.shape)
    x, kernel, _ = flax_dtypes.promote_dtype(x, kernel, None, dtype=x.dtype)

    kernel = peft.simulate_quantize(kernel, self.method)

    y = jnp.einsum(eqn, x, kernel)
    return y


class _IntEinsum(nn.Module):
  """LoRA wrapper around a Gemma Einsum."""

  _: dataclasses.KW_ONLY
  shape: tuple[int, ...]
  dtype: jnp.dtype

  def process_einsum_str(self, einsum_str: str) -> str:
    """Processes the einsum string."""

    einsum_str = einsum_str.replace(' ', '')
    if '->' not in einsum_str:
      raise ValueError(
          '`einsum_str` equation must be explicit and include "->".'
      )
    if einsum_str.count(',') != 1:
      raise ValueError(
          '`einsum_str` equation must have exactly two operands and '
          'therefore, exactly one comma character, instead of '
          f'{einsum_str.count(",")}'
      )
    return einsum_str

  def deduce_scale_shape(self, eqn: str) -> tuple[int, ...]:
    """Deduces the scale shape from the kernel shape."""
    axis_to_reduce = _quantization.get_axis_to_reduce_from_einsum_str(eqn)
    new_shape = list(self.shape)
    if axis_to_reduce is None:
      return tuple([1] * (len(self.shape) - 1)) + (self.shape[-1],)
    for axis in axis_to_reduce:
      new_shape[axis] = 1
    return tuple(new_shape)

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    eqn = self.process_einsum_str(eqn)
    kernel = self.param(
        'kernel',
        nn.initializers.ones_init(),
        self.shape,
        self.dtype,
    ).astype(x.dtype)
    scale = self.param(
        'scale',
        nn.initializers.ones_init(),
        self.deduce_scale_shape(eqn),
        x.dtype,
    )
    x, kernel, _ = flax_dtypes.promote_dtype(x, kernel, None, dtype=x.dtype)
    kernel = kernel / scale
    y = jnp.einsum(eqn, x, kernel)
    return y


def _clip_params(module: nn.Module):
  inf = float('inf')
  mk = lambda v: (lambda key, shape, dtype=None: jnp.array(v))  # pylint: disable=g-long-lambda
  return (
      module.param('clip_input_min', mk(-inf), ()),
      module.param('clip_input_max', mk(inf), ()),
      module.param('clip_output_min', mk(-inf), ()),
      module.param('clip_output_max', mk(inf), ()),
  )


class _Gemma4SimulateQuantizedEinsum(nn.Module):
  """QAT version of `gemma4._layers.Einsum` / `ClippedEinsum` (gemma4-qat).

  * Same parameters / names as the wrapped module (shared scope), so
    checkpoints load unchanged (no `convert_to_qat_checkpoint` renaming).
  * Q4 blocks along the einsum *contraction* axes = HF input axis
    (trace 04 §5): q/k/kv `D`, attn_vec `N,H`, gating `F`, linear `H`,
    PLE gate `D`, PLE proj `P`, per_layer_model_projection `d`.
  * `w_scale` is applied after quantization (scalar, commutes with minmax).
  * STE: forward uses Q(w), gradient w.r.t. w is identity.
  """

  _: dataclasses.KW_ONLY
  shape: tuple[int, ...]
  weight_name: str
  method: peft.QuantizationMethod
  block_size: int = 32
  initializer: nn.initializers.Initializer = nn.initializers.normal()
  param_dtype: Any = None
  w_scale: float | None = None
  clipped: bool = False
  wrapped: nn.Module

  def __post_init__(self):
    super().__post_init__()
    if self.scope is not None:
      nn.share_scope(self, self.wrapped)

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    w = self.param(self.weight_name, self.initializer, self.shape, self.param_dtype)
    if isinstance(w, dict) and 'w' in w:  # same workaround as gemma4._layers
      w = w['w']
    formula = _block_quantization.method_to_formula(self.method)
    if formula is not None:
      axes = _block_quantization.contraction_axes(eqn, w.ndim)
      w = _block_quantization.simulate_block_quantize(
          w, axes, block_size=self.block_size, formula=formula
      )
    elif peft.QuantizationMethod(self.method) != peft.QuantizationMethod.NONE:
      w = peft.simulate_quantize(w, self.method)
    if self.w_scale is not None:
      w = w * self.w_scale
    if self.clipped:
      imin, imax, omin, omax = _clip_params(self)
      x = jnp.clip(x, imin, imax)
      y = jnp.einsum(eqn, x, w)
      return jnp.clip(y, omin, omax)
    return jnp.einsum(eqn, x, w)


class _Gemma4IntEinsum(nn.Module):
  """Int inference version of Gemma 4 Einsum (per-output-channel scale over
  the contraction axes). Requires an int checkpoint (`kernel`, `scale`)."""

  _: dataclasses.KW_ONLY
  shape: tuple[int, ...]
  dtype: jnp.dtype
  w_scale: float | None = None
  clipped: bool = False

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    axes = _block_quantization.contraction_axes(eqn, len(self.shape))
    scale_shape = tuple(1 if i in axes else d for i, d in enumerate(self.shape))
    kernel = self.param('kernel', nn.initializers.ones_init(), self.shape, self.dtype).astype(x.dtype)
    scale = self.param('scale', nn.initializers.ones_init(), scale_shape, x.dtype)
    kernel = kernel / scale
    if self.w_scale is not None:
      kernel = kernel * self.w_scale
    if self.clipped:
      imin, imax, omin, omax = _clip_params(self)
      x = jnp.clip(x, imin, imax)
      return jnp.clip(jnp.einsum(eqn, x, kernel), omin, omax)
    return jnp.einsum(eqn, x, kernel)
