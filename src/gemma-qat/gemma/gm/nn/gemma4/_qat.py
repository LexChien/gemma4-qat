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

"""Trace-time QAT configuration for Gemma 4 embedding tables (gemma4-qat, M4).

`input_embedding` and `per_layer_embeddings` are created with `self.param`
inside `Embedder.setup` (gemma4/_modules.py), so they are not modules and the
`ModuleInterceptor` used by `QuantizationAwareWrapper` cannot replace them
(trace 05 §5). Instead the wrapper sets this context while tracing the model
and `Embedder` fake-quantizes the gathered rows (blocks never cross rows, so
Q(table)[ids] == Q(table[ids]); the full table is only quantized for the
tied lm_head / soft-token paths).
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses

import jax

from gemma.peft import _block_quantization


@dataclasses.dataclass(frozen=True)
class EmbedQATConfig:
  formula: str = 'minmax'          # 'minmax' (official grid, idempotent) | 'absmax' | 'q4_0'
  embed_block_size: int = 32       # input_embedding (and tied lm_head)
  ple_block_size: int = 32         # per_layer_embeddings (official QAT: 32; mobile-ct: 256)
  quantize_embed: bool = True
  quantize_ple: bool = True
  quantize_lm_head: bool = True    # decode() uses the tied, quantized table
  scale_dtype: object = None


_CTX: contextvars.ContextVar[EmbedQATConfig | None] = contextvars.ContextVar(
    'gemma4_embed_qat', default=None
)


def current() -> EmbedQATConfig | None:
  return _CTX.get()


@contextlib.contextmanager
def embed_qat(cfg: EmbedQATConfig | None):
  tok = _CTX.set(cfg)
  try:
    yield
  finally:
    _CTX.reset(tok)


def q_rows(x: jax.Array, block_size: int, cfg: EmbedQATConfig, n_axes: int = 1) -> jax.Array:
  """STE fake-quant in row blocks along the last `n_axes` axes."""
  axes = tuple(range(x.ndim - n_axes, x.ndim))
  return _block_quantization.simulate_block_quantize(
      x, axes, block_size=block_size, formula=cfg.formula, scale_dtype=cfg.scale_dtype
  )


# Contraction (= HF input) axes per Gemma 4 param name, see
# docs/Plan/trace/04_weights_map.md §5. Used to snap a trained param tree onto
# the Q4 grid for export (same blocks as the QAT wrappers).
CONTRACTION_AXES_BY_NAME = {
    'q_einsum': (1,),               # BTD,NDH->BTNH         [N,D,H]
    'k_einsum': (1,),               # BSD,KDH->BSKH         [K,D,H]
    'kv_einsum': (2,),              # BSD,CKDH->CBSKH       [C,K,D,H]
    'attn_vec_einsum': (0, 1),      # BTNH,NHD->BTD         [N,H,D]
    'gating_einsum': (2,),          # ...F,NHF->...NH       [2,F,D]
    'linear': (0,),                 # ...H,HF->...F         [F,D]
    'per_layer_input_gate': (0,),   # ...D,DP->...P
    'per_layer_projection': (0,),   # ...P,PD->...D
    'per_layer_model_projection': (0,),  # ...td,dnp->...tnp
}
_SKIP_PREFIXES = ('vision_encoder', 'audio_encoder')
_SKIP_SUBSTR = ('mm_input_projection', 'audio_input_projection')


def snap_params(params, *, block_size: int = 32, embed_block_size: int = 32,
                ple_block_size: int = 32, formula: str = 'minmax',
                quantize_embed: bool = True, quantize_ple: bool = True,
                scale_dtype=None):
  """Returns `params` with every QAT-quantized weight replaced by Q(w).

  Equivalent to what `gm.nn.QuantizationAwareWrapper(method=Q4_0_MINMAX)`
  computes in the forward pass, so `model.apply(snap_params(p))` ==
  `qat_model.apply(p)`. Multimodal towers / projections are left untouched.
  """
  bq = _block_quantization

  def fn(path, x):
    keys = [getattr(k, 'key', getattr(k, 'name', str(k))) for k in path]
    if keys and keys[0] in _SKIP_PREFIXES:
      return x
    if any(s in keys for s in _SKIP_SUBSTR):
      return x
    name = keys[-1]
    if name == 'w' and len(keys) >= 2:
      name = keys[-2]
    if name == 'input_embedding':
      if not quantize_embed:
        return x
      return bq.block_quantize(x, (x.ndim - 1,), block_size=embed_block_size, formula=formula, scale_dtype=scale_dtype)
    if name == 'per_layer_embeddings':
      if not quantize_ple:
        return x
      return bq.block_quantize(x, tuple(range(1, x.ndim)), block_size=ple_block_size, formula=formula, scale_dtype=scale_dtype)
    if name in CONTRACTION_AXES_BY_NAME:
      return bq.block_quantize(x, CONTRACTION_AXES_BY_NAME[name], block_size=block_size, formula=formula, scale_dtype=scale_dtype)
    return x

  return jax.tree_util.tree_map_with_path(fn, params)
