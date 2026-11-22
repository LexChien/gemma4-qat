r"""Mixed-precision QAT (Google-mobile-like) on top of the repo's QAT machinery.
The repo (gemma/gm/nn/_quantization.py) applies ONE method to all Dense/Einsum layers and never
touches embeddings; QuantizationMethod has no 2-bit entry. This file adds a per-module policy using
the repo's public wrappers + its private uniform simulator (gemma/peft/_quantization.py
_simulate_uniform_quantization, asserts bitwidth>=2) — private API, may break between commits.
Embedding / PLE QAT is NOT covered (no hook in repo); quantize them post-hoc (4-bit g256 is what
Google's mobile-ct does for PLE). Untested sketch.
"""
import dataclasses, functools
from flax import linen as nn
from gemma import gm, peft
from gemma.peft import _quantization as _q, _quantization_utils as _qu

def _bits_for(path: str) -> int:
  # Flax scope path, e.g. ('layer_17','mlp','gating_einsum'); names are repo-internal (check with jax.tree_util)
  s = "/".join(path)
  if "mlp" in s:
    import re
    m = re.search(r"layer_(\d+)", s)
    if m and int(m.group(1)) >= 15: return 2
  return 4

def _sim(x, bits, axis_to_reduce=None):
  return _q._simulate_uniform_quantization(
      x, bitwidth=bits, granularity=_qu.QuantizationGranularity.PER_CHANNEL, axis_to_reduce=axis_to_reduce)

class MixedQAT(nn.Module):
  """Like gm.nn.QuantizationAwareWrapper but uses Q4_0 for 4-bit and per-channel int2 for 2-bit MLPs.
  2-bit path: monkeypatches simulate_quantize per call; prototype only."""
  _: dataclasses.KW_ONLY
  model: nn.Module
  def __post_init__(self):
    super().__post_init__()
    if self.scope is not None: nn.share_scope(self, self.model)
  @nn.compact
  def __call__(self, *args, **kwargs):
    def replace(module):
      bits = _bits_for(module.scope.path if module.scope else ())
      method = peft.QuantizationMethod.Q4_0 if bits == 4 else peft.QuantizationMethod.INT4
      new = gm.nn._quantization._replace_by_simulated_quantization(module, method=method)
      return new
    with peft.ModuleInterceptor(replace):
      return self.model(*args, **kwargs)
# TODO(user): for true 2-bit, add QuantizationMethod.INT2 to gemma/peft/_quantization_utils.py and a
# matching `case` in simulate_quantize() calling _simulate_uniform_quantization(bitwidth=2, PER_CHANNEL).
