"""transformers 5.20.1 + compressed-tensors 0.19.0 workaround: a quantized nn.Embedding (embed_tokens / embed_tokens_per_layer)
has weight_packed/weight_scale instead of `weight`, and PreTrainedModel._init_weights calls init.normal_(module.weight)
-> AttributeError. Skip init for such embeddings (their packed params come from the checkpoint). Import before from_pretrained."""
import torch.nn as nn
from transformers import modeling_utils as _mu
_orig = _mu.PreTrainedModel._init_weights
def _init_weights(self, module):
    if isinstance(module, nn.Embedding) and getattr(module, "weight", None) is None:
        return
    return _orig(self, module)
_mu.PreTrainedModel._init_weights = _init_weights
