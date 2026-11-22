"""Tiny Gemma 4 (E2B-shaped, scaled down) for fast M4 tests."""
import dataclasses
from gemma import gm
from gemma.gm.nn.gemma4 import _config, _modules

def tiny_config(num_layers=5, vocab=512, d=128, ple=32):
  base = gm.nn.Gemma4_E2B.config
  pat = (_modules.AttentionType.LOCAL_SLIDING,) * 4 + (_modules.AttentionType.GLOBAL,)
  return dataclasses.replace(
      base, num_embed=vocab, embed_dim=d, hidden_dim=4 * d, head_dim=64,
      global_key_size=64, num_heads=2, per_layer_input_dim=ple,
      attention_types=_config.make_attention_layers_types(pat, num_layers=num_layers),
      kv_cache_sharing_config=_config.KVCacheSharingConfig(frac_shared_layers=0.4, share_global=True, share_local=True),
      override_kv_shared_ffw_hidden=8 * d, sliding_window_size=16,
      vision_encoder=None, audio_encoder=None)

def tiny_model(**kw):
  return gm.nn.Gemma4_E2B(config=tiny_config(**kw), text_only=True)
