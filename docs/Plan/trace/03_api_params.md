# 03 公開 API 參數表

> 路徑相對 `src/gemma/gemma/`。皆為讀碼 ✅（未執行）。

## 模型
| API | 簽章 / 欄位（預設） | 出處 |
|---|---|---|
| `gm.nn.Gemma4_E2B` | `text_only: bool = True`；`config = _gemma4_config(embed_dim=1536, num_heads=8, num_kv_heads=1, num_global_kv_heads=None, per_layer_input_dim=256, frac_shared_layers=20/35, sliding_window_size=512, k_eq_v_global=False, use_post_attn_norm=True, use_post_ffw_norm=True, override_ffw_hidden_for_kv_cache_sharing=True, vision_encoder=VisionEncoder(use_clipped_linears=True), audio_encoder=ConformerConfig())` | `gm/nn/gemma4/_gemma4.py:44,122-155` |
| `Transformer`（基底，KW_ONLY） | `return_last_only: bool|None=None`, `dtype=jnp.bfloat16`, `tokens: kontext.Key = REQUIRED`, `images: kontext.Key|None=None`, `config: TransformerConfig` | `gm/nn/gemma4/_transformer.py:110-132` |
| `Transformer.__call__` | `(tokens: Int['*B L'], *, images=None, audio=None, audio_lengths=None, positions=None, cache=None, attention_mask=None, return_last_only=None, return_hidden_states=None, audio_soft_token_counts=None) -> Output(logits, cache, hidden_states)` | `_transformer.py:235-254` |
| `Transformer.init_cache` | `(*, batch_size, dtype, cache_length, sharding=None)` | `_transformer.py:402-415` |
| `TransformerConfig` | `num_embed, embed_dim, hidden_dim, num_heads, head_dim, num_kv_heads, final_logit_softcap, use_post_attn_norm, use_post_ffw_norm, attention_types, attn_logits_soft_cap=None, sliding_window_size=None, qk_norm_with_scale=True, num_global_kv_heads=None, global_key_size=None, k_eq_v_global=False, global_rope_proportion=None, local_rope_proportion=None, local_base_frequency=10000, global_base_frequency=10000, local/global_scale_factor=1.0, per_layer_input_dim=0, kv_cache_sharing_config=None, override_kv_shared_ffw_hidden=None, vision_encoder=None, audio_encoder=None, enable_moe=False, num_experts=0, expert_dim=0, top_k_experts=8, moe_dense_hidden_dim=0, use_bidirectional_attention=None` | `gm/nn/gemma4/_config.py:87-131` |
| `KVCacheSharingConfig` | `frac_shared_layers=0.0, share_global=False, share_local=False` | `_config.py:32-38` |
| `Embedder` | `vocab_size, embed_dim, num_layers=0, per_layer_input_dim=0, vision_proj_dim=None, audio_proj_dim=None`；方法 `encode, decode, encode_logits, encode_vision, encode_audio, encode_per_layer_input(x, t, ignore_ple_tokens=False)` | `gm/nn/gemma4/_modules.py:60-198` |
| `Attention` | `num_heads, num_kv_heads, features, key_size, attn_type, rope_base_frequency=10000, rope_scale_factor=1.0, rope_proportion=None, attn_logits_soft_cap=None, sliding_window_size=None, qk_norm_with_scale=True, k_eq_v=False`；`__call__(x, segment_pos, cache, attn_mask, kv_shared_cache=None, skip_sliding_mask=False)` | `_modules.py:201-250` |
| `FeedForward` | `features, hidden_dim` | `_modules.py:422-427` |
| `Block` | 見欄位 `num_heads … top_k_experts=0`；`__call__(x, segment_pos, cache, attn_mask, per_layer_input=None, kv_shared_cache=None, skip_sliding_mask=False)` | `_modules.py:468-602` |
| gemma4 `Einsum` | `shape, weight_name='w', initializer=normal(), dtype=None, w_scale=None`；`__call__(eqn, x)` | `gm/nn/gemma4/_layers.py:22-47` |
| gemma4 `ClippedEinsum` | 同上 + 參數 `clip_input_min/max, clip_output_min/max`（預設 ±inf） | `_layers.py:50-90` |
| gemma4 `RMSNorm` | `with_scale=True`；eps 1e-6；乘 `scale` | `_layers.py:93-117` |
| gm.nn `Einsum`（Gemma 2/3 用） | `shape, weight_name='w', initializer, dtype=None`（**無 w_scale**） | `gm/nn/_layers.py:22-31` |
| `apply_rope` | `(inputs, positions, *, base_frequency, scale_factor=1.0, rope_proportion=1.0)` | `gm/math/_positional_embeddings.py:23-30` |

## 包裝器
| API | 欄位（預設） | 出處 |
|---|---|---|
| `gm.nn.QuantizationAwareWrapper` | KW_ONLY `method: peft.QuantizationMethod = NONE`, `model: nn.Module` | `gm/nn/_quantization.py:31-46` |
| `gm.nn.IntWrapper` | KW_ONLY `model`, `dtype = jnp.int4` | `gm/nn/_quantization.py:81-95` |
| `gm.nn.LoRA` | KW_ONLY `rank: int`, `model`, `dtype = jnp.bfloat16`, `verbose = False` | `gm/nn/_lora.py:46-65` |

## peft
| API | 簽章（預設） | 出處 |
|---|---|---|
| `QuantizationMethod` | `NONE, INT4, INT8, Q4_0, Q4_0_TRANSPOSE, SFP8` | `peft/_quantization_utils.py:29-45` |
| `QuantizationGranularity` | `PER_TENSOR, PER_CHANNEL, PER_BLOCK` | `:48-62` |
| `peft.simulate_quantize` | `(x, method, axis_to_reduce=None)` | `peft/_quantization.py:37-92` |
| `peft.quantize` | `(params, *, method, checkpoint_kernel_key='w', in_place_keys=False)`；**只支援 NONE/INT4/INT8**，其餘 `ValueError` | `peft/_quantization_utils.py:65-157` |
| `uniform_quantize` | `(x, *, bitwidth, granularity, dtype=int4, axis_to_reduce=None) -> (q, scale)`；DQ = q/scale | `:160-201` |
| `SimulateQuantizedDense` | `wrapped: nn.Dense, method=NONE` | `peft/_quantization.py:95-134` |
| `SimulateQuantizedEinsum` | `wrapped: nn.Einsum, method=NONE`；`__call__(inputs, einsum_str=None)` | `:137-203` |
| `IntDense` / `IntEinsum` | `wrapped, dtype=int4` | `:206-313` |
| `get_axis_to_reduce_from_einsum_str` | 只認 5 個 Gemma 2/3 字串 | `:505-521` |
| `LoRADense/Adapter` | `rank, wrapped/features, dtype=jnp.float_, a_init=kaiming_uniform(), b_init=zeros_init()` | `peft/_lora.py:28-86` |
| `LoRAEinsumAdapter` | `rank, einsum_str, shape, dtype=float_, a_init, b_init` | `:89-132` |
| `LoRAEinsum` / `LoRADenseGeneral(+Adapter)` | 同上 | `:138-280` |
| `split_params / merge_params` | `(params) -> SplittedParams` / `(original, lora)` | `peft/_tree_utils.py:29,111` |
| `fuse_params / unfuse_params` | `raise NotImplementedError()` | `peft/_tree_utils.py:140-145` |
| `ModuleInterceptor` | `(replace_module_fn)` context manager（`nn.intercept_methods`） | `peft/_interceptors.py:29-135` |

## Checkpoint / Sampler
| API | 簽章（預設） | 出處 |
|---|---|---|
| `gm.ckpts.LoadCheckpoint` | `path, quantize=False` | `gm/ckpts/_checkpoint.py:48-69` |
| `gm.ckpts.load_params` | `(path, *, params=None, donate=True, text_only=False, sharding=None, quantize=False, restore_concurrent_gb=None)` | `:213-222` |
| `gm.ckpts.save_params` | 見 `:191-` | `:191` |
| `gm.ckpts.SkipLoRA` | `wrapped: kd.ckpts.InitTransform` | `gm/ckpts/_lora.py:36-53` |
| `convert_to_qat_checkpoint` | `(params, *, checkpoint_kernel_key='w')` | `gm/ckpts/_quantization.py:25-64` |
| `CheckpointPath.GEMMA4_E2B_IT/PT` | `gs://gemma-data/checkpoints/gemma4-e2b-{it,pt}` | `gm/ckpts/_paths.py:68,73` |
| `gm.text.Sampler` | `model, params, tokenizer=None, sampling=Greedy(), forbidden_tokens=None, stop_tokens=None, cache_length=4096, max_out_length=2048, pad_length=(256,512,1024)` | `gm/text/_sampler.py:86-139` |
| `Sampler.sample` | `(prompt, *, images=None, max_new_tokens=None, stream=False, sampling=None, rng=None, return_state=False, last_state=None, sharding=None)` | `:238-250` |
| `gm.text.Gemma4Sampler` | 同 Sampler + `patch_size=16, max_soft_tokens=1120, pooling_kernel_size=3, audio_sample_rate=16000, audio_seq_length=750` | `gm/text/_gemma4_sampler.py:44-83` |
| `Gemma4Sampler.sample` | `(prompt, images=None, audio=None, audio_lengths=None, max_new_tokens=None, sampling=None, rng=None, return_state=False, ...)` | `:98-` |
| `gm.text.ChatSampler` | `model, params, multi_turn, print_stream, tokenizer, sampling, …, pad_length, patch_size, max_soft_tokens, pooling_kernel_size, audio_sample_rate, audio_seq_length, last_state, turns`；Gemma4 自動轉用 Gemma4Sampler | `gm/text/_chat_sampler.py:42-175` |
