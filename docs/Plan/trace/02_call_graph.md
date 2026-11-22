# 02 呼叫鏈（Call Graph）

> 路徑相對 `src/gemma/gemma/`，commit `9e1f741`。✅ 讀碼確認；⚠️ 推論、未實際執行。

## 1. 建構（construction）
```
gm.nn.Gemma4_E2B(tokens='batch.input', text_only=True)            gm/nn/__init__.py:43
 └ _Gemma4Base.__post_init__   text_only → config.replace(vision_encoder=None, audio_encoder=None)   gm/nn/gemma4/_gemma4.py:50-60
 └ Transformer.setup()                                             gm/nn/gemma4/_transformer.py:137-203
    ├ Embedder(vocab=262144, embed_dim=1536, per_layer_input_dim=256, num_layers=35, vision/audio proj dim)  _modules.py:60-110
    ├ kv_cache_sharing_patterns = create_kv_cache_sharing_patterns(...)   _config.py:55-84   → [0..14, 13,13,13,13,14, ...]
    ├ 35 × Block(name=f'layer_{i}', hidden_dim=_get_hidden_dim(i), rope 依型態)   _transformer.py:157-195
    │   └ Block.setup: pre_attention_norm, skip_scale, Attention(...), post_attention_norm,
    │                  _setup_dense (pre_ffw_norm, FeedForward, post_ffw_norm), PLE gate/proj/norm   _modules.py:496-560
    ├ final_norm = RMSNorm()                                       _transformer.py:196
    └ audio_encoder（若未 text_only）                                _transformer.py:200-203
```
✅ Flax `setup` 是延遲執行，實際建立在第一次 `init/apply` 時。

## 2. Forward（`Transformer.__call__`, `_transformer.py:222-342`）
```
__call__ (nn.jit, flatten_unflatten_batch_dim)
 ├ initialize_param_with_dtype(bf16, exclude=[vision/audio/mm/lora])     :289-302
 ├ _encode_and_get_inputs                                                :417-521
 │   ├ (純文字) _types.Input(...) → embedder.encode(tokens_with_mm)       :489-496
 │   ├ positions / attention_mask 預設由 Input 產生                        :502-506
 │   └ embedder.encode_per_layer_input(x, tokens)                         :508-511 → _modules.py:169-198
 ├ _apply_attention                                                       :344-390
 │   for i, block: kv_shared_cache = new_cache[f'layer_{pattern[i]}'] (若共享)   :360-366
 │                 block(x, positions, cache, mask, per_layer_inputs[...,i,:], kv_shared_cache)  :376-386
 │   final_norm                                                           :389
 ├ return_last_only → 取最後有效 token                                      :318-321
 ├ embedder.decode(x) = x @ Eᵀ                                            :332, _modules.py:137
 └ softcap 30                                                             :334-336
```

## 3. Attention（`_modules.py:242-397`）
`q_einsum('BTD,NDH->BTNH')` → `query_norm` → `apply_rope`（`gm/math/_positional_embeddings.py:23-`）→ `kv_einsum('BSD,CKDH->CBSKH')` → `key_norm`/`value_norm` → （共享層取用 `kv_shared_cache['k'/'v']`，否則 rope(k) 並寫入 cache）→ logits（無縮放）→ sliding mask（local）→ `softmax` → `attention_weights`(Identity, 可攔截) → `attn_vec_einsum('BTNH,NHD->BTD')` → 回傳 `new_cache={'k','v',(end_index, positions)}`。✅

## 4. MLP（`_modules.py:422-465`）
`FeedForward.__call__`：`_layers.Einsum(shape=(2,hidden,1536), weight_name='gating_einsum')` + `nn.share_scope` → `gelu(gate0)*gate1` → `_layers.Einsum(shape=(hidden,1536), weight_name='linear')`。✅
注意：這裡的 `_layers` 是 **`gm/nn/gemma4/_layers.py`**（`_modules.py` 內 import），不是 `gm/nn/_layers.py`。✅

## 5. PLE（`_modules.py:169-198`, `:640-659`）
Embedder：`per_layer_model_projection('...td,dnp->...tnp')`（含 `w_scale=1536^-0.5`）→ `per_layer_projection_norm` → `+ PLE[t]*16` → `*rsqrt(2)`。
Block 第 3 步：`per_layer_input_gate('...D,DP->...P')` → `gelu` × `per_layer_input` → `per_layer_projection('...P,PD->...D')` → `post_per_layer_input_norm` → 殘差相加 → `*skip_scale`。✅

## 6. Sampler
```
gm.text.ChatSampler(model, params, ...)            gm/text/_chat_sampler.py:42-
 └ _is_gemma4 → _inner_sampler = Gemma4Sampler(...) gm/text/_chat_sampler.py:160-175
gm.text.Gemma4Sampler.sample(prompt, images, audio, max_new_tokens, sampling, rng, ...)   gm/text/_gemma4_sampler.py:44-120
gm.text.Sampler.sample(prompt, *, images, max_new_tokens, stream, sampling, rng, return_state, last_state, sharding)  gm/text/_sampler.py:86-139, :238-250
 └ SamplerLoop（prefill + 逐 token decode，使用 Transformer.init_cache）    gm/text/_sampler_loop.py:107, gm/nn/gemma4/_transformer.py:402-415
```
⚠️ 取樣內部細節（prefill 分段、pad_length 選擇）未逐行追蹤。

## 7. LoadCheckpoint(quantize=True)
```
gm.ckpts.LoadCheckpoint(path, quantize=True).transform(state)   gm/ckpts/_checkpoint.py:48-69
 └ load_params(path, params=state.params, quantize=True)         :213-337
    ├ ocp.StandardCheckpointer.restore（依 metadata 部分還原）     :245-290
    ├ if quantize: _quantization.convert_to_qat_checkpoint(output) :296-297 → gm/ckpts/_quantization.py:25-64
    │     只做「改名」：{'w': W} → {'_SimulateQuantizedEinsum_0': {'kernel': W}}；
    │     gating_einsum/linear → '_SimulateQuantizedEinsum_gating_einsum' / '_linear'
    │     **不量化任何數值**（:292-293 TODO 自述）
    └ as_nested + 多模態參數轉 f32                                  :300-337
```
✅ `quantize=True` 只改參數樹結構；⚠️ 改名後的 key（`_SimulateQuantizedEinsum_*`）是否與 `gm/nn/_quantization.py:141-147` 產生的模組名（`module.name + 'qat'`，且 `share_scope`）對得上，**未實測**；對 Gemma 4 而言更根本的問題見 §8。

## 8. QAT wrapper → peft._quantization
```
gm.nn.QuantizationAwareWrapper(method=Q4_0, model=Gemma4_E2B(...))   gm/nn/_quantization.py:31-78
 └ __call__: with peft.ModuleInterceptor(_replace_by_simulated_quantization):  :56-63, peft/_interceptors.py:57-
      match module:                                                   gm/nn/_quantization.py:128-149
        nn.Dense      → peft.SimulateQuantizedDense                   peft/_quantization.py:95-134
        nn.Einsum     → peft.SimulateQuantizedEinsum                  peft/_quantization.py:137-203
        gm.nn._layers.Einsum → _SimulateQuantizedEinsum               gm/nn/_quantization.py:136-147, :172-214
        其他          → 原樣返回
 _SimulateQuantizedEinsum.__call__: kernel=param → peft.simulate_quantize(kernel, method) → einsum   :205-214
   └ simulate_quantize → _simulate_uniform_quantization(STE) → _q4_0 → _pack/_unpack     peft/_quantization.py:37-92, :316-451
```
**重點（重新驗證 ✅）**：`case _layers.Einsum()` 的 `_layers` 是 `gemma.gm.nn._layers`（`gm/nn/_quantization.py:24`）。Gemma 4 的所有線性層是 `gm/nn/gemma4/_layers.py:22` 的 `Einsum` 與 `:50` 的 `ClippedEinsum`，二者都直接繼承 `nn.Module`，**不是** `gm.nn._layers.Einsum` 的子類，也不是 `nn.Dense/nn.Einsum`。因此 match 全部落到 `case _`，**對 Gemma 4 文字主幹完全不做假量化**，QAT 實際等於一般 bf16 全參數微調。⚠️ 未執行 trace 驗證（需裝 JAX/Flax 後跑 `jax.make_jaxpr` 或 print interceptor 命中數）。

另：就算 match 上，`_SimulateQuantizedEinsum` 不會套用 gemma4 Einsum 的 `w_scale`（`gm/nn/_quantization.py:205-214` 對比 `gemma4/_layers.py:45-46`），PLE 投影會少乘 `1536^-0.5`；但其後是 RMSNorm，純量縮放幾乎被抵銷（只差 eps）。⚠️ 推論。

## 9. IntWrapper
`gm.nn.IntWrapper(model, dtype=int4)`（`gm/nn/_quantization.py:81-125`）→ `_replace_by_int`（`:152-169`）同樣只認 `nn.Dense / nn.Einsum / gm.nn._layers.Einsum` → `_IntEinsum`（`:217-268`：`kernel/scale` 參數，`kernel/scale` 反量化）。對 Gemma 4 同樣 **不會替換** ✅（讀碼）。

## 10. LoRA
```
gm.nn.LoRA(rank, model, dtype=bf16, verbose)        gm/nn/_lora.py:46-100
 └ ModuleInterceptor(_replace_by_lora)              :75-85, :110-142
     _SUPPORTED_MODULES 含 gemma4 Einsum / ClippedEinsum / gemma3n Einsum   :32-43
     → _LoRAEinsum(name='_LoRAEinsum_<weight_name>' 或 None)               :136-140, :145-170
         return wrapped(eqn, x) + LoRAEinsumAdapter(einsum_str=eqn, shape)(x)
 LoRAEinsumAdapter.setup: get_lora_einsum_str_and_shapes → a:[in_dims..., r], b:[r, out_dims...]   peft/_lora.py:89-132, peft/_einsum_utils.py:23-75
   a_init = kaiming_uniform, b_init = zeros
```
✅ Gemma 4 有被 LoRA 支援（含視覺 ClippedEinsum）。✅ **無 alpha/r 縮放**：輸出就是 `x·A·B`（`peft/_lora.py:131-132`）。
✅ **無 merge**：`peft.fuse_params` / `unfuse_params` 直接 `raise NotImplementedError()`（`peft/_tree_utils.py:140-145`）；只有 `split_params / merge_params`（把 lora 子樹拆開/放回，不是把 A·B 加回 W）。
⚠️ 權重存取：`_LoRAEinsum` 不 share_scope（`gm/nn/_lora.py:153-157`），參數樹多一層 `_LoRAEinsum_*`/`lora`；載入原始 ckpt 要用 `gm.ckpts.SkipLoRA`（`gm/ckpts/_lora.py:36-53`）。

## 11. Kauldron 訓練迴圈
`examples/lora.py:49-116`：`kd.train.Trainer(train_ds, model=gm.nn.LoRA(rank=4, model=gm.nn.Gemma3_4B(...)), init_transform=gm.ckpts.SkipLoRA(wrapped=gm.ckpts.LoadCheckpoint(path)), train_losses={'xentropy': kd.losses.SoftmaxCrossEntropyWithIntLabels(logits='preds.logits', labels='batch.target', mask='batch.loss_mask')}, optimizer=kd.optim.partial_updates(optax.adafactor(5e-3), mask=kd.optim.select('lora')), checkpointer=..., evals=...)`。
啟動：`python -m kauldron.main --cfg=examples/lora.py --cfg.workdir=...`（`examples/lora.py:31-35`）。
QAT 版本：repo 範例中**沒有** Gemma 4 QAT 範例；`QuantizationAwareWrapper` 與 `LoadCheckpoint(quantize=True)` 只在 `gm/nn/_quantization.py` 與 `gm/ckpts/_checkpoint.py` 中出現（`rg` 結果）✅。⚠️ Kauldron 內部（`kd.train`）未追蹤，屬外部套件。
