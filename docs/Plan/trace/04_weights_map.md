# 04 權重對照表（HF safetensors ↔ orbax JAX）

> 資料來源：`raw/*.safetensors_header.json`、`raw/*.config.json`、`raw/orbax_*.tree.json`、`raw/orbax_*._METADATA.json`、
> `raw/q4_grid_check*.json`、`raw/orbax_vs_hf_compare.{txt,json}`（2026-10-17 以 `tools/compare_orbax_hf.py` 實測）。
> ✅ = 數值實測（bit-exact 或統計）；☑️ = 標頭/形狀確認；⚠️ = 推論。

## 1. HF checkpoint 清單（皆單檔 `model.safetensors`，全部 BF16）

| ckpt | 張量數 | 參數 | 大小 |
|---|---|---|---|
| `gemma-4-E2B`（pt） | 2011 | 5,123,178,979 | 10.246 GB |
| `gemma-4-E2B-it` | 2011 | 5,123,178,979 | 10.246 GB |
| `gemma-4-E2B-it-qat-q4_0-unquantized` | 1951 | 5,104,298,467 | 10.209 GB |

☑️ 三份 config 只有 `transformers_version` 不同（5.5.0.dev0 vs 5.6.2）。

### 分組（it；qat 差異見括號）
| 群組 | 張量 | 參數 | bf16 GB |
|---|---|---|---|
| `embed_tokens_per_layer` [262144, 8960] | 1 | 2,348,810,240 | 4.698 |
| 文字 MLP gate/up/down | 105 | 1,557,135,360 | 3.114 |
| `embed_tokens` [262144,1536]（tie lm_head） | 1 | 402,653,184 | 0.805 |
| 音訊塔 + `embed_audio` | 752 | 307,184,384 | 0.614 |
| 文字 attention q/k/v/o | 140（qat 100） | 297,271,296（qat 278,396,928） | 0.595（0.557） |
| 視覺塔 + `embed_vision` | 659 | 168,544,704 | 0.337 |
| 每層 `per_layer_input_gate`/`per_layer_projection` | 70 | 27,525,120 | 0.055 |
| `per_layer_model_projection` + `per_layer_projection_norm` | 2 | 13,762,816 | 0.028 |
| norms / `layer_scalar` | 281（qat 261） | 291,875 | 0.001 |

**QAT 版刪掉的 60 個張量** ☑️：第 15-34 層的 `self_attn.k_proj.weight`、`v_proj.weight`、`k_norm.weight`（各 20）。這些層用的是共享 KV（01 §1），原本就是死權重（JAX 仍會算再丟棄，`_modules.py:275-287`）。

### 文字主幹每層張量（HF 名稱，前綴 `model.language_model.layers.{i}.`）
`input_layernorm[1536]`, `self_attn.q_proj[2048|4096,1536]`, `self_attn.k_proj/v_proj[256|512,1536]`, `self_attn.q_norm/k_norm[256|512]`, `self_attn.o_proj[1536,2048|4096]`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `mlp.gate_proj/up_proj[6144|12288,1536]`, `mlp.down_proj[1536,6144|12288]`, `post_feedforward_layernorm`, `per_layer_input_gate[256,1536]`, `per_layer_projection[1536,256]`, `post_per_layer_input_norm[1536]`, `layer_scalar[1]`。（`|` 右側 = global 層 head_dim 512 / KV 共享層 MLP 12288）

## 2. orbax 樹（`weights/orbax/checkpoints/gemma4-e2b-{it,pt}`）
- 1318 個葉節點：838 個 ndarray + 480 個 scalar（clip 值）。☑️ it 與 pt 形狀/dtype 完全一樣。
- dtype：**1167 個 f32，151 個 bfloat16**；bf16 全部在 `audio_encoder`（norm scale、depthwise conv、pos_proj、feature、output_projection）。文字主幹、embedder、視覺全為 f32。☑️（修正先前「orbax 全 f32」的說法）
- 分佈：`audio_encoder` 751、`layer_0..34` 525（每層 15）、`vision_encoder` 33（16 層堆疊成 `stacked_layers.block.*`，首軸 16）、`embedder` 8、`final_norm` 1。
- embedder 8 個：`input_embedding[262144,1536]`、`per_layer_embeddings[262144,35,256]`、`per_layer_model_projection.w[1536,35,256]`、`per_layer_projection_norm.scale[256]`、`mm_input_projection.w[768,1536]`、`audio_input_projection.w[1536,1536]`、`mm_input_embedding_extra[128,768]`、`audio_input_embedding_extra[128,1536]`（後兩者 **HF 無對應**）。
- 數量對應：HF 2011 = 文字 600（35×17+5）+ 視覺 658（16×41+2）+ 音訊 751 + embed_vision 1 + embed_audio 1；orbax 1318 = 文字 530（35×15+5）+ 視覺 33 + 音訊 751 + mm/audio 投影 2 + extra 2。差異來自 kv→k/v、gating→gate/up 拆分（每層 +2），以及視覺 16 層堆疊。☑️

## 3. JAX ↔ HF 名稱 / 形狀 / 轉換（✅ = it 與 pt 實測 bit-exact，`bf16(orbax_f32)` == HF）

文字層（JAX 前綴 `layer_{i}.`，HF 前綴 `model.language_model.layers.{i}.`）：
| JAX | 形狀 | HF | 形狀 | 轉換 HF = f(JAX) | 驗證 |
|---|---|---|---|---|---|
| `attn.q_einsum.w` | [8,1536,H] | `self_attn.q_proj.weight` | [8H,1536] | `w.transpose(0,2,1).reshape(8H,1536)` | ✅ L0,4,14,15,19,34 |
| `attn.kv_einsum.w[0]` | [1,1536,H] | `self_attn.k_proj.weight` | [H,1536] | `w[0,0].T` | ✅ |
| `attn.kv_einsum.w[1]` | [1,1536,H] | `self_attn.v_proj.weight` | [H,1536] | `w[1,0].T` | ✅ |
| `attn.attn_vec_einsum.w` | [8,H,1536] | `self_attn.o_proj.weight` | [1536,8H] | `w.reshape(8H,1536).T` | ✅ |
| `attn.query_norm.scale` | [H] | `self_attn.q_norm.weight` | [H] | 相同 | ✅ |
| `attn.key_norm.scale` | [H] | `self_attn.k_norm.weight` | [H] | 相同 | ✅ |
| `mlp.gating_einsum.w[0]` | [F,1536] | `mlp.gate_proj.weight` | [F,1536] | **不轉置** | ✅ |
| `mlp.gating_einsum.w[1]` | [F,1536] | `mlp.up_proj.weight` | [F,1536] | **不轉置** | ✅ |
| `mlp.linear.w` | [F,1536] | `mlp.down_proj.weight` | [1536,F] | `.T` | ✅ |
| `pre_attention_norm.scale` | [1536] | `input_layernorm.weight` | | 相同 | ✅ |
| `post_attention_norm.scale` | | `post_attention_layernorm.weight` | | 相同 | ✅ |
| `pre_ffw_norm.scale` | | `pre_feedforward_layernorm.weight` | | 相同 | ✅ |
| `post_ffw_norm.scale` | | `post_feedforward_layernorm.weight` | | 相同 | ✅ |
| `post_per_layer_input_norm.scale` | | `post_per_layer_input_norm.weight` | | 相同 | ✅ |
| `per_layer_input_gate.w` | [1536,256] | `per_layer_input_gate.weight` | [256,1536] | `.T` | ✅ |
| `per_layer_projection.w` | [256,1536] | `per_layer_projection.weight` | [1536,256] | `.T` | ✅ |
| `skip_scale` | [1] | `layer_scalar` | [1] | 相同 | ✅ |
| （無） | | | | `value_norm` 無參數 | ☑️ |

全域（HF 前綴 `model.language_model.` 除另註）：
| JAX | HF | 轉換 | 驗證 |
|---|---|---|---|
| `embedder.input_embedding` [262144,1536] | `embed_tokens.weight` | 相同（**未**預乘 √1536） | ✅ 64 隨機列 |
| `embedder.per_layer_embeddings` [262144,35,256] | `embed_tokens_per_layer.weight` [262144,8960] | `reshape(V, 35*256)`（未預乘 16） | ✅ 64 列 |
| `embedder.per_layer_model_projection.w` [1536,35,256] | `per_layer_model_projection.weight` [8960,1536] | `reshape(1536,8960).T`（it/pt **未**預乘 1536^-0.5） | ✅ |
| `embedder.per_layer_projection_norm.scale` [256] | `per_layer_projection_norm.weight` | 相同 | ✅ |
| `final_norm.scale` | `norm.weight` | 相同 | ✅ |
| `embedder.mm_input_projection.w` [768,1536] | `model.embed_vision.embedding_projection.weight` [1536,768] | `.T` | ✅ |
| `embedder.audio_input_projection.w` [1536,1536]（方陣） | `model.embed_audio.embedding_projection.weight` | `.T`（不轉置時 cos≈0.001） | ✅ |

視覺（JAX `vision_encoder.transformer.stacked_layers.block.*[l]`，HF `model.vision_tower.encoder.layers.{l}.*`；768×768 方陣）：
`attn.q_einsum.w[l]`[12,768,64] → `self_attn.q_proj.linear.weight` = `transpose(0,2,1).reshape(768,768)` ✅；`kv_einsum.w[l,0/1]` → `k_proj/v_proj.linear.weight` 同法 ✅(k)；`attn_vec_einsum.w[l]`[12,64,768] → `o_proj.linear.weight` = `reshape(768,768).T` ✅；`pre_attention_norm.scale[l]` → `input_layernorm.weight` 相同 ✅；`mlp.gating_einsum.w[l,0/1]`→`gate/up_proj.linear.weight`、`mlp.linear.w[l]`→`down_proj.linear.weight` ⚠️（依文字層規則推論，未實測）；`*.clip_{input,output}_{min,max}[l]` → `*.{input,output}_{min,max}`（shape [16]→每層 []）⚠️ 數值未比對（腳本形狀比較失敗）；`entry.input_projection.w`→`patch_embedder.input_proj.weight`、`entry.pos_emb[10240,2,768]`→`patch_embedder.position_embedding_table[2,10240,768]` ⚠️ 推論（transpose(1,0,2)）。

音訊（JAX `audio_encoder.conformer/stacked_layers_{l}.*`，HF `model.audio_tower.layers.{l}.*`；1024×1024 方陣）：
`trans_atten.self_atten.{query,key,value}.kernel` → `self_attn.{q,k,v}_proj.linear.weight` = `.T` ✅(q,k)；`lconv.linear_end.kernel` → `lconv1d.linear_end.linear.weight` = `.T` ✅；`trans_atten.post.kernel`[8,128,1024] → `self_attn.post.linear.weight` = `reshape(1024,1024).T` ✅；`relative_position_embedding.pos_proj.kernel`[1024,8,128] → `self_attn.relative_k_proj.weight` = `reshape(1024,1024).T` ✅；`trans_atten.pre_norm.scale` → `norm_pre_attn.weight` 相同 ✅；`per_dim_scale` 相同 ✅；`fflayer_start/end.*` → `feed_forward1/2.*`、`lconv.depthwise_conv1d.kernel`[5,1,1024] → `lconv1d.depthwise_conv1d.weight`[1024,1,5]、`feature.*` → `subsample_conv_projection.*`、`output_projection` → `output_proj` ⚠️ 依命名推論，未實測。

## 4. RMSNorm 慣例與縮放（✅ 實測）
- **HF 存的就是 JAX 的 `scale`（乘 w，不是 1+w）**：所有文字/視覺/音訊 norm 候選 `w` bit-exact，`w-1`/`w+1` 皆不符。q_norm/k_norm 的 scale 本身接近 0 附近的小值（`w-1` 時 cos=-1）。→ 寫 JAX→HF 轉換器時 **不要** 做 ±1。
- 嵌入/PLE：HF 與 orbax 都存未縮放值；`√1536`、`√256`、`rsqrt(2)` 都是 forward 時乘（`_modules.py:124,197-198`）。
- **`per_layer_model_projection`**：
  - HF it/pt = orbax（未縮放）bit-exact。
  - HF **QAT** 對 orbax-it：未縮放候選 norm 比 0.0186；乘 `1536^-0.5` 後 norm 比 0.731，與其他 QAT 張量對 it 的比例（q_proj 0.697、embed_tokens 0.826）同一量級；cos 0.981。→ **QAT 版的投影很可能已預乘 1/√1536**（≈0.0255），而 it/pt 沒有。✅ 統計證據；⚠️ 確切成因未知。
  - 影響：投影輸出接著是 `per_layer_projection_norm`（RMSNorm，對最後一軸 256 正規化，`_modules.py:192-193`），整體純量縮放會被抵銷，只剩 eps=1e-6 的差異 ⚠️（推論；若 HF 執行時也乘 1/√1536，QAT 版的 RMS 會再小 39 倍，需確認 eps 是否開始起作用）。

## 5. QAT checkpoint 與 Q4_0 格點（✅ 實測；每 32 個一組，「≤16 級」= 該組唯一值 ≤16）
抽樣 4096 組 / 張量（嵌入取 256 隨機列）。HF 權重 `[out,in]`，「last」= 沿輸入軸（嵌入表則沿嵌入維）。

| 張量（qat） | last：uniq_max / frac≤16 | first：uniq_max | it 對照 |
|---|---|---|---|
| L0 q_proj / o_proj / gate / down，L20 up，L4 k_proj | 16 / 1.0 | 32 | it 全為 32 / 0.0 |
| L0 per_layer_input_gate、per_layer_projection | 16 / 1.0 | 32 | 32 |
| `per_layer_model_projection` | 16 / 1.0 | 32 | 32 |
| **`embed_tokens`** | **16 / 1.0** | 32 | 32 |
| **`embed_tokens_per_layer`（PLE）** | **16 / 1.0** | 32 | 32 |
| L34 q_proj、L4 o_proj（global） | 16 / 1.0 | 32 | 32 |
| norms（input_layernorm） | 31 | 31 | 未量化 |
| 視覺 q_proj、`embed_vision`、`embed_audio` | 32 / 0.0 | 32 | **未量化**（與 it 相同） |
| 音訊 q_proj | **4 / 1.0** | 30 | it **也是 4**（原始權重就是低位元格點） |

結論：
1. 官方 QAT 對 **文字主幹全部線性層 + embed_tokens + PLE 表 + PLE 投影** 都做了 4-bit、每 32 個一組、**沿 HF 輸入軸（JAX 的 D/收縮軸）** 的量化；視覺與 embed 投影未量化。✅（新增：先前未量化嵌入的推測被推翻 → 嵌入與 PLE 也在 Q4 格點上）
2. 音訊塔的線性層在 it/pt 原檔就已是每 32 組 ≤4 級（≈2-bit），QAT 版未改動。✅ 實測；⚠️ 格式意義未確認。
3. `frac_on_grid≈0.40`（以 `amax/8` 為步長的整數誤差 <0.02 的比例）偏低，原因可能是 scale 先存 fp16、值再存 bf16 所致的舍入，或官方用的不是 `max/-8` 對稱格點。⚠️ 未確認，唯一值 ≤16 是較可靠的證據。
4. QAT 權重與 it 差異：q_proj cos 0.9945、rel_l2 0.31；嵌入 cos 0.990；norm cos ≥0.9999。

### 分組方向與 repo `_q4_0` 的對照（讀碼 + 上表）
repo `_pack` 沿 JAX kernel **最後一軸** 分 32（`peft/_quantization.py:336-347`；`Q4_0` 時 `transpose=False`，`:76-81`, `:366-394`）：
| 層 | JAX shape | 最後一軸 | 官方需要（收縮軸） | `Q4_0` | `Q4_0_TRANSPOSE`（交換最後兩軸） |
|---|---|---|---|---|---|
| q_einsum | [N,D,H] | H（輸出） | D | ✗ | ✓ |
| kv_einsum | [2,K,D,H] | H（輸出） | D | ✗ | ✓ |
| attn_vec | [N,H,D] | D（輸出） | N·H | ✗ | ✓（沿 H，256%32=0 等價） |
| gating | [2,F,D] | D（輸入） | D | ✓ | ✗ |
| linear | [F,D] | D（輸出） | F | ✗ | ✓ |
| PLE gate [D,P] / proj [P,D] | | P / D（輸出） | D / P | ✗ | ✓ |
→ 兩種 method 都無法一次對齊所有層（`Q4_0_TRANSPOSE` 只錯 gating）。✅ 讀碼推導。

## 6. 尚未實測 / 待補
- 視覺 MLP、clip 值、pos_emb、音訊 FFN/conv 的 JAX↔HF 轉換。
- HF transformers `Gemma4` 實作是否在 forward 乘 `1/√1536`（本機未安裝 transformers，未查）。
