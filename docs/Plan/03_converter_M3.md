# M3 JAX（orbax）→ HF safetensors 轉換器（Gemma 4 E2B）

- 版本：v1.0
- 執行：2026-10-17 14:15–14:50（UTC+8），全部在 Lex 的 MacBookPro 上執行
- 轉換器：`tools/convert_jax_to_hf.py`（專案根目錄）
- 驗證腳本：`docs/Plan/m3/tools/`；原始結果：`docs/Plan/m3/raw/`；輸出：`out/m3/`
- 限制：沒有使用 llama.cpp；`src/` 和 `weights/` 都只讀取、沒有修改

## 1. 結論
**全部通過。** 用 orbax 轉出的 it 和 pt bf16 檔，**整個 `model.safetensors` 跟官方 HF 檔的 sha1 完全相同**（逐位元組一致，連張量順序和 header 都一樣）。2011 個張量的名稱、形狀、dtype 和數值全部一致；transformers 算出的 logits 逐位元相同；MLX 算出的 PPL 和 greedy 生成文字也都跟 M0 的結果完全相同。

| # | 檢查 | 結果 |
|---|---|---|
| 1 | it bf16：2011 個張量的名稱、形狀、dtype、數值都跟 `weights/gemma-4-E2B-it` 一致 | ✅ 2011/2011 bit-exact（`raw/compare_it_bf16.json`） |
| 2 | it bf16：整個檔案的 sha1 | ✅ `10492838…` 和官方相同 |
| 3 | pt bf16：2011 個張量跟 `weights/gemma-4-E2B` 一致 | ✅ 2011/2011（`raw/compare_pt_bf16.json`） |
| 4 | pt bf16：整個檔案的 sha1 | ✅ `b679f17a…` 和官方相同 |
| 5 | it f32（`--dtype f32`）：用 RNE 轉成 bf16 後跟 HF 比 | ✅ 2011/2011（`raw/compare_it_f32_vs_bf16ref_cast.json`） |
| 6 | `--text_only`（Gemma4ForCausalLM 格式，600 個張量）跟 HF 的文字部分比 | ✅ 600/600（`raw/compare_it_text_only.json`） |
| 7 | `--drop_shared_kv`：張量名稱集合跟官方 QAT HF 檔比 | ✅ 1951 個，名稱集合完全相同（`raw/dropkv_vs_qat_names.json`） |
| 8 | transformers 5.20.1（MPS，bf16）載入：沒有缺少或多出的 key | ✅ it、pt、text_only、f32 全部 0 missing、0 unexpected |
| 9 | transformers logits（BOS＋1023 token，wikitext 和繁中）跟官方 HF 比 | ✅ it、pt、text_only 都 bit-exact（max_abs 0）（`raw/logits_torch*.json`） |
| 10 | MLX（mlx-lm 0.32.2）PPL，用 M0 的同一套方法和 token 集 | ✅ it 155.3800／139.3756、pt 7.7153／10.8586，跟 M0 完全相同（`raw/bench_m3-*.json`） |
| 11 | MLX 載入 `--text_only`（Gemma4ForCausalLM 格式） | ✅ PPL 也完全相同 |
| 12 | MLX greedy 生成 128 token 的文字 | ✅ 跟 M0 逐字相同 |

f32 版（附註）：用 f32 權重和 f32 運算時，NLL 是 4.1224／4.7368，bf16 是 4.1365／4.7560，logits 的 max_abs 是 3.6，argmax 一致率 98.7%／97.1%。這是**運算精度不同造成的**，不是轉換誤差：f32 權重轉回 bf16 後是逐位元相同的。

## 2. 用法
```bash
.venv-m0/bin/python tools/convert_jax_to_hf.py \
  --orbax weights/orbax/checkpoints/gemma4-e2b-it --out out/m3/it-bf16 \
  [--hf_ref weights/gemma-4-E2B-it] [--dtype bf16|f32] [--text_only] [--drop_shared_kv]
```
- `--hf_ref`：config、tokenizer、chat_template、processor 等檔案從這個資料夾複製。沒指定時，會依 orbax 名稱自動對應：`-it` 對 `gemma-4-E2B-it`，其他對 `gemma-4-E2B`。
- `--dtype`：輸出 bf16（round-to-nearest-even）或 f32。音訊裡原本就是 bf16 的 151 個張量，選 f32 時會無損升成 f32。
- `--text_only`：只輸出語言模型，key 是 `model.*`，config 用 `text_config`，`architectures=Gemma4ForCausalLM`；不複製 processor 設定。
- `--drop_shared_kv`：拿掉 KV 共享層（15–34 層）的 `k_proj`、`v_proj`、`k_norm`，共 60 個張量，跟官方 QAT HF 檔的格式一樣。
- 輸出 `convert_report.json`：張量數、參數數、沒用到的 orbax 葉節點。
- 需要的套件：numpy、tensorstore、torch、safetensors（`.venv-m0`，tensorstore 0.1.85）。轉換一次約 30 秒，peak RSS 約 18–19.5GB。

## 3. 轉換規則（全部已用 bit-exact 驗證）
規則來自 `trace/04_weights_map.md`。原本標成 ⚠️（推論）的視覺 MLP、clip 值、pos_emb、音訊 FFN、conv、feature、output_proj 這次也全部確認 bit-exact。

| 範圍 | JAX | HF | 轉換 |
|---|---|---|---|
| 文字 attention | `q_einsum.w` [N,D,H] | `q_proj` | `transpose(0,2,1).reshape(N·H,D)` |
| | `kv_einsum.w[c]` [K,D,H] | `k_proj`（c=0）、`v_proj`（c=1） | 同上 |
| | `attn_vec_einsum.w` [N,H,D] | `o_proj` | `reshape(N·H,D).T` |
| 文字 MLP | `gating_einsum.w[0/1]` [F,D] | `gate_proj`／`up_proj` | **不轉置** |
| | `linear.w` [F,D] | `down_proj` | `.T` |
| PLE | `per_layer_input_gate.w`、`per_layer_projection.w` | 同名 | `.T` |
| | `per_layer_embeddings` [V,35,256] | `embed_tokens_per_layer` | `reshape(V,8960)` |
| | `per_layer_model_projection.w` [1536,35,256] | 同名 | `reshape(1536,8960).T` |
| norm／scalar | `*.scale`、`skip_scale` | `*.weight`、`layer_scalar` | 直接複製（不加減 1） |
| 嵌入 | `input_embedding` | `embed_tokens` | 直接複製（不乘 √d） |
| 視覺 | 16 層疊在第 0 軸 `[l]` | `encoder.layers.{l}` | 規則同文字層；clip [16] 變成每層 0 維 scalar |
| | `entry.input_projection.w` | `patch_embedder.input_proj` | `.T` |
| | `entry.pos_emb` [10240,2,768] | `position_embedding_table` [2,10240,768] | `transpose(1,0,2)` |
| 音訊 | `ffn_layer*/linear_*/query/key/value.kernel` | `*.linear.weight` | `.T`（方陣也轉置） |
| | `trans_atten.post.kernel` [8,128,1024] | `self_attn.post.linear` | `reshape(1024,1024).T` |
| | `pos_proj.kernel` [1024,8,128] | `relative_k_proj` | `reshape(1024,1024).T` |
| | `depthwise_conv1d.kernel` [5,1,1024] | [1024,1,5] | `transpose(2,1,0)` |
| | `subsampling_{0,1}.kernel` [3,3,Cin,Cout] | `layer{0,1}.conv` [Cout,Cin,3,3] | `transpose(3,2,0,1)` |
| | `feature.input_proj.kernel` [32,32,1024] | `input_proj_linear` | `reshape(1024,1024).T` |
| | `output_projection.kernel／bias` | `output_proj.weight／bias` | `.T`／直接複製 |
| | 名稱對照 | | `fflayer_start/end`→`feed_forward1/2`、`lconv.ln`→`lconv1d.pre_layer_norm`、`final_ln`→`norm_out`、`trans_atten.pre/post_norm`→`norm_pre/post_attn`、`clip_{input,output}_{min,max}`→`{input,output}_{min,max}` |
| 投影 | `mm_input_projection.w`、`audio_input_projection.w` | `embed_vision`／`embed_audio.embedding_projection` | `.T` |
| 丟棄 | `mm_input_embedding_extra`、`audio_input_embedding_extra` | （HF 沒有對應） | 不輸出 |

orbax 有 1318 個葉節點，完整轉換時用到 1316 個，只有上面兩個 extra 沒用到。

## 4. 過程中修正的問題
| 問題 | 原因 | 修正 |
|---|---|---|
| 第一次轉換有 928 個 shape mismatch（`[1]` 對 `[]`） | `np.ascontiguousarray` 會把 0 維 scalar 變成 1 維 | `to_torch` 改成保留原本的 shape（`raw/run_it_pass1.log`、`compare_it_bf16_pass1.json`） |
| logits 測試卡住超過 12 分鐘 | 用 `dtype=bfloat16` 載入 20GB 的 f32 檔，transformers 在 CPU 上逐張轉型，極慢 | f32 檔改用 f32 載入（`raw/run_load_attempt1_hung.log`） |
| 備註 | transformers 載入完整 it 時，進度顯示 1951 個權重，比檔案裡的 2011 少 | KV 共享層的 k/v/k_norm 不會被載入（transformers 會忽略），沒有 missing 或 unexpected，不影響結果 |

## 5. 對後續的意義
- M4 從原始碼做 QAT 時，訓練出的 orbax 權重可以直接用這個轉換器轉成 HF，再接 MLX 4-bit（M0 路線）、compressed-tensors 和 PEFT。
- 建議 M4 的 QAT 產出加上 `--drop_shared_kv`，跟官方 QAT HF 格式保持一致；加上 `--text_only`，可以直接給 mlx-lm 和 Gemma4ForCausalLM 使用。
- 這次只驗證了 it 和 pt 兩份官方 orbax 權重。repo 的 QAT 訓練輸出會不會保留同樣的樹狀結構（例如被 QAT wrapper 改名），要等 M4 實際產出後再確認。

## 6. 檔案
- `tools/convert_jax_to_hf.py`：轉換器
- `docs/Plan/m3/tools/`：`m3_compare_tensors.py`（逐張量比對）、`m3_logits_torch.py`（transformers logits）、`m3_run_it.sh`、`m3_run_rest.sh`、`m3_run_load.sh`。PPL 沿用 `m0/tools/m0_bench_mlx.py`
- `docs/Plan/m3/raw/`：`compare_{it_bf16,pt_bf16,it_f32_vs_bf16ref_cast,it_text_only}.json`、`compare_it_bf16_pass1.json`、`dropkv_vs_qat_names.json`、`logits_torch.json`、`logits_torch_pt.json`、`bench_m3-conv-{it,it-text,pt}-bf16.json`、`run_it.log`、`run_it_pass1.log`、`run_rest.log`、`run_load.log`、`run_load_attempt1_hung.log`
- `out/m3/`：`it-bf16`（9.6G）、`pt-bf16`（9.6G）、`it-f32`（19G）、`it-text-bf16`（8.7G）、`it-dropkv-bf16`（9.5G），共約 57GB，都是驗證用，可以刪
