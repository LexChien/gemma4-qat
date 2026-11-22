# M12-01 原始碼追蹤：gemma_native v0.2.1 能吃哪些權重格式、為什麼 SBS 壓不到 2GB、國父差異從哪來

日期 2026-11-20（Asia/Taipei）。全部結論來自讀原始碼或實測，行號以下列快照為準。

## 0. 原始碼來源（已驗證，不是猜的）
- 網站包 `Gemma4-Portable/metadata/manifest.json`：version 0.2.1、`sourceSnapshotSHA256 = 92f6d81f…2efcc`、`upstreamMathematicalBaseCommit = e0d9d1af…`。
- `ALife-Platform/Google-Gemma4/build/native-20261014-r1/provenance/native-20261014-r4/source-manifest.json` 的 sha256 = **92f6d81f…2efcc**（與 manifest 相同）；`portable-zip-20261014-r1/stage-inputs.json` 寫 `sourcePackage: native-20261014-r4`。
- 快照 271 個檔逐一比對 manifest sha256：**0 不一致**。以下 `gemma-native/…`、`gemma.cpp/…` 都指這份快照（M12 複製到 `out/m12/work/src/`）。
- 注意：`ALife-Platform/runs/.../work/source/gemma-native`（10/15 修改過）與 `Google-Gemma4/work/source/gemma.cpp`（有本機 patch）都**不是** 0.2.1 的確切原始碼。

## 1. gemma_native 載入模型的硬性條件
| 位置 | 內容 |
|---|---|
| `gemma-native/executor.cc:22-26` | `map=kTrue`（mmap 整個 SBS）、`to_bf16=kFalse`、`sfp_embedding=kFalse` |
| `gemma-native/executor.cc:27-35` | `fixed_matmul_tiling=true`、`fuse_fixed_q4 = --packed-q4`、greedy（temperature 0、top-k 1） |
| `gemma-native/executor.cc:66-67` | 必須 `model == GEMMA4_2B_LM` 且 config `weight == kQ4_0` |
| `gemma-native/executor.cc:68-69` | 必須 `num_layers==35 && vocab_size==262144 && model_dim==1536`，否則 `Unexpected E2B execution graph` |
| `gemma-native/executor.cc:124-126` | 每步 logits 長度必須 ==262144 |
| `gemma.cpp/gemma/model_store.cc:484-513` | 每個張量 rows/cols 必須等於 config 推出的形狀；**型別直接採用檔案 TOC 裡的型別**（`mat.SetType(file_mat->GetType())`） |
| `gemma.cpp/gemma/tensor_info.cc:62-68, 152-173` | `c_embedding` 形狀 {vocab, 1536}、`ple_embeddings` {vocab, 35×256}、`ple_model_proj` {8960, 1536}。`min_size = kBF16` 只是轉換器提示：全樹只在 `tensor_info.h:64` 宣告、`python/configs.cc:131` 綁定，載入端不檢查 |

## 2. 支援的權重型別與每個元素的位元數
- `compression/types.h:282-297` 型別：f32、bf16、sfp、nuq、i8、q4_0、mxfp4（另有 f64/u32/u64/u16/u8/int8 非權重用）。
- `compression/types.h:305-319` 位元數：sfp 8、i8 8+、bf16 16、**nuq 4.5**、**q4_0 4.5**（`:117-124`，每 32 個一組，bf16 scale + 16 byte）、**mxfp4 4.25**（`:132-142`，每 32 個一組 + 1 byte scale）。**沒有低於 4.25 bit 的型別。**
- `util/mat.h:411-444` `CallUpcasted`：查表（詞嵌入、PLE）支援 f32/bf16/sfp/i8/q4_0/mxfp4/nuq（`GEMMA_ENABLE_NUQ` 預設 1，`types.h:62-63`）。
- `gemma.cpp/CMakeLists.txt:214-220`：bf16/f32/mxfp4/nuq/sfp/i8/q4_0 七種 MatMul 都有編進去。

## 3. c_embedding 與 PLE 怎麼讀
- 詞嵌入查表 `gemma/gemma.cc:230-251` `EmbedTokenFromWeights`：`CallUpcasted` → `DecompressAndZeroPad` 解出一列 → 乘 `sqrt(1536)×scale`。**任何支援型別都可以，包含 Q4_0。**
- 輸出層 `gemma/gemma.cc:1454-1462` `FinalLogits`：沒有獨立 lm_head 時直接拿 `c_embedding` 做 MatMul（tied）。Q4_0 也走一般 Q4_0 MatMul。
- PLE `gemma/gemma.cc:1055-1132`：`x`→bf16（`:1067`）→ 與 `ple_model_proj` MatMul（`:1084`）→ RMSNorm → 以 token id 查 `ple_embeddings` 一列（`:1111`，`CallUpcasted`，任何型別）。
- `--packed-q4` 不影響「能用什麼型別」，只影響 Q4_0 MatMul 是否共用已量化的 activation（`ops/matmul-inl.h:408-431` `PrepareSharedQ4`）。
- 實測：把 `c_embedding`、`ple_model_proj` 改成 Q4_0 的 SBS（候選 A），**原版 binary 可以正常載入並作答**（17×23=391）。

## 4. 檔案格式
- BlobStore V2（`io/blob_store.cc:90-100`）：Header + 資料 + 尾端目錄 + Header；檔尾補齊到 64 KiB（`:40`）。內容 = 每個張量一個 blob + `config` + `tokenizer` + `toc`（張量型別/形狀/scale），498 個 blob。
- tokenizer 是 HF BPE 打包（轉換器 `convert_from_safetensors.py:108-203` `pack_bpe_tokenizer`）。Gemma 4 聊天模板在 `gemma/tokenizer.cc:82-84` 寫死 token id **2、105、106、107、2364、4368**。

## 5. 大小的上下限（size math，build 前算好）
每列（每個 token）成本：c_embedding Q4_0 = 1536/32×18 = **864 B**；PLE Q4_0 = 8960/32×18 = **5,040 B**；合計 **5,904 B/token**。

| 方案 | 檔案大小 | 現有 binary？ | 備註 |
|---|---|---|---|
| 原檔／M11（c_embedding bf16 768 MiB） | 3,221,159,936 B（3.000 GiB） | ✓ | |
| A：c_embedding＋ple_model_proj 改 Q4_0（無損） | 2,622,554,112 B（2.442 GiB） | ✓ | 已實作 |
| 理論下限：全部權重改 MXFP4（4.25 bit，有損、離開 QAT 格點） | 約 2.48e9 B（估算：4,647,157,760 個權重 ×17/32 + tokenizer 7.9 MB） | ✓ | 仍 > 2e9，未實作 |
| B：A＋詞表裁成 154,624 | 1,984,823,296 B（1.849 GiB） | ✗（`executor.cc:68`、`:125` 寫死 262144） | 需另一支 binary |

**結論：現有 binary 不可能做到 < 2,000,000,000 B。** vocab 被寫死成 262144，而 PLE（262144 × 5,040 B = 1.32 GB）加上其他權重，就算全部改成最小的 4.25 bit 也約 2.48 GB。要 < 2GB 只能減少列數（裁詞表），這需要放寬 `executor.cc` 兩處 vocab 檢查並重編 gemma_native。

## 6. 數值路徑（國父差異）
- 每個 MatMul 的輸入 activation 先轉 bf16（例 `gemma/activations.h:757` `x_bf`、`gemma.cc:1067`）。
- **Q4_0 MatMul 在 M ≤ 4 時（decode 每步都是；網站 `--prefill 4` 時 prefill 也是）走 INT8×INT4 核心**（`ops/matmul-inl.h:445-463`）：activation 先量化成 int8，**每列每個 KC 段只有一個 scale = max|x|/127**（`:884-930`，`kMaxKC=6144`，`ops/matmul.h:65`）。MLX 是 bf16 activation × 4-bit 權重。`--packed-q4 0` 不會改掉這條路（只是不共用量化結果），所以 M11 測 `--packed-q4 0` 仍答錯是合理的。
- KV cache 是 bf16（`gemma/kv_cache.h:33`）。
- `GEMMA4_2B_LM` 的 `final_cap = 0`（`gemma/configs.cc:541`），HF/MLX 是 30；softcap 是單調函數，對 greedy 的 argmax 沒有影響。
- 實測結果見 `docs/logs/2026-11-20_M12_sbs_2gb.md`「國父差異」一節。
