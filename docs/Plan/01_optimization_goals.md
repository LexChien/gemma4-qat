# Gemma 4 E2B QAT 優化計畫：優化目標

- 版本：v0.1（草案）；2026-11-13 增補 §10 M10 進度
- 建立：2026-10-16 01:27（台北時間）
- 硬體：Apple M1 Max、64GB 統一記憶體、macOS 15.3.1
- 限制：**不使用 llama.cpp（不產 GGUF）**

## 1. 總目標
以 `google-deepmind/gemma` 原始碼（commit 9e1f741）從底層重做 QAT，產出一個自建的 Gemma 4 E2B 4-bit 版本：可用 LoRA／全參數微調，可導入 PEFT、Unsloth，可用 compressed-tensors 打包，最終檔案 < 3GB。

## 2. 量化目標（必達）
| 編號 | 目標 | 標準 | 驗證方式 |
|---|---|---|---|
| G1 | 檔案大小 | 4-bit 成品 **< 3.0GB**（含 tokenizer／config） | `du -sh` 實測 |
| G2 | 4-bit 格式 | 權重 4-bit（必要處可 2/8-bit 混合） | 讀 safetensors header 確認 dtype／位元 |
| G3 | compressed-tensors | 產出可被 transformers／vLLM 讀取的 compressed-tensors 版 | `from_pretrained` 載入成功 |
| G4 | MLX 版 | 產出 Mac 可直接推論的 MLX 4-bit 版 | `mlx_lm.generate` 正常輸出 |
| G5 | 微調相容 | LoRA 與全參數微調皆能在 QAT 權重上執行 | PEFT 跑完 smoke test、loss 下降 |
| G6 | 框架相容 | 可導入 PEFT；Unsloth（需 CUDA） | PEFT 在 Mac 實測；Unsloth 於 CUDA 環境實測 |

## 3. 品質目標
| 編號 | 目標 | 標準（暫定，待基準量測後調整） |
|---|---|---|
| Q1 | 困惑度 | 自建 4-bit 相對 bf16 基準 perplexity 增幅 **≤ 10%** |
| Q2 | 對照官方 | 不劣於官方 `qat-mobile-ct`（2.67GB）同測資結果 |
| Q3 | 任務能力 | 指定評測集（待定）分數下降 ≤ 3 個百分點 |
| Q4 | 對話正常 | 中英文對話、chat template 正常，無亂碼 |

## 4. 效能目標（Mac M1 Max）
| 編號 | 目標 | 標準（暫定） |
|---|---|---|
| P1 | 推論記憶體 | 峰值 < 6GB |
| P2 | 生成速度 | MLX 4-bit ≥ 30 tokens/s（待基準量測後調整） |
| P3 | 載入時間 | < 10 秒 |

## 5. 範圍
- **包含：** 文字能力為主；QAT（Q4_0／INT4）、PLE 與 embedding 量化、LoRA、全參數微調、MLX 與 compressed-tensors 匯出。
- **待決定：** 是否保留圖片（vision）與音訊（audio）編碼器（保留約多 0.3–0.6GB）。
- **不包含：** llama.cpp／GGUF。

## 6. 壓到 3GB 以下的策略
依官方實測（`qat-w4a16-ct` 8.32GB vs `qat-mobile-ct` 2.67GB），關鍵是：
1. PLE（`embed_tokens_per_layer`，4.70GB bf16）量化為 int4 group 256（約 1.17GB）。
2. `embed_tokens`／`lm_head` 量化（int4 per-channel 或 int2）。
3. Decoder 層 int4（Q4_0 group 32），後段 MLP 視品質考慮 int2。
4. 視需求移除或 int8 量化 vision／audio。

## 7. 階段與里程碑
| 階段 | 內容 | 產出 |
|---|---|---|
| M0 | 基準量測：bf16 與官方 4-bit 的大小、perplexity、速度 | `docs/logs` 基準報告 |
| M1 | 環境建置（Python 3.12、JAX、gemma 原始碼、MLX、PEFT） | 可執行環境 |
| M2 | 直接量化官方 QAT 權重 → MLX／compressed-tensors，實測大小 | 第一個 < 3GB 版本 |
| M3 | 撰寫 JAX（orbax）→ HF safetensors 轉換器並驗證數值一致 | `convert_jax_to_hf.py` |
| M4 | 從原始碼跑 QAT（smoke test → 完整訓練，可能需 TPU／CUDA） | 自建 QAT 權重 |
| M5 | LoRA／全參數微調實測（PEFT on MPS；Unsloth on CUDA） | 微調報告 |
| M6 | 最終打包、評測、文件 | 成品＋`docs/PDF` 報告 |

## 8. 風險
- M1 Max 64GB 跑完整 QAT 可能記憶體不足或極慢 → 備案：雲端 TPU／CUDA。
- repo 無 2-bit QAT 選項 → 需自行擴充或只做訓練後量化。
- mlx-lm／llm-compressor 對 Gemma 4 與 PLE 量化的支援未實測。

## 9. 待 Lex 確認
1. 是否保留 vision／audio？
2. 評測集與品質門檻（Q1–Q3 數字）。
3. 完整 QAT 是否允許使用雲端 TPU／CUDA？


## 10. 進度更新（2026-11-13，M10 任務 1–2）
狀態細節與數字見 `docs/logs/2026-11-04_M10.md`、`02_baseline_M0.md` §3。此處只標必達項在 M10 之後的判定：

| 編號 | 判定 | 依據 |
|---|---|---|
| G3 | ✅ | 自建 `out/m10/m9-ct-g256`（2.55 GiB，<3GB）可用 transformers 5.20.1 `from_pretrained` 載入（需 `g4qat.ct_compat`）；PPL 與 M9 MLX 幾乎一致。官方 mobile-ct 在同修正下亦可載入。vLLM 未測。 |
| Q2 | ✅ | 同測資：m9ct PPL 47.01／40.67，官方 mobile-ct 59.56／61.72；50 題思考關知識 12/15 vs 10/15、指令 15/15 vs 14/15、亂碼 10/10 vs 9/10。 |
| P1／P3 | ✅（M9 MLX 4-bit） | 生成峰值 2.98 GB、載入 2.62 s。HF 解壓路徑約 23 GB／17 s，不作為判定。 |
| G6 | ⏳ | 任務 3，尚未開始。 |
