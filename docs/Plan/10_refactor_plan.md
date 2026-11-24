# 10：從原始碼重建／重構計畫（g4qat）v0.1

> 日期：2026-11-04（UTC+8）｜依據：Lex 追加需求（A＋B＋C 三個範圍）｜不動 `src/gemma`、`weights/`；新程式放 `src/g4qat/`
> 與 M10 任務 1–5 交錯進行：先完成任務 1、2（便宜）→ 重構 B＋C → 任務 3（G6 PEFT）、4（M10 訓練，在新程式碼上跑）→ 任務 5；A 盡量平行。

## 1. 目標
- **A｜JAX QAT 重構**：把 `src/gemma-qat`（gemma repo 的 QAT 改動）與 `docs/Plan/m4,m5/tools` 的 JAX 部分整理成乾淨的新程式碼：Gemma 4 E2B wrapper、Q4_0 g32 官方格點（固定官方 scale）、STE、可合併 LoRA-QAT、KL-to-teacher。
- **B｜自寫 MLX 模型與訓練**：不依賴 `mlx_lm` 內部（不用 `mlx_lm.load`、`models.gemma4_text`、`tuner.trainer.grad_checkpoint`、`batch_generate`）。自寫 Gemma 4 E2B 文字模型（PLE、KV-sharing、sliding/global attention、softcap）、分塊 lm_head／loss、on-grid LoRA-QAT、auto-resume、exact-grid 打包、取樣／生成。
- **C｜整併工具**：`docs/Plan/m0–m9/tools`（約 70 個檔）、`tools/`（5 個）、根目錄腳本（11 個）收斂成一個套件＋CLI（`g4qat <子命令>`）＋測試＋README。舊腳本保留不刪（文件引用的原始數據仍可重現），在 README 標註對應的新指令。

## 2. 套件結構（`src/g4qat/`）
| 模組 | 內容 | 取代的舊工具 |
|---|---|---|
| `grid.py` | Q4_0 g32：官方 scale 讀取、levels_fixed、q4_fixed、MLX／ct／GGUF-free 打包 unpack/pack，round-trip 測試 | m4_grid_*、m8_qat_lora（格點部分）、tools/mlx_exact_grid.py |
| `io/weights.py` | HF safetensors、MLX pack、orbax(JAX) 讀寫與 key 映射 | tools/convert_*、m5_make_text_hf、strip_towers |
| `mlx/model.py` | 自寫 Gemma 4 E2B 文字模型（MLX） | mlx_lm.models.gemma4_text |
| `mlx/qat_lora.py` | QATLoRALinear（STE、固定格點、merge）、teacher bypass | m8_qat_lora |
| `mlx/loss.py` | 分塊 lm_head＋softcap＋CE／KL（Metal 0000000e 對策） | m8/m9_train 內嵌 |
| `mlx/train.py` | 訓練迴圈：micro-batch 累積、全域 step checkpoint（json 提交標記、不刪）、resume、loop 腳本 | m8/m9_train(.sh) |
| `mlx/generate.py` | KV cache、批次生成、chat template（明確 enable_thinking） | mlx_lm.generate/batch_generate |
| `export/mlx_pack.py`、`export/ct.py` | exact-grid MLX 打包；compressed-tensors（G3，本次已完成第一版） | m8/m9_export、tools/pack_compressed_tensors |
| `jax/`（A） | wrapper、格點、STE、LoRA-QAT、KL；與 MLX 共用 `grid.py` 的測試向量 | src/gemma-qat 的改動、m5_jax_lora_qat、m8_parity_jax |
| `data/` | 提示產生、蒸餾、fact-check、SFT build、一般文本 | m8_make_prompts、m8_distill、m9_factcheck、m9_build_sft |
| `eval/` | PPL（M0 測資）、50 題 chat、grade、latency、HF 評測 | m0_bench_*、m6/m7_latency、m7_grade、g4qat/hf_eval |
| `ct_compat.py` | transformers 5.20.1 量化 Embedding 初始化修正 | （新） |
| `cli.py` | `g4qat export-mlx / export-ct / train / eval-ppl / eval-chat / grade / bench / parity` | 各 pipeline.sh 內的指令 |
| `tests/` | pytest：格點 round-trip、pack 無損、STE 梯度、loss 分塊等價、模型 parity | — |

## 3. Parity 測試（B 的驗收）
1. **格點**：`grid` 的 unpack(MLX U32) 與 `mx.dequantize` 一致；ct pack/unpack round-trip 位元一致；官方 pack 經 Q 不變（0 blocks 改變）。
2. **模型 logits**：自寫 MLX 模型 vs `mlx_lm`（同一 M9 pack）：固定 prompt 的 logits max_abs ≤ 1e-2（bf16），argmax 100%；vs HF transformers（bf16 dense, CPU fp32 參考）在 M0 基準的雜訊範圍內（M8 量到官方 kernel 雜訊 0.875）。
3. **生成**：greedy 50 題 token 序列與 mlx_lm 版一致率（預期 ≥ 95%，差異來自 kernel）。
4. **訓練**：同 seed、同資料 10 步，自寫 trainer 的 loss 曲線 vs m9_train 差 < 1%；KL 未訓練時 ≈ 0（M9 雜訊底 0.0015）。
5. **打包**：自寫 export 與 m9_export 對同一 checkpoint 產生的 levels 完全相同（lossless、max_abs 0）。
6. **JAX（A）**：MLX vs JAX 的 Q／STE／LoRA-merge 在測試向量上 0 差異（沿用 M8 parity 方式），E2B 前向 logits 對 HF 參考。

## 4. 順序與里程碑
1. 任務 1（G3 ct）＋任務 2（P1/P3/Q2）：已於 22:5x 以 Terminal 脫離執行（`m10_t12_pipeline.sh`）。
2. **C-1**：建立套件骨架、`pyproject`（`pip install -e src/g4qat` 的 editable 安裝於 `.venv-m0`）、`grid.py`＋測試、`export/ct.py`、`eval/`。
3. **B**：`mlx/model.py` → parity 2/3 → `loss.py`、`qat_lora.py`、`train.py` → parity 4/5。
4. 任務 3（G6 PEFT NaN）：用 `g4qat` 的 hook 工具（逐層 inf/nan 檢查）。
5. 任務 4（M10 訓練）：以新 trainer 跑；資料由 `data/` 產生。
6. **A**：在 B 的測試向量建立後平行進行（JAX 環境 `.venv-jax`）。
7. 任務 5：乾淨系統延遲腳本（`g4qat bench --clean-check`）。
8. **C-2**：舊腳本對照表、README、測試全綠。

## 5. 風險
- 自寫模型與 mlx_lm 的 kernel 差異（sdpa、gelu 近似）會造成小幅 logits 差：以容差＋argmax 驗收。
- MLX 自寫 batch 生成較慢：先單序列＋KV cache，必要時再做批次。
- transformers 5.20.1 的量化 Embedding 需要 `ct_compat`（見任務 1）。
