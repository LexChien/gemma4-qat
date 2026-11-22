# Gemma 4 E2B Trace Code — 索引

- 專案：`~/Documents/AI/gemma4-qat`；原始碼 `src/gemma`（`google-deepmind/gemma` commit `9e1f741`，唯讀）
- 權重（唯讀）：`weights/gemma-4-E2B`、`weights/gemma-4-E2B-it`、`weights/gemma-4-E2B-it-qat-q4_0-unquantized`、`weights/orbax/checkpoints/gemma4-e2b-{it,pt}`
- 限制：不使用 llama.cpp。
- 版本：v1.0（2026-10-17 直接在 Mac 重建；10-16 的初版暫存在 box，box 重置後遺失）
- 標記：✅ 讀碼/實測確認　⚠️ 推論或未執行驗證　☑️ 標頭/形狀確認

| 檔案 | 內容 |
|---|---|
| [01_architecture.md](01_architecture.md) | 設定值、各層公式、mermaid forward、參數分解 |
| [02_call_graph.md](02_call_graph.md) | 建構、forward、attention、MLP、PLE、sampler、LoadCheckpoint(quantize)、QAT wrapper、IntWrapper、LoRA、Kauldron |
| [03_api_params.md](03_api_params.md) | 公開 API 完整簽章/預設值 |
| [04_weights_map.md](04_weights_map.md) | 各 ckpt 張量清單、orbax 樹、JAX↔HF 對照、數值比對、Q4 格點 |
| [05_quantization_internals.md](05_quantization_internals.md) | 各 QuantizationMethod 數學、軸、包/跳過、STE |
| [06_open_questions.md](06_open_questions.md) | 待解問題 |
| `raw/` | safetensors 標頭、config、orbax `_METADATA`/tree、`q4_grid_check*.json`、`orbax_vs_hf_compare.{txt,json}`、`compare_run.log` |
| `tools/` | `dump_orbax_tree.py`、`q4_grid_check.py`、`compare_orbax_hf.py`（用 `.venv/bin/python` 在專案根目錄執行） |

## 關鍵結論（摘要）
1. **repo 的 QAT/IntWrapper 對 Gemma 4 不起作用**：只比對 `gm.nn._layers.Einsum`，Gemma 4 用 `gm.nn.gemma4._layers.Einsum/ClippedEinsum`（02 §8）。LoRA 有支援 Gemma 4（02 §10）。
2. **Q4_0 分組方向**：官方沿 HF 輸入軸每 32 一組；repo `_q4_0` 沿 kernel 最後一軸，`Q4_0` 只對 gating、`Q4_0_TRANSPOSE` 只錯 gating（04 §5）。
3. **官方 QAT 也量化了 embed_tokens 與 PLE 表**（≤16 級/32 組，實測），視覺與 embed 投影未量化；音訊原檔就是 ≤4 級（04 §5）。
4. **JAX↔HF 轉換全部 bit-exact 驗證**（文字層含 global/共享層、嵌入、PLE、方陣），**RMSNorm 存 w（不是 1+w）**（04 §3-4）。
5. QAT ckpt 的 `per_layer_model_projection` 約預乘 1/√1536（it/pt 沒有）；其後有 RMSNorm，功能上大致抵銷（04 §4）。
6. orbax 並非全 f32：音訊 151 個張量為 bf16（04 §2）。
7. LoadCheckpoint(quantize=True) 只改名；peft.quantize 無 Q4_0；LoRA 無 merge、無 alpha/r 縮放（02 §7, §10；05 §4）。
