# 06 待解問題 / 未驗證項目

| # | 問題 | 現況 | 如何驗證 |
|---|---|---|---|
| 1 | repo QAT 對 Gemma 4 是否真的 no-op | 讀碼確認 class 不匹配（`gm/nn/_quantization.py:24,131-149` vs `gm/nn/gemma4/_layers.py:22,50`）；未執行 | 裝 JAX/Flax，用 `QuantizationAwareWrapper(method=Q4_0)` 跑一次 forward，比較與未包裝的 logits 是否 bit-identical；或在 interceptor 內計數 |
| 2 | 官方 QAT 是否用與 repo 相同的 Q4_0 公式（`-max/8`、fp16 scale） | 唯一值 ≤16 ✅，但整數格點吻合率僅 ~40% | 對每塊反推 d（最小非零間距）並檢查 x/d 是否為整數、d 是否為 fp16 可表示 |
| 3 | QAT ckpt 的 `per_layer_model_projection` 為何約小 1/√1536 | 統計證據 ✅（04 §4）；HF 執行時是否再乘 1/√1536 未查 | 安裝 transformers ≥5.6 讀 `modeling_gemma4.py`；或用 MLX/transformers 實跑比較 RMSNorm 前的 RMS |
| 4 | 音訊塔權重本來就是 ≤4 級/32 組的意義 | it/pt/qat 都一樣 ✅ | 檢查 scale 結構（是否為 2-bit + per-block scale）；決定是否可直接 2-bit 打包 |
| 5 | 視覺 MLP、clip 值、pos_emb、音訊 FFN/conv 的 JAX↔HF 轉換 | 依命名推論 ⚠️ | 擴充 `tools/compare_orbax_hf.py` |
| 6 | `convert_to_qat_checkpoint` 改名結果是否與 QAT 模組參數樹對得上 | 未執行 ⚠️ | 修好 #1 後 `model.init` 比對 tree |
| 7 | `peft.quantize` 對 Gemma 4 會在 `per_layer_model_projection` 拋錯 | 讀碼推論 ⚠️ | 實跑；本專案預計不用它（只有 INT4/INT8） |
| 8 | 共享 KV 層的 kv_einsum 刪除是否影響 JAX 端 | JAX 仍建立參數（`_modules.py:275-281`），刪了會在 load 時缺 key | 若要用 QAT HF 權重回灌 JAX，需補零或改碼 |
| 9 | LoRA 無 merge、無 alpha | ✅ 讀碼（`peft/_tree_utils.py:140-145`, `peft/_lora.py:131-132`） | 自寫 merge：對每個 `_LoRAEinsum` 以 lora einsum 計算 ΔW 加回 W |
| 10 | Mac 上 JAX 執行 QAT 可行性 | 未測；jax-metal 實驗性 | 先跑 CPU 小步驟 forward/backward 確認記憶體與速度 |
| 11 | 3GB 目標 | 估算：純文字全 Q4_0 ≈ 2.61GB（01 §4）；官方 QAT 也已量化 embed+PLE ✅ | 實際打包 MLX / compressed-tensors 量檔案大小 |
| 12 | `Gemma4Sampler` prefill / pad 細節 | 未逐行追蹤 | 需要時再補 02 §6 |
| 13 | 視覺 clip 值數值比對 | 腳本 shape 比較失敗（[] vs [1]） | 修正腳本後重跑 |
