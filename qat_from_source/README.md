# 從原始碼重新做 Gemma 4 E2B QAT（JAX/Flax）

我讀的是 `google-deepmind/gemma` 原始碼，commit 9e1f741（2026-10-15）。

## 原始碼查到的事實
- **有 QAT 程式碼**：`gemma/gm/nn/_quantization.py` 的 `QuantizationAwareWrapper` 會把 Dense/Einsum 換成模擬量化層。量化方法有 `INT4`（per-channel）、`Q4_0`（per-block，block=32）、`Q4_0_TRANSPOSE`、`INT8`、`SFP8`，定義在 `gemma/peft/_quantization_utils.py`；反向傳播用 straight-through estimator（`gemma/peft/_quantization.py`）
- **真 int4 推論**：`gm.nn.IntWrapper` 加上 `peft.quantize(...)`，但只支援 INT4/INT8 per-channel
- **載入 QAT 檔**：`gm.ckpts.LoadCheckpoint(path, quantize=True)` 會呼叫 `convert_to_qat_checkpoint`（`gemma/gm/ckpts/_quantization.py`）
- **支援 Gemma 4**：`gm.nn.Gemma4_E2B`（35 層，`gemma/gm/nn/gemma4/_gemma4.py`）、`gm.text.Gemma4Tokenizer`
- **權重來源**：`gm.ckpts.CheckpointPath.GEMMA4_E2B_PT/IT` 指向 `gs://gemma-data/checkpoints/gemma4-e2b-{pt,it}`（`gemma/gm/ckpts/_paths.py`）。Kaggle 的頁面寫著「Gemma 4 Flax checkpoints will be updated on Kaggle when possible」（`docs/checkpoints.md`）。repo 不能直接讀 HF safetensors
- **LoRA**：有 `gm.nn.LoRA`、`gm.ckpts.SkipLoRA`，範例在 `examples/lora.py`
- **訓練框架**：kauldron（`python -m kauldron.main --cfg=...`），需要 Python 3.12 以上
- **沒有匯出功能**：repo 裡沒有任何轉成 safetensors／HF transformers 的程式碼（找不到 safetensors/huggingface 字樣）

## 檔案
- `setup_mac.sh`：建 Mac 環境
- `qat_e2b.py`：kauldron 的 Q4_0 QAT 設定（全參數，從非 QAT 的 IT 權重開始）
- `qat_mixed.py`：混合精度 QAT 雛形（未測試）
- `export_params.py`：把權重對齊到 Q4_0 網格，存成 safetensors（保留 JAX 原本的命名）

## 流程
```bash
bash setup_mac.sh && source .venv-jax/bin/activate
python -m kauldron.main --cfg=qat_e2b.py --cfg.workdir=$PWD/workdir
python export_params.py --ckpt workdir/checkpoints/ckpt_2000 --out export/
# 需要自己寫 JAX→HF 的命名轉換（目前沒有）→ HF 資料夾 →
mlx_lm convert --hf-path hf_out --mlx-path q4 -q --q-bits 4 --q-group-size 32   # 跟 Q4_0 的 block 32 對齊
```

## Mac 可不可行（評估）
- JAX 在 Mac 上用 CPU 可以跑；jax-metal 是實驗性的，常跟不上新版 JAX，也缺一些運算，這個 repo 能不能用沒有確認
- 全參數 QAT 的記憶體（估計）：5.1B 參數，fp32 權重加 Adafactor 約 25–30GB，再加上 activation。64GB 的 Mac 用 CPU 勉強可以，但會非常慢（每步可能要幾十秒到幾分鐘，未實測）
- 建議：在 Mac 上只做小規模的煙霧測試，正式 QAT 用 TPU（Colab/GCP）或 CUDA GPU（A100/H100 40GB 以上，安裝 `jax[cuda12]`）

## 官方檔案大小（已查證：HF API + safetensors 檔頭）
| 官方版本 | model.safetensors | 說明 |
|---|---|---|
| google/gemma-4-E2B-it（bf16） | 10.28GB（Kaggle） | 非 QAT |
| gemma-4-E2B-it-qat-q4_0-unquantized | 10.21GB | bf16：PLE 4.78、解碼層 3.67、embed 0.81、音訊 0.61、視覺 0.34 GB |
| gemma-4-E2B-it-qat-w4a16-ct | 8.32GB | 只有解碼層的 Linear 是 int4 g32；PLE 4.70GB 跟 embed 0.81GB 維持 bf16，音訊/視覺也是 bf16 ← 這就是它這麼大的原因 |
| gemma-4-E2B-it-qat-mobile-ct | 2.67GB | embed/lm_head int2（per-channel）；PLE int4 g256（1.17GB）；attention 全部加上 MLP 第 0–14 層用 int4；MLP 第 15–34 層用 int2；per_layer gate/proj 跟視覺用 int8；音訊大多 int2；activation 用 int8 static |
| gemma-4-E2B-it-qat-mobile-transformers | 約 2.49GB（Kaggle） | 同一套混合精度（quant_method "gemma"，quantize_embeddings=true） |

## QAT 原始碼能不能做出官方 mobile 的方案（已查證）
- `gm.nn.QuantizationAwareWrapper` 只會把**同一種**方法套到所有 Dense/Einsum 層，**不會**動到 embedding/PLE
- `QuantizationMethod` 沒有 2-bit 選項；內部的 `_simulate_uniform_quantization` 可以支援 bitwidth≥2（`gemma/peft/_quantization.py`）
- 所以：4-bit 的 QAT 可以直接做；2-bit MLP 需要自己在 enum 加 INT2（雛形在 `qat_mixed.py`，未測試）；PLE/embed 沒辦法做 QAT，只能事後量化

## 建議做法（低於 3GB，估計值）
1. 用 `qat_e2b.py` 做 Q4_0 QAT（從 gs://gemma-data/checkpoints/gemma4-e2b-it 開始，18.25GB，可以匿名下載）
2. 用 `export_params.py` 匯出，再轉成 HF（需要自己寫命名對照）
3. 用 `../quantize_mlx_mixed.py --profile safe` 轉成 MLX，約 2.6GB；或用 `../quantize_w4a16.py --recipe compact`，約 2.4GB
## 未確認／限制
- gs://gemma-data 能不能匿名讀取
- Kaggle 上 Gemma 4 Flax 檔案什麼時候上架
- `Gemma4_E2B(text_only=True)` 能不能用
- FFN 第一層要用 `Q4_0` 還是 `Q4_0_TRANSPOSE`
- kauldron 存檔的路徑結構
- JAX→HF 的權重命名對照
