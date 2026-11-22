# Gemma 4 E2B QAT 微調與 4-bit 打包流程

基底模型：`google/gemma-4-E2B-it-qat-q4_0-unquantized`（Apache-2.0，未設 gated，不需要 HF token）
架構：35 層、總參數 5.1B（含 PLE 嵌入，有效參數 2.3B）、128K context、多模態（文字、圖片約 150M、音訊約 300M）

## 主要路線：Mac（Apple Silicon）
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U torch transformers accelerate peft trl datasets safetensors mlx-lm
# A) PEFT LoRA（MPS）→ 合併 → 移除視覺/音訊塔（可選）→ MLX 4-bit
python train_lora.py --data sample.jsonl --out out/lora
python merge.py --adapter out/lora --out out/merged
python strip_towers.py --src out/merged --out out/merged_text   # 可選，只保留文字
python quantize_mlx_mixed.py --hf out/merged --profile safe     # 或 --profile mobile
# B) 全程用 MLX：bash mlx_lora.sh lora   或   bash mlx_lora.sh full
```
MLX 的 4-bit 量化預設會連 Embedding（包括 PLE 嵌入表）一起量化，這是壓到 3GB 以下的關鍵。MLX-LM 對 Gemma 4 的支援還沒確認。

## 次要路線：compressed-tensors W4A16（給 vLLM／CUDA 用）
```bash
python quantize_w4a16.py --model out/merged --recipe w4a16                          # 官方方案（8.3GB）
python quantize_w4a16.py --model out/merged --out out/w4a16 --recipe compact   # embed/lm_head int4、PLE int4 g256（相容性未確認）
```

## Mac 的限制
- Unsloth、bitsandbytes（QLoRA）只支援 CUDA
- llm-compressor 的 GPTQ 需要 CUDA；compressed-tensors 的推論主要靠 vLLM（CUDA）
- 記憶體（估計值）：LoRA 建議 16–24GB 以上；全參數微調（凍結嵌入、Adafactor）建議 32GB 以上

## 需要 CUDA 的步驟
`train_lora_unsloth.py`、QLoRA、`quantize_w4a16.py`（GPTQ）、vLLM 推論。建議 GPU 24GB 以上，全參數微調建議 40GB 以上。

## 官方檔案大小（已查證：HF API + safetensors 檔頭）
| 官方版本 | model.safetensors | 說明 |
|---|---|---|
| google/gemma-4-E2B-it（bf16） | 10.28GB（Kaggle） | 非 QAT |
| gemma-4-E2B-it-qat-q4_0-unquantized | 10.21GB | bf16：PLE 4.78、解碼層 3.67、embed 0.81、音訊 0.61、視覺 0.34 GB |
| gemma-4-E2B-it-qat-w4a16-ct | 8.32GB | 只有解碼層的 Linear 是 int4 g32；PLE 4.70GB 跟 embed 0.81GB 維持 bf16，音訊/視覺也是 bf16 ← 這就是它這麼大的原因 |
| gemma-4-E2B-it-qat-mobile-ct | 2.67GB | embed/lm_head int2（per-channel）；PLE int4 g256（1.17GB）；attention 全部加上 MLP 第 0–14 層用 int4；MLP 第 15–34 層用 int2；per_layer gate/proj 跟視覺用 int8；音訊大多 int2；activation 用 int8 static |
| gemma-4-E2B-it-qat-mobile-transformers | 約 2.49GB（Kaggle） | 同一套混合精度（quant_method "gemma"，quantize_embeddings=true） |

## 本專案的大小估計（估計值，依官方張量大小推算）
| 產物 | 估計 |
|---|---|
| MLX `quantize_mlx_mixed.py --profile safe`（全部 4-bit g64，含 embed/PLE，只保留文字） | 約 2.6GB |
| MLX `--profile mobile`（仿官方 mobile 的 2/4/8-bit 混合） | 約 2.0–2.3GB |
| compressed-tensors `--recipe compact`（只保留文字） | 約 2.4GB；保留塔（int8）約 2.9GB |
| compressed-tensors `--recipe w4a16`（官方方案） | 約 8.3GB ❌ |

重點：要壓到 3GB 以下，**一定要把 PLE（embed_tokens_per_layer，24 億參數）量化**。

## 檔案
common.py、quantize_mlx_mixed.py、train_lora.py、train_lora_unsloth.py、train_full.py、merge.py、strip_towers.py、quantize_w4a16.py、mlx_lora.sh、sample.jsonl

## QAT 注意事項
大幅微調會讓權重偏離 QAT 的量化網格。全參數微調請用低學習率（1e-5），量化後一定要評估品質。
