# M0 基準量測報告（Gemma 4 E2B，Mac M1 Max）

- 版本：v1.0
- 執行：2026-10-17 12:51–14:10（UTC+8），全部在 Lex 的 MacBookPro 上執行
- 硬體：Apple M1 Max、64GB 統一記憶體、macOS 15.3.1
- 限制：**完全沒用 llama.cpp／GGUF**；`src/` 與既有權重沒有修改（只在 `weights/` 新增兩個官方 ct 資料夾）
- 原始數據：`docs/Plan/m0/raw/`；腳本：`docs/Plan/m0/tools/`；量化產物：`out/m0/`

## 1. 結論摘要
1. **已有版本低於 3GB 而且品質過關：`MLX QAT 4-bit g64`** 大小 2.636GB（2.46GiB），PPL 相對 QAT bf16 是 wikitext **+3.1%**、繁中 **+5.9%**，符合 G1 和 Q1（≤10%）。它跟 `quantize_mlx_mixed.py --profile safe` 的輸出完全相同（sha1 一致）。
2. `MLX QAT 4-bit g32` 是 2.926GB，也低於 3GB，但 PPL 是 +7.9%／+6.1%，比 g64 還差，餘裕也只剩 0.07GB，**不建議**。
3. **仿官方 mobile 的混合 2-bit 方案不行：** 2.19GB，但 PPL 是 +146%／+372%。**官方 mobile-ct**（反量化、只模擬權重）也有 +87%／+125%。拿 2-bit 換大小，代價太大。
4. **一定要從 QAT 權重出發：** 拿普通 it bf16 直接做 4-bit，PPL 增幅是 +14.7%／+35.6%（g64），QAT 版只有 +3.1%／+5.9%。
5. 效能方面，4-bit 版生成約 **105–125 tok/s**，載入約 2.4 秒（warm cache），生成時峰值記憶體 2.3–3.0GB，P1、P2、P3 都達標。
6. 官方 compressed-tensors 在 Mac 上：`w4a16-ct` 可以用 transformers 載入（會先解壓成 bf16）；`mobile-ct` 在 MPS 上 **segfault**、在 CPU 上報 AttributeError，**載入失敗**。

## 2. 結果總表
PPL 的 Δ：it 系列跟 `E2B-it bf16` 比，QAT 系列和官方 ct 跟 `QAT-q4_0-unquantized bf16` 比。官方 ct 是反量化成 bf16 後用 MLX 評測，所以速度和記憶體不列（不具代表性）。

| 版本 | 磁碟大小 GB (GiB) | <3GB | PPL wikitext | Δ | PPL 繁中 | Δ | 生成 tok/s | 載入 s | 生成峰值記憶體 GB | PPL 峰值記憶體 GB | 備註 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E2B-it bf16（HF 原檔） | 10.279 (9.57) | ❌ | 155.38 | — | 139.38 | — | 27.9 | 2.83 | 9.31 | 12.26 | 含 vision/audio；MLX 只載文字 |
| QAT-q4_0-unquantized bf16（HF 原檔） | 10.241 (9.54) | ❌ | 47.22 | — | 40.34 | — | 26.2 | 3.33 | 9.31 | 12.23 | QAT 基準 |
| MLX it 4-bit g64 | 2.636 (2.46) | ✅ | 178.23 | +14.7% | 188.98 | +35.6% | 116.4 | 2.41 | 2.70 | 5.70 | 全部 4-bit（含 embed/PLE），純文字 |
| MLX it 4-bit g32 | 2.926 (2.73) | ✅ | 160.93 | +3.6% | 157.53 | +13.0% | 95.9 | 2.43 | 2.99 | 5.99 | 同上 |
| MLX QAT 4-bit g64 | 2.636 (2.46) | ✅ | 48.69 | +3.1% | 42.73 | +5.9% | 117.3 | 2.39 | 2.70 | 5.70 | = quantize_mlx_mixed safe（sha1 相同） |
| MLX QAT 4-bit g32 | 2.926 (2.73) | ✅ | 50.93 | +7.9% | 42.81 | +6.1% | 104.9 | 2.44 | 2.99 | 5.99 |  |
| MLX QAT mobile 混合 2/4/8-bit | 2.193 (2.04) | ✅ | 116.08 | +145.8% | 190.31 | +371.8% | 124.5 | 2.34 | 2.26 | 5.26 | embed 2-bit、MLP≥15 2-bit、PLE 4-bit g128 |
| 官方 qat-w4a16-ct | 8.349 (7.78) | ❌ | 49.80 | +5.5% | 41.81 | +3.6% | — | — | — | — | 反量化後以 MLX bf16 評測（weight-only） |
| 官方 qat-mobile-ct | 2.704 (2.52) | ✅ | 88.34 | +87.1% | 90.60 | +124.6% | — | — | — | — | 反量化後評測；忽略 int8 activation |
| E2B pt bf16（參考） | — | — | 7.72 | — | 10.86 | — | 25.1 | 3.59 | 9.30 | 12.23 | 基礎預訓練版，僅作 PPL 參考 |

- 磁碟大小用的是 apparent bytes（檔案大小加總，含 config 和 tokenizer，不含 `.cache`），單位 GB=1e9；括號裡是 GiB。`du -sk` 的數字見 `raw/sizes.json`。
- 生成速度是 speed pass 兩次的平均（各次數值見 `results_table.json`，g32 兩次分別是 113.1／78.7，波動很大）。第一輪 PPL 測試裡的速度數值，因為當時同時有其他工作在跑，所以不採用。
- 「PPL 峰值記憶體」包含 1024×262144 的 fp32 logits（約 1GB）和 log-softmax 暫存，比實際推論高，不拿來判定 P1。
- **PPL 偏高是正常的：** it／QAT 是指令微調模型，拿來算純文字 PPL 本來就會偏高。pt 基礎版是 7.72／10.86，代表量測方式沒問題。目前**只看相對增幅**。

## 3. 達標對照
| 目標 | 標準 | M0 結果 | 判定 |
|---|---|---|---|
| G1 大小 | < 3.0GB | MLX QAT 4-bit g64 = 2.636GB（純文字，沒有 vision／audio） | ✅（純文字）；含 vision／audio 的還沒做 |
| G2 4-bit | 權重 4-bit | MLX affine 4-bit g64，embed 和 PLE 也是 4-bit | ✅ |
| G3 compressed-tensors | 可載入 | M10：自建 `out/m10/m9-ct-g256`（2.55 GiB）transformers 5.20.1 載入成功（需 `g4qat.ct_compat`）；官方 mobile-ct 在同修正下亦可載入（先前失敗）。PPL 47.01／40.67 對齊 M9 MLX。見 `docs/logs/2026-11-04_M10.md` | ✅ M10 |
| G4 MLX 版 | `mlx_lm` 推論正常 | 所有 MLX 版都能正常生成中英文（thinking 頻道輸出正常） | ✅ |
| Q1 PPL 增幅 | ≤ 10% | QAT g64：+3.1%／+5.9% | ✅ |
| Q2 不劣於 mobile-ct | — | M0：QAT g64（48.69／42.73）優於反量化模擬的 mobile-ct（88.34／90.60）。M10 以 transformers 同腳本重測：m9ct 47.01／40.67 vs 官方 mobile-ct 59.56／61.72，對話亦較優 | ✅（M10 再確認；mobile-ct 原為 W4A8，見 §6） |
| P1 推論記憶體 | 峰值 < 6GB | 4-bit 生成峰值 2.26–2.99GB（MLX）；process RSS 3.4–4.1GB | ✅ |
| P2 生成速度 | ≥ 30 tok/s | 4-bit 約 96–125 tok/s；bf16 約 25–28 | ✅ |
| P3 載入時間 | < 10 秒 | 4-bit 約 2.3–2.5 秒（warm page cache）；cold start 沒量 | ✅（warm） |

建議門檻更新（v0.2，等 Lex 確認）：P2 提高到 ≥ 80 tok/s，Q1 維持 ≤ 10%，但要求 wikitext 和繁中**兩個都要**過。

## 4. 方法
- **環境：** `.venv-m0`（Python 3.13.16，Homebrew 的版本；pyenv 只有 3.14／3.9，所以沒用 3.12），完整套件見 `raw/pip_freeze.txt`。

  | 套件 | 版本 |
  |---|---|
  | mlx / mlx-metal | 0.32.4 |
  | mlx-lm | 0.32.2（內建 `gemma4`／`gemma4_text`） |
  | torch | 2.14.1（MPS 可用） |
  | transformers | 5.20.1（內建 gemma4） |
  | compressed-tensors | 0.19.0 |
  | safetensors / datasets / accelerate | 0.8.0 / 5.1.0 / 1.15.0 |
- **量化：** `mlx_lm convert -q --q-bits 4 --q-group-size {64,32}`（affine）。mlx-lm 的 gemma4 `sanitize()` 會自動拿掉 vision／audio，所以 MLX 版都是**純文字**，而且 `embed_tokens` 和 `embed_tokens_per_layer`（PLE）都會一起量化成 4-bit。mobile 混合方案用 `tools/m0_quant_mixed.py`（修正版，見 §5）。
- **評測資料（`raw/eval_tokens.json`，固定 token id）：** wikitext-2-raw-v1 test 用 `\n\n` 串起來，取前 40,960 個 token；繁中是 zh.wikipedia（variant=zh-tw）8 篇條目（臺灣、臺北市、玉山、日月潭、台積電、故宮、高鐵、阿里山）取前 20,480 個 token，原文存在 `raw/zh_tw_source.txt`。所有版本的 tokenizer.json 都是同一份（sha1 `ba27496…`）。
- **PPL：** 上下文 1024（BOS + 1023 個 token），stride 512，只計算新的 token；log-softmax 用 fp32（`tools/m0_bench_mlx.py`）。
- **交叉驗證：** 前 4096 個 token 用 torch/transformers（MPS）和 MLX 各算一次 it bf16，結果是 wikitext 195.42 對 194.45、繁中 100.55 對 99.61，差距 <1%，MLX 的 PPL 實作可信（`raw/bench_*_x4096.json`）。
- **速度、記憶體、載入：** 固定 prompt「請用繁體中文詳細介紹台灣的地理、歷史與文化，至少寫五段。」，套 chat template，greedy，先 warmup 8 token，再生成 128 token；用 `mlx_lm.stream_generate` 的 generation_tps 和 `mx.get_peak_memory()`；載入時間是 `load()` 加上 `mx.eval(parameters)`。每個模型跑一個獨立 process，連跑兩次取平均（`m0_run_phase2.sh` speed pass）。
- **官方 ct：** 因為 mobile-ct 無法載入，所以用 `tools/m0_ct_dequant.py`，透過 compressed-tensors 的 `unpack_from_int32` 自己做對稱反量化（channel／group），轉成 bf16 HF 目錄，再用同一套 MLX 流程評測。`lm_head.weight` 跟 `embed_tokens` 逐位元相同，所以拿掉後設成 tied。**int8 activation（input_scale／output_scale）被忽略**，所以 mobile-ct 的數字是 W-only 模擬。

## 5. 失敗和問題
| 項目 | 狀況 | 原因／處理 |
|---|---|---|
| `quantize_mlx_mixed.py`（專案根目錄） | 會 TypeError | mlx-lm 0.32 呼叫 predicate 只傳 `(path, module)` 兩個參數，原腳本寫的是三個。修正版放在 `m0/tools/m0_quant_mixed.py`（原檔沒動） |
| mobile 方案 PLE group 256 | ValueError | MLX 只支援 group 32、64、128，改用 128 |
| 官方 mobile-ct 用 transformers 載入 | ❌ MPS segfault（exit 139）；CPU 報 `Gemma4TextScaledWordEmbedding has no attribute 'weight'` | transformers 5.20.1 加 compressed-tensors 0.19.0，對量化過的 embedding 支援不完整，改走反量化評測。log：`raw/ct_load_attempt_mobile.log` |
| 官方 w4a16-ct 用 transformers 載入 | ✅（MPS，解壓成 bf16，記憶體 8.6GB） | smoke：2048 token PPL 47.44／33.68（`raw/bench_ct-w4a16_smoke.json`）。Mac 上沒有原生 int4 kernel，沒有速度上的意義 |
| ct 反量化版用 MLX 載入（第一次） | ValueError：`lm_head.weight` not in model | 確認跟 embed 逐位元相同後拿掉、設 tied，重跑成功 |
| ct 的 Mac 原生 4-bit 推論 | 不支援 | compressed-tensors 的原生執行要用 vLLM／CUDA；Mac 只能用 MLX 版 |
| 含 vision／audio 的 <3GB 版本 | 沒做 | mlx-lm 只支援文字；要做多模態得改用 mlx-vlm（之後的工作） |

## 6. 觀察（給 M2 以後參考）
- **官方 w4a16-ct 不是在 QAT 權重的格點上：** 它跟 qat-q4_0-unquantized 比，線性層的 rel-RMS 誤差約 6.7%（`raw/dequant_check_w4a16.json`），等於重新做了一次 RTN g32 量化。embed 和 PLE 在 w4a16 裡是 bf16，跟 QAT 權重逐位元相同。這跟 trace 文件裡「QAT 權重只有約 40% 符合 repo 縮放公式」的發現一致。
- **官方 mobile-ct 是另外訓練的權重：** 跟 qat-q4_0 的差距 rel-RMS 達 38–56%，而且原本設計是 W4A8／W2A8（靜態 int8 activation）。只模擬權重的 PPL 不能完全代表它在 LiteRT 等目標平台上的品質。
- **g32 和 g64：** QAT 權重原本是 Q4_0 g32，但 MLX affine g32 的 wikitext PPL 反而比 g64 差（50.93 對 48.69）。這表示 MLX 的 affine（含 bias）格點跟官方 Q4_0 格點對不上，M2 應該試著「直接沿用 QAT 格點」來打包，而不是重新 RTN。
- **2-bit 是品質崩壞的來源：** 不管是 embed／lm_head 用 2-bit，還是後段 MLP 用 2-bit，做 PTQ（訓練後量化）都不可行；要用就得靠 QAT（M4）。
- **大小結構：** 4-bit g64 的 2.636GB 裡，PLE（2.35B 參數）約 1.3GB，佔最多。如果之後要加回 vision／audio，需要額外 0.3–0.6GB 的空間。

## 7. 下一步（M2）
1. 以 `MLX QAT 4-bit g64` 作為目前的 <3GB 基準（2.636GB，+3.1%／+5.9%）。
2. 試 mlx-vlm 做多模態 4-bit，量含 vision／audio 的大小。
3. 用 llm-compressor（需要 CUDA）或自己寫打包器，產出 compressed-tensors 版，並同時量化 embed 和 PLE。
4. 研究直接沿用 QAT 的 Q4_0 格點，而不是重新 RTN。

## 8. 檔案
- `docs/Plan/m0/raw/`：`bench_*.json`（每個模型和每次 speed run）、`bench_mlx_all.log`、`bench_phase2.log`、`bench_phase3.log`、`convert_pass{1,2}.log`、`ct_load_attempt_{mobile,w4a16}.log`、`dequant_check_{w4a16,mobile}.json`、`download_ct.log`、`eval_tokens.json`、`zh_tw_source.txt`、`sizes.json`、`results_table.{md,json}`、`pip_freeze.txt`
- `docs/Plan/m0/tools/`：`m0_download_ct.py`、`m0_convert_all.sh`、`m0_quant_mixed.py`、`m0_prepare_data.py`、`m0_bench_mlx.py`、`m0_bench_torch.py`、`m0_ct_dequant.py`、`m0_check_dequant.py`、`m0_sizes.py`、`m0_summary.py`、`m0_make_table.py`、`m0_run_mlx_all.sh`、`m0_run_phase2.sh`、`m0_run_phase3.sh`
- `out/m0/`：`mlx-it-q4g{64,32}`、`mlx-qat-q4g{64,32}`、`mlx-qat-safe-g64`、`mlx-qat-mobile-g64`、`ct-{w4a16,mobile}-dequant-bf16`（各約 10GB，評測用，可以刪）
- `weights/`：新增 `gemma-4-E2B-it-qat-mobile-ct`（2.70GB）、`gemma-4-E2B-it-qat-w4a16-ct`（8.35GB）
