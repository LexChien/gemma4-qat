# M5：在 Mac（M1 Max 64GB）上做小規模 LoRA-QAT — v1.0

日期：2026-10-17（UTC+8）｜ 起點：**官方 QAT 權重** `gemma-4-E2B-it-qat-q4_0-unquantized`（不是 IT）
程式碼：`src/gemma-qat` 新分支 `m5`（commit `60e24b1`）；專案工具：`tools/convert_hf_to_jax.py`、`tools/mlx_exact_grid.py`、`tools/pack_compressed_tensors.py`
限制遵守：沒有使用 llama.cpp；`src/gemma` 和 `weights/` 都沒有修改；所有產出都寫在 Mac 上。

## 1. 結果總表（M0 方法：MLX、ctx 1024、stride 512、BOS；wikitext-2 **test**＋8 篇繁中維基，全部不在訓練資料裡）
| 模型 | 大小（參數 GB） | PPL wikitext | PPL zh-TW | 生成 tok/s | 峰值記憶體 GB |
|---|---|---|---|---|---|
| 官方 QAT bf16（M0，參考） | 9.3 | 47.22 | 40.34 | — | — |
| **基準：官方 QAT MLX 4-bit g64（M0）** | 2.604 | 48.69 | 42.73 | 84.6 | — |
| 官方 QAT MLX 4-bit g32 affine（M0） | 2.893 | 50.93 | 42.81 | 108.5 | — |
| **官方 QAT MLX g32 exact-grid（M5 新工具）** | 2.893 | **47.17** | **40.27** | 112.6 | 2.99 |
| **JAX LoRA-QAT（70 步）→ MLX g32 exact-grid** | 2.893 | **13.43** | **22.68** | 111.4 | 2.98 |
| JAX LoRA-QAT（70 步）→ MLX g64 affine | 2.604 | 14.09 | 24.44 | 65.7* | 2.69 |
| PEFT LoRA（PyTorch MPS） | — | ✗ 3 次都 NaN 或發散 | | | |
| MLX-LM LoRA／QLoRA | — | ✗ 2 次正式訓練都出現 Metal Internal Error | | | |

\* g64 的生成速度明顯比 M0 基準（84.6）低，判斷是單次量測的波動（同一份權重的 g32 是 111 tok/s）。速度要做正式比較的話，應該用 M0 的 speed pass（各跑兩次）重新量測。

目標「不比基準差、而且小於 3 GB」：✅ 兩個 g32 exact（2.893 GB）都達成；JAX LoRA-QAT g64（2.604 GB，跟基準一樣大）也達成。

## 2. 最佳方案
1. **最推薦的產出：JAX LoRA-QAT（官方 QAT 起點）→ 合併 → 吸附格點（Q4_0_MINMAX）→ MLX g32 exact-grid**：2.89 GB，PPL 13.4／22.7。
2. **完全不訓練、只換打包方式**：官方 QAT 直接用 g32 exact-grid 打包，PPL 從 g64 的 48.7／42.7 降到 47.2／40.3，等於 bf16 的品質，大小 2.89 GB。這是 M4 第二部分「保留 32-block 格點」的成果。
3. 注意：PPL 大幅下降，主要原因是**文體適應**。IT 模型對純文字續寫的 PPL 本來就偏高（pt-bf16 是 7.7／10.9），只要很少的步數，就能把它往純文字的方向拉。這不代表聊天能力變好。目前 128 token 的中文 thinking 生成看起來正常，但正式的對話品質評測還沒做。

## 3. 方法
### 3.1 資料（`m5/raw/data_meta.json`）
- 英文：wikitext-2-raw-v1 **train**（630 篇；授權 CC BY-SA 3.0）。
- 繁中：HF `zetavg/zh-tw-wikipedia`（繁中維基 markdown 轉純文字；用 streaming shuffle，seed 0，取 651 篇、300 萬字）。授權：維基百科內容 CC BY-SA（dataset card 沒有寫明授權）。**注意：CC BY-SA 是 share-alike，不算嚴格意義上的 permissive**。已排除 M0 評測用的 8 個標題（實際抽樣命中 0 篇）。
- 中英文混合後共 4.69M token（每篇文件前面加 BOS）。最後 64×257 個 token 留作 validation。

### 3.2 JAX LoRA-QAT（`gm.nn.gemma4._qat_lora.QATLoRA`，m5 分支）
- **可合併的公式**：y = x·Q(W + s·A·B)，Q 用 STE。所以「合併後吸附格點」的權重，跟訓練時的 forward 完全相同（單元測試 5／5：logits 差小於 1e-6）。原版的 `gm.nn.LoRA` 是 Q(W)+BA，合併後就不在格點上了，而且它沒有 `fuse_params`。
- LoRA 套用在 q／kv／attn_vec／gating／linear：rank 16，α=32，總共 23.6M 參數。embed 和 PLE 表凍結，但仍然會被假量化。
- **量化公式改用 Q4_0_MINMAX，沒有照原計畫用 ABSMAX**。實測（加 BOS 的 validation loss）：不量化 3.597；MINMAX 3.585；ABSMAX 3.631。官方權重本來就在 minmax 格點上，MINMAX 是冪等的；ABSMAX 會把大約 32% 的 block（最大值落在 +7 的那些）重新量化，品質反而變差。
- 訓練：AdamW lr 2e-4（warmup 20 步，cosine），batch 1 × seq 256，CPU 上跑 3600 秒：**70 步、17,920 token**。每步中位數 37.9 秒（單獨跑時約 24 秒；跟 PEFT 或 MLX 同時跑時 60–127 秒）。峰值 RSS 38.1 GB。每 10 步存一次 checkpoint（`out/m5/jax_lora_main/ckpt.pkl`，可以續跑）。
- validation loss：3.585 → 2.985（第 50 步）→ **2.928**（第 70 步）。

### 3.3 匯出（`m5_export.sh`）
`merge_lora` → `snap_params(minmax)` → orbax → `tools/convert_jax_to_hf.py --text_only --drop_shared_kv`（540 個張量）→
- **MLX g32 exact-grid**（`tools/mlx_exact_grid.py`）：直接把格點寫成 MLX affine 格式：q = L+8，scale = bf16(d)，bias = −8·scale。278 個張量、1.45 億個 block，100% 都在格點上；跟 HF 權重的最大誤差 0.002，來自 scale 的 bf16 捨入。
- MLX g64：用 `mlx_lm convert` 做 affine 重新量化（格點沒有保留）。
- **compressed-tensors g32**（`tools/pack_compressed_tensors.py`）：跟官方 w4a16-ct 一樣的版面，276 個 Linear，格點 100%。**大小 6.56 GB，超過 3 GB**，因為 ct 只量化 Linear，embed 和 PLE 表（3.3B 參數）留在 bf16。還沒測試能不能用 transformers 載入。
- 附帶發現：官方的 `w4a16-ct` scale 是 absmax/7.5，也就是對 QAT 權重**重新量化**過，只有 11.6% 的元素跟 QAT 格點一致。我們的 ct 打包則精確保留了 QAT 格點。

## 4. 比較方案和失敗紀錄（都已經停止，不再重試）
| 方案 | 結果 | 紀錄 |
|---|---|---|
| MLX-LM QLoRA（底座用 g32 exact） | 6 iter 的短測可以跑；20–30 iter 時出現 `[METAL] Command buffer execution failed: Internal Error (0000000e)` | 測試輸出 |
| MLX-LM LoRA（bf16 底座），第 1 次 | 第一次 report 之前就出現 Metal Internal Error | `m5/raw/mlx_lora_attempt1_metal_error.log` |
| MLX-LM LoRA（加 grad checkpoint），第 2 次 | 第 83/1000 iter 出現 Metal Internal Error（短測時約 166 tok/s） | `mlx_lora_attempt2_metal_error.log` |
| PEFT 0.21.2 LoRA，MPS bf16，第 1 次 | 第 3 步變成 NaN（LoRA 權重是 bf16，用 AdamW 更新） | `peft_attempt1_nan.*` |
| PEFT，fp32 LoRA＋clip，第 2 次 | 第 14 步 NaN（底座 bf16 在 MPS 上） | `peft_attempt2_nan.*` |
| PEFT，整個模型 fp32，第 3 次 | 沒有 NaN，約 12 秒／步（1024 token，約 85 tok/s，比 JAX CPU 快 8 倍左右），但 loss 從約 3 一路升到 6.6，發散了，停在第 80 步 | `peft_attempt3_fp32_diverged.*` |

推測的原因：MLX 的錯誤跟同時有 JAX 程序（約 25–38 GB）在跑、統一記憶體壓力有關（可能是 GPU timeout）；PEFT 可能是 transformers Gemma4 的訓練路徑在 MPS 上有數值問題，例如 sdpa、sliding mask 或 softcap。這兩件事都還沒有確認。

## 5. 下一步建議
- 把 MLX LoRA 或 PEFT 在**沒有其他程序搶資源**的情況下單獨再跑一次，找出 Metal 錯誤和發散的原因。GPU 路線的速度是 JAX CPU 的 5–8 倍，可以在 1–2 小時內看到 0.5–1M token。如果成功，再接 `snap` 加 exact-grid g32 輸出（工具都已經支援 `--snap`）。
- 對話品質評測：用 chat template 做繁中問答，確認文體適應沒有傷到 IT 能力。必要時加入對話資料，或用 IT bf16 當 teacher 做蒸餾。
- 如果需要 ct 小於 3 GB，必須連 Embedding 一起量化（官方 mobile-ct 就是這樣），但 M0 時 transformers 載入失敗。

## 6. 檔案
- `m5/tools/`：`m5_prepare_data.py`、`m5_jax_lora_qat.py`、`m5_export.sh`、`m5_make_text_hf.py`、`m5_peft_mps.py`、`m5_mlx_lora.yaml`
- `m5/raw/`：`data_meta.json`、`train_tokens.npy`、`mlx_data/`、`jax_lora_probe_Q4_0_{MINMAX,ABSMAX}.{json,log}`、`jax_lora_main.{json,log}`、`export_jaxlora.log`、`bench_m5-{qat-g32exact,jaxlora-g32exact,jaxlora-g64}.json`、MLX／PEFT 失敗紀錄
- `out/m5/`：`mlx-qat-g32exact`（2.7G）、`mlx-jaxlora-g32exact`（2.7G）、`mlx-jaxlora-g64`（2.5G）、`ct-qat-g32exact`、`ct-jaxlora-g32`（各 6.1G）、`jaxlora-hf`、`qat-text-hf`、`jax_lora_main/`
