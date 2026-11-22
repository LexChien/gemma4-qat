# M4（第一部分）：讓 google-deepmind/gemma 的 QAT 支援 Gemma 4 E2B — v1.0

日期：2026-10-17（台北時間）｜ 環境：MacBook Pro M1 Max 64GB，macOS 15.3.1，`.venv-jax`（jax/jaxlib 0.11.2 CPU、flax 0.12.10、kauldron 1.4.4、orbax-checkpoint 0.12.7、optax 0.2.8）
程式碼：`src/gemma-qat`（從 `src/gemma` 9e1f741 複製，分支 `m4-gemma4-qat`）；`src/gemma` 和 `weights/` 都沒有修改。

## 1. 結論摘要
| 項目 | 結果 |
|---|---|
| (a) QuantizationAwareWrapper／IntWrapper 能辨識 `gemma4._layers.Einsum`／`ClippedEinsum` | ✅ E2B text_only 有 **246** 個 Einsum 被包裝（每層 7 個 × 35 層，加上 `per_layer_model_projection`）；原版是 0 個（trace 05 §5） |
| (b) Q4_0 block 沿 HF 輸入軸（= einsum 收縮軸）切 32 一組，跟官方格點一致 | ✅ 官方 QAT 權重 278 個量化張量 100% 在格點上（PLE projection norm 本來就不量化，所以排除）；我們的實作跟 numpy 參考版只差在 round 的平手值 |
| (c) embed_tokens／PLE 表（4-bit，group 大小可設定）／PLE projection 用 STE 模擬量化 | ✅ `EmbedQATConfig`（embed／ple block 預設 32），tied lm_head 也一起量化 |
| (d) LoadCheckpoint 和參數路徑跟 wrapper 一致 | ✅ wrapper 共用 scope、參數名稱相同，所以 orbax 可以直接載入；遇到 Gemma 4 的樹，`convert_to_qat_checkpoint` 不做任何轉換 |
| 單元測試 | ✅ 8／8 通過（`m4/raw/unit_tests.txt`） |
| 官方 IT orbax → 我們的 Q4 → 跟官方 qat-q4_0-unquantized 比對 | ✅ 完成（數字見 §4） |
| 小型 QAT 冒煙訓練（CPU） | ✅ loss 有下降（數字見 §5） |
| 匯出 → `tools/convert_jax_to_hf.py` → MLX 4-bit g64 → PPL | ✅ 2.5 GB（2.60 GB 參數），PPL 見 §6 |
| jax-metal | ⛔ 沒有測試：最新版 jax-metal 0.1.1 是配 jaxlib 0.4.34 的，跟 jax 0.11.2 不相容；這次全部在 CPU 上跑 |

## 2. 修改內容（`git log`，分支 m4-gemma4-qat）
1. `a726cd7` peft：新增依軸切 block 的量化（`_block_quantization.py`：`contraction_axes`、`block_quantize`、`simulate_block_quantize`（STE））和 `Q4_0_MINMAX`。
2. `ac7d05c` gemma4：`_qat.py`（`EmbedQATConfig`、contextvar `embed_qat()`、`q_rows`）；`Embedder.encode`／`decode`／`encode_logits`／`encode_per_layer_input` 加上 STE 假量化（逐列量化取出的 row，等同於量化整張表）。
3. `b509cdb` gm.nn：`_Gemma4SimulateQuantizedEinsum`（同樣的參數名、initializer、dtype；處理 `{'w':…}` dict；沿收縮軸切 block，用 STE；量化後才乘 `w_scale`；`ClippedEinsum` 會建立 clip 參數）和 `_Gemma4IntEinsum`（`kernel`＋`scale`，scale 沿收縮軸）。新欄位：`block_size`、`quantize_embeddings`、`quantize_ple`、`quantize_lm_head`、`embed_block_size`、`ple_block_size`、`quantize_towers`（預設 False，跟官方一樣不量化 vision／audio 和 mm／audio projection）、`log_replacements`。
4. `02c8b06` ckpts：`convert_to_qat_checkpoint` 遇到 Gemma 4 的樹（有 `embedder.per_layer_embeddings`）就原樣回傳，新增 `is_gemma4_params()`。
5. `71ddee1` gemma4：新增 `snap_params()`（匯出時把參數吸附到格點，跟 wrapper 的 forward 等價）、`CONTRACTION_AXES_BY_NAME`、`_qat_test.py`。
6. `6a519e3` peft：新增 `Q4_0_ABSMAX`（d = absmax/8，clip 到 [-8,7]），建議訓練時用它；`LAST_REPLACEMENTS` 改成只追加、不清空（flax 在同一個 process 裡只會在第一次攔截模組時呼叫 replace fn；之後 jit／grad 不會再記錄，但量化確實有作用，已經驗證過數值）。

收縮軸對照（= HF `[out,in]` 的 in 軸）：q `(1,)`、k `(1,)`、kv `(2,)`、attn_vec `(0,1)`、gating `(2,)`、linear `(0,)`、per_layer_input_gate／per_layer_projection／per_layer_model_projection `(0,)`、input_embedding `(1,)`、per_layer_embeddings `(1,2)`。

專案層級（不在 src）：`tools/convert_jax_to_hf.py` 會讀 `_METADATA.use_zarr3`，自動選 zarr／zarr3 driver。環境方面：Apple Silicon 上沒有 `np.float128`，所以 kauldron 匯入會失敗；在 `.venv-jax/site-packages` 加了 `gemma4qat_np_shim.pth`（把它 alias 成 `longdouble`）。另外補裝了 `dialog`、`bagz`、kauldron 1.4.4（用 `--no-deps` 安裝，因為 macOS 上沒有 tensorflow-cpu）。

## 3. 單元測試（`src/gemma-qat/gemma/gm/nn/gemma4/_qat_test.py`，8 項通過，13 秒）
- `contraction_axes`：8 種 equation 都正確。
- minmax 格點：每個 block 的 level 都在 [-8,7] 而且是整數，而且是冪等的。
- `Q4_0_ABSMAX`：+absmax 會落在 +7·d（被 clip），−absmax 落在 −8·d；`peft.simulate_quantize` 算出來也一樣。
- STE 的梯度是 identity。
- E2B text_only：包裝了 246 個 Einsum，而且包裝前後的參數樹（結構、shape、dtype）完全相同（530 個 leaf）。
- 縮小版 Gemma 4（5 層，PLE，KV 共享）：QAT forward 和「先 `snap_params` 再跑原模型」的 logits 最大差 **3.9e-7**；跟全精度相比差 0.57（表示量化確實生效）；包裝了 36 個 Einsum。
- `ClippedEinsum`（quantize_towers=True）：參數樹相同，輸出等於 clip(clip(x)·Q(w))，誤差 0。
- IntWrapper：初始化／forward 的 shape 都正確（有 `kernel`／`scale`，logits 是 [1,8,512]）。

## 4. 官方格點比對（`m4/raw/q4_vs_official.json`，279 個張量，89 秒；embed／PLE 表每 32 列抽樣）
- 官方 qat-q4_0-unquantized：block 在格點上的比例 = **1.000**（用 least-squares 求 d，殘差 < 0.05）。用 minmax 重新量化時，level 100% 冪等；但在 bf16 下，只有約 84% 的元素逐位元相同（minmax 用 bf16 的 min／max 推回 d 會有 1 ulp 的偏差，level 本身不變）。
- 量化官方 IT 權重之後，level 跟官方 QAT 一致的比例（平均每個元素）：

| 張量類型 | minmax | **absmax/8** | repo Q4_0 |
|---|---|---|---|
| q_proj | 0.679 | **0.806** | 0.785 |
| o_proj | 0.689 | **0.833** | 0.813 |
| gate／up／down | 0.689／0.695／0.701 | **0.818／0.831／0.834** | 0.798／0.811／0.814 |
| embed_tokens | 0.671 | **0.797** | 0.777 |
| PLE 表 | 0.692 | **0.841** | 0.821 |

  整個 block 完全一致的比例都很低（< 4%），因為 QAT 訓練把權重移動了不少：IT 跟 QAT 之間的 rel-L2 是 0.12–0.56。`per_layer_model_projection` 的 rel-L2 是 52.7，`down_proj` 是 1.46，看起來官方在這兩個張量上有重新縮放；不過 block 量化對正的縮放是等變的，所以不影響 level 的計算。
- 結論：用來匯出／吸附格點時選 `Q4_0_MINMAX`（冪等）；訓練時建議用 `Q4_0_ABSMAX`（跟官方 level 最接近）；repo 原本的 Q4_0（fp16 scale、trunc）介於兩者之間，但它的 scale 有正負號，會出現 +8 這個 level，不符合官方格點。
- JAX 實作和 numpy 參考版：最大差 0.0146，只出現在 round 的平手值（我用隨機資料測試，最大差是 7e-9，也就是 1 ulp）。

## 5. 冒煙訓練（`m4_smoke_train.py`，`m4/raw/smoke_train.json`）
設定：官方 IT orbax → `gm.ckpts.load_params(quantize=True, text_only=True)`（bf16，14 秒，RSS 6.1 GB）。參數樹的 shape 跟 `model.init` 完全一致；唯一的差別是上游原本就有的 `mlp.gating_einsum／linear` 會存成 `{'w':…}`（Einsum 和我們的 wrapper 都能處理）。
訓練：QAT `Q4_0_ABSMAX`；embedder 凍結（tables 仍然會被假量化）；decoder 用 f32 訓練；Adafactor lr 3e-4；batch 1 × seq 128；wikitext-2 **train** 取 8 條序列輪流訓練 30 步；held-out 是另一段 train 文字。

| 指標 | 數值 |
|---|---|
| held-out loss（QAT）前 → 後 | **7.282 → 5.090** |
| held-out loss（全精度）前 → 後 | 6.184 → 4.704 |
| 訓練序列 0 的 loss（QAT）前 → 後 | 5.422 → 0.456（過擬合，預期內） |
| 每步時間 | 第一步 30.9 秒（含編譯），之後中位數 **9.9 秒／步** ≈ 12.9 token／秒 |
| 峰值 RSS | **33.8 GB**（dry run 用 seq 32：29.1 GB、7.6 秒／步） |
| 匯出的鍵跟官方 orbax 的文字鍵 | 完全一致（530 個） |

## 6. 匯出 → HF → MLX 4-bit g64（`m4_export.sh`）
- `snap_params(formula='absmax')` → orbax（6.3 GB）→ `convert_jax_to_hf.py --text_only --drop_shared_kv`（14 秒，540 個張量；沒用到的 40 個是 KV 共享層的 k／v，符合預期）→ `out/m4/smoke-hf`（8.7 GB）。
- `m4_check_grid_hf.py`：HF 檔裡 278 個量化張量 **100%** 都在 4-bit／block-32 格點上。
- `mlx_lm convert -q --q-bits 4 --q-group-size 64` → `out/m4/mlx-smoke-q4g64`：**2.5 GB**（參數 2.60 GB，< 3 GB ✅）；生成 117.7 tok/s，峰值 2.70 GB；中文 thinking 生成正常。
- PPL（M0 的方法：ctx 1024、stride 512、加 BOS）：

| 模型 | wikitext | zh-TW |
|---|---|---|
| M0 it-q4g64 | 178.23 | 188.98 |
| M0 qat-q4g64（官方 QAT） | 48.69 | 42.73 |
| **M4 smoke-q4g64（IT＋30 步 QAT）** | **60.31** | **120.40** |
| M4 step0 對照（IT 只吸附格點 absmax、不訓練） | 165.18 | 197.68 |

  注意：wikitext 的大幅下降主要是**領域適應**（訓練資料是 wikitext train，不是 PPL 用的 test），不代表已經達到官方 QAT 的品質；zh-TW 從 189 降到 120，但還是比官方 QAT 的 42.7 差很多。和 step0 對照相比，30 步訓練讓 wikitext 從 165 降到 60、zh-TW 從 198 降到 120，可見 QAT 訓練確實有效果。這次只是驗證整條流程可以跑通。

## 7. 在 M1 Max 上做完整 QAT 的可行性（估算）
- 實測 CPU 約 12.9 token／秒（decoder 訓練、embedder 凍結），相當於約 0.16 TFLOP/s。以 100M token 計算約 90 天，1B token 約 2.5 年 → **M1 Max 不適合做完整 QAT**，只適合做冒煙測試、除錯，或很小規模的 LoRA／QAT 微調。
- 記憶體：decoder f32＋Adafactor 時是 33.8 GB；如果連 embed／PLE 表（2.75B 參數）也要訓練，用 f32＋Adam 需要超過 60 GB，64 GB 的機器放不下。
- GPU／TPU（估算：每 token 約 6×2.0e9 FLOP，MFU 40%）：1×H100 約 33k token／秒 → 1B token 約 8.5 小時（需要 80 GB 以上的 HBM，或凍結 PLE、用 8-bit optimizer）；TPU v5e-8 約 52k token／秒 → 1B token 約 5.3 小時。
- jax-metal 跟 jax 0.11 不相容。如果想用 Mac GPU，要另外建一個 jax 0.4.34＋jax-metal 0.1.1 的舊環境，但 gemma4 原始碼需要新版 jax／flax，所以不建議。

## 8. 已知限制和下一步
- 冒煙訓練只訓練 decoder；QAT 的 STE 雖然有套用在 embed／PLE 上，但表本身是凍結的。完整 QAT 應該用蒸餾（以 IT bf16 當 teacher）、更多元的語料，在 GPU／TPU 上跑。
- MLX 的 g64 affine 會重新量化，沒辦法完全保留 block-32 的格點。下一步（M4 第二部分）：直接把格點打包成 compressed-tensors（pack-quantized g32 symmetric），以及 MLX g32。
- `LAST_REPLACEMENTS` 只會在 process 裡第一次攔截時記錄。

## 9. 檔案
- `m4/tools/`：`m4_grid_probe.py`、`m4_grid_formula.py`、`m4_grid_refine.py`、`m4_q4_vs_official.py`、`m4_tiny.py`、`m4_make_train_tokens.py`、`m4_smoke_train.py`、`m4_export.sh`、`m4_check_grid_hf.py`
- `m4/raw/`：`unit_tests.txt`、`qat_grid_{probe,formula,refine}.json`、`q4_vs_official.{json,log}`、`train_tokens.npy`、`smoke_train{,_dry,_step0}.{json,log}`、`export_smoke{,_step0}.log`、`grid_check_smoke_hf.json`、`bench_m4-smoke-q4g64.json`、`bench_m4-smoke_step0-q4g64.json`
- `out/m4/`：`smoke_orbax`、`smoke-hf`、`mlx-smoke-q4g64`、`smoke_step0_*`（共約 35 GB，可以刪除）
