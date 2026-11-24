# M7：讓 Gemma 4 E2B QAT「真正關閉思考」v1.0

> 日期：2026-10-18（時間皆為 UTC+8）｜機器：MacBook Pro M1 Max 64GB｜目錄：`docs/Plan/m7/`（`tools/`、`raw/`），輸出在 `out/m7/`
> 限制：不使用 llama.cpp／GGUF；未修改 `src/gemma`、`weights/`、`src/gemma-qat`。

## 1. 結論（重要更正）

**官方 QAT 模型本來就能完全關閉思考，不需要再訓練。** M6 的結論「E2B QAT 檢查點實際上沒辦法關閉思考」是**評測工具造成的錯誤**，原因有兩個：

1. **mlx_lm 0.32 預設開啟思考。** `TokenizerWrapper.apply_chat_template()` 在呼叫端沒有傳 `enable_thinking` 時，會自動補上 `enable_thinking=self.has_thinking`；Gemma 4 有 `<|channel>thought`，所以這個值是 True（`mlx_lm/tokenizer_utils.py` 第 394–395 行）。M6 第一版 chat eval 和 `mlx_lm generate` CLI 都因此被開啟了思考。
2. **M6 修正時用了大模型的格式。** M6 在 prompt 後面預填空的 `<|channel>thought\n<channel|>`，但這是 12B／26B／31B 的關閉格式。E2B／E4B 的官方關閉格式**不會出現任何 channel token**。對 E2B 預填空的 thought 區塊，反而會讓它把推理寫進答案。

這次用正確格式重新量測 50 題：

- 官方 QAT、M5、M6 三個模型都是 **50/50 沒有 thought 區塊、0 題推理外洩**。
- 首個答案 token 的 median 為 **0.043 秒**，總時間 median 為 **1.09 秒**，decode 約 100 tok/s。

依據正確結果，**M7 不進行 SFT 訓練**，以免花好幾個小時解決一個不存在的問題。改為交付以下內容：

- 修正後的評測工具。
- 一個「預設不思考」的打包版本：權重和官方相同、用 hardlink，不額外佔空間；只改 chat template。

## 2. Chat template 追蹤（第 1 步）

來源：`out/m5/qat-text-hf/chat_template.jinja`，和 `weights/gemma-4-E2B-it`、qat-q4_0-unquantized、mlx-qat-g32exact 的 md5 相同（`9d795d4e…`）。官方文件：ai.google.dev/gemma/docs/capabilities/thinking，以及 HF `google/gemma-4-E2B` README。

| 狀態 | E2B／E4B 正確 token 序列 | 模型輸出 |
|---|---|---|
| 思考**關** | `<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n`（最後幾個 id：106 107 105 4368 107） | 直接回答，最後輸出 `<turn|>`（106） |
| 思考**開** | `<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{prompt}<turn|>\n<|turn>model\n` | `<|channel>`(100) `thought\n…` `<channel|>`(101)，接著是答案，最後 `<turn|>` |
| 只適用 12B 以上（**E2B 不適用**） | 思考關時在 prompt 尾端多加 `<|channel>thought\n<channel|>` | — |

補充：

- 範本裡的 `enable_thinking` 預設為 false（`| default(false)`），但 mlx_lm 會覆蓋這個預設值。
- 有系統訊息時，`<|think|>` 會放在第一個 system turn 的最前面。
- 歷史中的 assistant 訊息會經過 `strip_thinking()` 去掉思考內容；只有在最後一則 user 訊息之後的 `reasoning_content` 才會以 thought channel 的形式呈現。

## 3. 評測結果（50 題 zh-TW，greedy，思考關，上限 256 tokens）

背景 CPU 負載（開始時記錄在 `raw/cpu_hogs_start.txt`）：Chrome Helper 約 100%、Logi Options+ 約 65%、mediaanalysisd 約 43%、mds_stores 約 34%。依指示沒有關閉任何使用者程式。

### 思考關的遵從度與品質

| 模型 | 思考關遵從 | 出現 thought 區塊 | 推理外洩 | 用英文回答 | 知識 | 指令 | 垃圾／不可答 | 繁體字比例 | 平均 rep4 | 平均字數 |
|---|---|---|---|---|---|---|---|---|---|---|
| **官方 QAT g32 exact（正確格式）** | **50/50** | 0 | 0 | 1（g01 亂碼題） | 12/15 | **14/15** | **8/10** | 0.9994 | 0.030 | 179 |
| M5 JAX LoRA-QAT g32 exact | 50/50 | 0 | 0 | 0 | 11/15 | 14/15 | 5/10 | 0.987 | 0.044 | 168 |
| M6 MLX LoRA snap g32 exact | 50/50 | 0 | 0 | 1 | 12/15 | 14/15 | 8/10 | 1.000 | 0.039 | 184 |
| 官方 QAT ＋ 空 thought 預填（M6 的錯誤做法） | 6/50 | 32 | 37 | 37 | 12/15 | 4/15 | 5/10 | 0.982 | 0.148 | 604 |

判定方式：

- **thought 區塊：** 生成內容中出現 `<|channel>` 或 `<channel|>`。
- **推理外洩：** 出現 "Thinking Process"、"The user is…"、"**Analyze"、"Let me think" 等句型。
- **用英文回答：** 去掉程式碼和 LaTeX 後，英文字母比例超過 0.5（翻譯、英文問候、JSON 題除外）。這是語言選擇的問題，和思考無關。

### 端到端延遲（思考關，warm；每格 median / P90）

| 模型 | run | TTFT | 首個答案 token | 總時間 | decode tok/s | 答案 tokens | cold 首題 TTFT／總時間 |
|---|---|---|---|---|---|---|---|
| 官方 QAT g32 exact | 1 | 0.043 / 0.064 s | 0.043 / 0.064 s | **1.09 / 2.68 s** | 100.1 / 107.4 | 102 / 256 | 0.18 / 0.42 s |
| 官方 QAT g32 exact | 2 | 0.050 / 0.077 s | 0.050 / 0.077 s | 1.14 / 3.03 s | 91.3 / 95.6 | 83 / 256 | — |
| M5 JAX LoRA | 1 | 0.061 / 0.082 s | 0.061 / 0.082 s | 0.89 / 3.47 s | 76.8 / 111.9 | 67 / 256 | 0.17 / 1.60 s |
| M5 JAX LoRA | 2 | 0.066 / 0.096 s | 0.066 / 0.096 s | 1.10 / 4.03 s | 67.0 / 73.8 | 67 / 256 | — |
| M6 MLX LoRA | 1 | 0.073 / 0.082 s | 0.073 / 0.082 s | 1.87 / 4.62 s | 58.0 / 74.2 | 105 / 256 | 0.18 / 0.41 s |
| M6 MLX LoRA | 2 | 0.082 / 0.091 s | 0.082 / 0.091 s | 2.09 / 5.09 s | 52.2 / 57.1 | 98 / 256 | — |
| （對照）M6 思考開，官方 | — | 0.11 s | 8.2 / 14.4 s | 9.8 / 23.6 s | 約 44（背景負載重） | — | — |

三個模型的權重格式完全相同。decode 速度差異來自執行期間背景負載逐漸上升：M5、M6 是在官方模型之後才跑的，run 2 也比 run 1 慢。

- **TTFT／首個答案 token 已達「flash」等級：** 50–80 ms。
- **總時間主要由答案長度決定：** P90 都是撞到 256 token 上限的長答案。

### 其他指標

- **思考開仍然正常：** M6 的思考開評測有明確傳入 `<|think|>`，所以結果有效。官方模型 25 題的成績是知識 6/8、指令 6/7、垃圾題 2/5，繁體字比例 1.000。新的打包版本用 `{"think": true}` 測試，確認仍會輸出 `<|channel>thought`。
- **大小和 PPL：** 打包版本和官方 g32 exact 權重相同，參數 2.89GB（磁碟 2.7GB，用 hardlink 不重複佔用），PPL 47.17／40.27（M5 量測）。

## 4. 交付內容

1. **`out/m7/mlx-qat-g32exact-nothinkdefault/`：** safetensors 是官方 g32 exact 的 hardlink，只改了 `chat_template.jinja`：`enable_thinking = think | default(false)`。
   - 效果：`mlx_lm generate`、`mlx_lm server` 不帶參數時直接回答（已驗證：「台灣最高的山是**玉山**。它的海拔是**3,952 公尺**。」）。
   - 開啟思考的方式：`--chat-template-config '{"think": true}'`，或在 server 請求帶 `chat_template_kwargs`。
   - 代價：這個版本會忽略 `enable_thinking=True`，所以 HF 或其他框架要開思考時，也得改傳 `think=True`。
2. **不改模型的做法：** 繼續使用官方模型，呼叫時一律明確傳 `enable_thinking=False`（mlx_lm CLI 用 `--chat-template-config '{"enable_thinking": false}'`，server 用 `chat_template_kwargs`）。
3. **修正後的評測工具：**
   - `m7_latency.py`：思考關時明確關閉、不預填；保留 `--empty-prefill` 只供對照；會保存原始 token 文字。
   - `m7_grade.py`：計算遵從度，並把推理外洩和用英文回答分開統計。
   - `m7_baseline.sh`：可續跑的量測管線。
   - `m7_progress.sh`：Terminal 進度條，並寫入 `out/m7/STATUS.txt`。

## 5. 為什麼沒有訓練，以及若仍要做 SFT 的方案

- **沒有需要修的行為：** 遵從度已經是 50/50，SFT 的主要目標不存在。
- **JAX LoRA-QAT 太慢：** M5 實測約 5 tok/s（70 步 × 256 tokens 花了 1 小時）。一份約 30 萬 token 的 SFT 資料要跑大約 17 小時。
- **一般 MLX LoRA 不適合 g32 exact：** snap 後 PPL 從 13.2 退到 27.3，且約每 300 iter 會出現 Metal 錯誤。

若 Lex 仍想改善中文風格，例如 g01 亂碼題用英文回答、garbage 題 8/10：

- **資料：** 用官方 QAT 模型思考開蒸餾答案（去掉 thought），題目從 zh-TW 維基標題（CC BY-SA 4.0）加上程式產生的指令題組成；生成內容依 Gemma Terms of Use 使用；以 OpenCC s2twp 轉為臺灣用字，過濾英文推理和撞上限的答案；另外混入約 20% 思考開樣本；loss 只算 assistant tokens。
- **訓練：** 需要把 M5 的可合併 LoRA-QAT 移植到 MLX，也就是在 forward 中先對 `W + s·BA` 做 32-block minmax 假量化（STE），這樣 g32 exact 打包才會無損；並修好 resume 的編號，加上自動續跑迴圈。

**建議另立 M8 處理，不屬於「關閉思考」的範圍。**

## 6. 阻礙

- **Metal 錯誤未解決：** MLX 訓練的 Metal Internal Error 仍存在（M6），但 M7 沒有訓練，所以不受影響。
- **速度量測仍受背景負載影響：** run 2 比 run 1 慢約 10%，M5、M6 的 decode 也被拉低。
