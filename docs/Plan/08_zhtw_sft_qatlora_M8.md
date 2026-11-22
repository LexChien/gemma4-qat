# M8：zh-TW 蒸餾 SFT × 官方格點可合併 LoRA-QAT（MLX）v1.0

> 日期：2026-10-18（Asia/Taipei）｜機器：M1 Max 64GB｜框架：MLX 0.32.4／mlx_lm 0.32
> 工具：`m8/tools/`｜原始數據：`m8/raw/`｜交付：`out/m8/mlx-m8-g32exact`（2.89GB）｜日誌：`../logs/2026-10-18_M8.md`
> 依據：`07_nothink_M7.md` 第 5 節（Lex 核准）

## 1. 目標
在不破壞思考關（官方 E2B 格式）與思考開的前提下，提升 zh-TW 回答品質：繁體與臺灣用語、修正 g01 英文回覆、亂碼／無法回答提示的處理。

## 2. 方法
### 2.1 MLX 版 on-grid 可合併 LoRA-QAT（`m8_qat_lora.py`）
- 前向：y = x·Q(W + s·AB)ᵀ，Q 為 Q4_0 g32、STE（w + stop_gradient(q − w)）。
- **格點固定為官方 pack 的 scale d**（bias = −8d），L = clip(round(W′/d), −8, 7)。理由：官方 scale 無封閉公式可重現（minmax 改變 55% blocks、absmax/8 改變 74%）。
- **基底改用 deployed-dense**（pack 反量化），因 HF bf16 與 pack 差 bf16 scale 捨入（≤0.0039）；基底因而是 Q 的不動點。
- 目標 205 個 linear（q/k/v/o/gate/up/down；KV-sharing 下 k/v 僅 15 層），rank 16、scale 2，可訓參數 24,158,208。
- Parity（`raw/parity.txt`）：MLX vs JAX minmax 0/768 blocks 不同、y 差 0；STE 梯度非零；真實模型 57,360,384 blocks 經 Q 後 0 改變、max_abs 0。

### 2.2 蒸餾 SFT 資料（`m8_make_prompts.py`、`m8_distill.py`、`m8_build_sft.py`）
- 1197 個自建提示：主題×模板知識題、程式化指令題（算術／排序／JSON／是非）、自寫亂碼澄清回覆 8 種、無法回答題 ~124、維基導言閱讀題 153、思考開子集 220。排除評測集實體，並以 4-gram Jaccard > 0.5 再剔除 86 筆。
- 教師：官方 QAT（`out/m5/mlx-qat-g32exact`），加 zh-TW 指引 system prompt（context distillation；學生訓練時不含指引），`batch_generate` batch 32。
- 過濾：簡體 27、英文 50、無拒答 10、超長 2；OpenCC s2twp 改寫 377 筆答案。
- 結果：train 971／valid 51；類別 know 324、know_think 95、inst 176、inst_think 18、read 144、read_think 42、gib 110、unans 81、unans_think 32；target tokens 167,744（只對 assistant token 計 loss）。
- 來源與授權：提示為自寫／程式產生；閱讀題導言取自中文維基百科（CC BY-SA 4.0）；答案由 Gemma 4 E2B（Gemma 使用條款）生成。

### 2.3 訓練（`m8_train.py`、`m8_train_loop.sh`）
- 2 epoch、batch 4（micro-batch 1 梯度累積，以 batch 總 target token 正規化）、lr 1e-4、warmup 20、cosine 至 0.1×、486 步。
- Checkpoint：`step_{全域步數:06d}`，json 最後寫入作為提交標記，每 5 步，保留最後 3 個；auto-resume loop 上限 400 次。
- **Metal `0000000e` 問題**：前 6 次嘗試皆崩潰（含 micro-batch 後）。根因為 GPU 逾時（`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`），262k 詞表 lm_head 與其 split-k 反向 matmul 是已知元凶（MLX #1231／#1363）。**修正：hidden states 後每 128 位置分塊計算 lm_head＋softcap＋CE**。修正後 04:28–08:03 一次跑完 481 步、0 次崩潰（約 26 秒／步，峰值 12.2GB）。
- Val loss：0.801 → 0.446（100）→ 0.382（200）→ 0.365（300）→ 0.359（400）→ **0.353（486）**。

### 2.4 打包（`m8_export.py`、`m8_verify.py`）
只改 205 個目標 linear 的 4-bit level，scale／bias 原封沿用官方 pack，其餘張量照抄。

## 3. 結果（M8 vs 官方 QAT g32exact）

| 項目 | 官方 QAT | **M8** |
|---|---|---|
| 大小 | 2,893,390,749 B（2.7GiB） | 2,893,390,749 B（2.7GiB，<3GB ✅） |
| 打包無損（dequant vs 訓練權重 max_abs） | — | **0.0**（lossless ✅）；改變 11.74M／57.36M blocks、44.5M levels、clip 1.81M |
| Logit 檢查（打包 vs 訓練時） | 0.875（kernel 雜訊基準） | 0.8125、argmax 一致 100% |
| PPL wikitext／zh-TW | 47.17／40.27 | **102.62／93.60**（↑2.2–2.3×）❌ |
| **思考關（50 題）** 遵從（無 thought） | 50/50 | 50/50 ✅ |
| 知識 | 12/15 | **10/15** ❌（k01 玉山→「太魯閣山」、k06 南投→「花蓮縣」） |
| 指令 | 14/15 | **15/15**（i14 10 字標題修正） |
| 亂碼／無法回答 | 8/10 | **10/10**（g01、g05 修正） |
| 英文回覆 | 1（g01） | **0** ✅ |
| 繁體比率（含簡體字之答案數） | 0.9994（1） | **1.000（0）** ✅ |
| 平均答案長度／觸頂 256 | 179 字／14 | 104 字／7 |
| 延遲：首 token／總時間 median（2 輪） | 0.098–0.106 秒／2.14–2.31 秒 | 0.093–0.094 秒／**0.73–0.86 秒** |
| 解碼 tok/s median | 40.9–42.6 | 46.2–48.9 |
| **思考開（25 題，stride 2）** 有 thought block | 24/25 | 25/25 ✅ |
| 知識／指令／亂碼 | 6/8、6/7、2/5 | 6/8、**7/7、5/5** |
| 英文回覆 | 2（k11、g01） | 2（k11、i06） |
| 繁體比率 | 1.000 | 1.000 |
| 思考 tokens median／首答案時間 median | 324–367／8.5–8.6 秒 | **277–284／6.4–6.6 秒** |
| 總時間 median／p90 | 9.75–9.85／24.4–24.8 秒 | 6.9–7.2／16.9–17.0 秒 |
| 觸頂 1024 | 6 | 1 |

（s 類 10 題長文為人工評估題，兩者自動判定皆不計分；記憶體峰值兩者皆約 3.1GB。）

## 4. 發現
1. **風格目標達成**：思考關 0 英文回覆、0 簡體、亂碼題 10/10，g01 由英文改為繁中澄清；思考開亂碼 2/5→5/5；臺灣用語（臺灣、公尺等）經 s2twp 內化。思考關／開格式皆維持正確。
2. **知識與泛化退化**：PPL 上升 2.2×，知識題 12→10，且錯誤屬「自信的錯事實」（玉山、日月潭位置）。推測原因：(a) 2 epoch × lr 1e-4 × 205 層 rank 16 對 16.8 萬 token 過擬合（val loss 仍在降，但只衡量同分布）；(b) 教師答案本身含錯事實，學生學到「簡短篤定」語氣而非事實；(c) 無通用語料 replay／KL 約束。
3. **答案變短、延遲降低**：平均答案 179→104 字，總時間 median 2.2→0.8 秒；但 s 類長文仍會觸頂，部分短答（k03、k14）少了解釋。
4. **思考 trace 洩漏指引**：bench 樣本中思考開 trace 出現「I must be a Traditional Chinese assistant for Taiwan…」，是教師 trace 中提到 system 指引而被學到；思考仍以英文進行（答案為繁中）。
5. **工程**：MLX on-grid LoRA-QAT 與官方格點完全對齊，打包無損且大小不變；Metal 0000000e 以分塊 lm_head 根治。

## 5. 結論與建議
M8 **不建議取代官方 QAT 作為預設**：風格／亂碼處理明顯改善，但事實知識與 PPL 退化不可接受。建議下一步（M9 候選）：
- 降低強度：1 epoch、lr 3e-5～5e-5、rank 8，或只訓練 attention／後半層；保留較早 checkpoint（本次只保留最後 3 個）並以知識題＋PPL 選點。
- 加 KL-to-teacher（logit 蒸餾）或通用 zh-TW／英文語料 replay，限制 PPL 上升（目標 ≤ +10%）。
- 蒸餾資料加事實核對（知識題用維基事實比對）、移除思考 trace 中提及 system 指引的句子。
- 若只需風格修正，可先試 system prompt／M7 nothink-default 打包＋s2twp 後處理作為零訓練基線。
