# Gemma 4 E2B QAT（gemma.cpp SBS）

Gemma 4 E2B 的量化感知微調（QAT / LoRA-QAT）與 [gemma.cpp](https://github.com/google/gemma.cpp) SBS 打包實驗。目標是在現有 `gemma_native` 0.2.1 上，提供可載入、可正確回答的傳統中文／英文模型檔。本倉庫**不使用** llama.cpp，也不產生或依賴 GGUF。

**English.** Research tree for Gemma 4 E2B QAT and LoRA-QAT aimed at gemma.cpp SBS (not llama.cpp / GGUF). Code, milestone tools, plans, and measured logs are here. Model weights and `.sbs` files are **not** in this repository (GitHub’s file size limit is 2 GB; the recommended artifact is 2,622,554,112 bytes). See the recommended-artifact section and `docs/logs/` for measured numbers.

## 建議產物（不包含在本倉庫）

在工作機上的建議檔是 M13 的 envelope 蒸餾結果：

| 項目 | 值 |
|---|---|
| 檔名 | `out/m13/gemma4-e2b-ft185-q4emb.sbs` |
| 大小 | **2,622,554,112 B**（剛好是不改 stock binary、不裁詞表時的上限） |
| sha256 | `3a22d31303b2117f99b6b2cd16813eff427e85f88f2ee2c67efb05bb59891050` |
| 執行檔 | 現有 `gemma_native` 0.2.1（不必換 binary） |
| 驗證 | exact-grid verify PASS；480 次原生推論完成；記憶體高峰約 1.50 GB |

GitHub 單檔上限為 2 GB（Release 附件與 LFS 同限），因此權重與 SBS **不會**上傳。請從本機 `out/m13/` 使用，或依下方工具路徑自行重建。對照用的 `out/m13/gemma4-e2b-ft100-q4emb.sbs` 大小相同，sha256 為 `c45b3946709df3a0dc87f7e33cf249874eb2e33e35f6adc6be224261c6056ecb`。

權重授權請遵守 [Gemma Terms of Use](https://ai.google.dev/gemma/terms)。

## 里程碑摘要（M8–M13，數字皆來自 `docs/logs` 實測）

細節與原始表格見 `docs/logs/` 與 `docs/Plan/`。以下不另推估。

### M8（2026-10-18）— 可合併 LoRA-QAT 與 zh-TW 蒸餾 SFT

- 日誌：[`docs/logs/2026-10-18_M8.md`](docs/logs/2026-10-18_M8.md)，工具：`docs/Plan/m8/tools/`
- MLX 上移植可合併 LoRA-QAT：`y = x·Q(W + s·AB)`，Q4_0 g32、STE；與 JAX minmax／block quant 逐 bit 一致。
- 匯出驗證無損，大小 **2.893 GB**。
- 思考關評測：遵從 50/50、知識 10/15、指令 15/15、亂碼 10/10、英文回覆 0、繁體 1.000。
- PPL 102.6／93.6（官方 QAT 47.2／40.3）。風格改善，但知識與 PPL 退化，不建議取代官方預設。

### M9（2026-10-18）— 資料清理與 KL-to-teacher

- 日誌：[`docs/logs/2026-10-18_M9.md`](docs/logs/2026-10-18_M9.md)，工具：`docs/Plan/m9/tools/`
- 目標：保留 M8 的 zh-TW 改善，PPL 增幅 ≤ 10%，知識 ≥ 官方 12/15。
- 選 step 100；打包 `out/m9/mlx-m9-g32exact` 無損（max_abs 0.0），**2,893,390,749 B**。
- 完整評測：PPL **47.19／40.34**（官方 47.17／40.27），知識 **12/15**，兩項目標達成。

### M10（2026-11-04）— 自建 compressed-tensors 與套件骨架

- 日誌：[`docs/logs/2026-11-04_M10.md`](docs/logs/2026-11-04_M10.md)
- 交付檔 `out/m10/m9-ct-g256`：**2,734,387,967 B（2.547 GiB）**，可被 transformers 載入。
- 同一套 M0 測資（wikitext 40,960／zh-TW 20,480 tokens）：m9ct PPL **47.01／40.67**，優於官方 mobile-ct **59.56／61.72**；對話知識 12/15、指令 15/15、亂碼 10/10。
- 新增 `src/g4qat` CLI／grid 測試骨架。重構階段 B／A 未開始。

### M11（2026-11-20）— 轉成 gemma.cpp SBS

- 日誌：[`docs/logs/2026-11-20_M11_sbs.md`](docs/logs/2026-11-20_M11_sbs.md)，工具：`docs/Plan/m11/tools/`（含 `sbs_io.py`、`m11_exact.py`）
- 轉換器可重現原網站 SBS（sha256 相同）。M9／M8 的 Q4_0 block 與 MLX 逐 bit 一致，verify PASS。
- M9／M8 SBS 皆為 **3,221,159,936 B**。
- 網站 profile、50 題（知識／指令／亂問題）：原檔 12／14／7，M9 11／15／10，M8 12／15／10。M9 在 native 將「國父」答成蔣介石（與 MLX 不同，屬運算數值路徑，權重已驗證相同）。

### M12（2026-11-20）— 嘗試壓到 2 GB 以下

- 日誌：[`docs/logs/2026-11-20_M12_sbs_2gb.md`](docs/logs/2026-11-20_M12_sbs_2gb.md)，工具：`docs/Plan/m12/tools/`；binary 差異：`docs/Plan/m12/raw/gemma_native_m12.diff`
- stock `gemma_native` 將詞表寫死為 262,144，在不改執行檔時最小為候選 A：**2,622,554,112 B**。
- 裁詞表到 154,624 並重編 binary 後，M8 版 **1,984,823,296 B（1.849 GiB）**，sha256 `e2b69b0a26c9e7575ddb8bafb4a4dc8e9c9d33ff9f140aaa63f6fe9962d07d4e`；事實題 19/22，國父可答對，但**必須**搭配自建 binary。Linux 版當時未實測。

### M13（2026-11-21）— 不改 binary，提升中英回答品質

- 日誌：[`docs/logs/2026-11-21_M13_quality.md`](docs/logs/2026-11-21_M13_quality.md)，工具：`docs/Plan/m13/tools/`
- 約束：沿用 stock binary、不裁詞表、大小 ≤ 2,622,554,112 B。
- 網站實際送的是 JSON envelope。在 M8 上做 envelope 蒸餾 LoRA-QAT（rank 8，lr 4e-5，185 steps）。建議 **ft185**（見上節）。

120 題（網站 profile；zh／60、en／60）與 50 題（知識／指令／亂問題）：

| 模型 | 裸問題 zh／en | 網站 envelope zh／en | 裸 p50 | envelope p50 |
|---|---|---|---|---|
| 原檔 | 48／59 | 37／55 | 12／14／7 | 10／14／8 |
| A-M8 | 49／60 | 38／56 | 10／15／10 | 7／14／9 |
| A-M9 | 48／60 | 36／57 | 11／14／9 | （未列於 ft 對照表） |
| ft100 | 48／60 | 45／55 | 10／15／10 | 11／14／9 |
| **ft185** | 46／59 | **48／56** | 9／15／10 | **11／15／10** |

envelope 下中文回答長度約從 A-M8 的 24 字回到約 105–108 字。英文在 envelope 下仍約 55–56/60，沒有提升。國父題不穩：ft185 只在 envelope 下答對、裸問題答成蔣介石；ft100 相反。濁水溪、左營、澎湖、孟浩然等題各版皆錯，屬 2B 模型知識上限。

## 如何重現

工作機是 Apple Silicon Mac（本專案在 M1 Max 64GB 上跑完）。長工作請寫 STATUS、顯示進度。規則：不用 llama.cpp／GGUF；不要改 `src/gemma`、`weights/`，也不要改網站套件 `Gemma4-Portable`。

1. 需要官方權重時，請依 Gemma Terms 自行取得（例如 `google/gemma-4-E2B-it-qat-q4_0-unquantized`），放到本機 `weights/`（已 gitignore）。
2. 上游 Gemma 函式庫：`bash scripts/fetch_upstream_gemma.sh`（commit `9e1f74131ba9d45e5fe6ae7bb3941d9f2abb39c3`）。LoRA-QAT 改動用 `src/gemma-qat/`。
3. 早期 MLX／HF 腳本在倉庫根目錄，說明見 [`docs/README_initial_mlx.md`](docs/README_initial_mlx.md)。
4. 里程碑腳本：
   - M8／M9 訓練與匯出：`docs/Plan/m8/tools/`、`docs/Plan/m9/tools/`
   - compressed-tensors：`docs/Plan/m10/tools/`、`src/g4qat/`
   - SBS 轉換與 exact verify：`docs/Plan/m11/tools/`（`m11_convert.sh`、`m11_exact.py`、`sbs_io.py`）
   - 2.62 GB／裁詞表實驗：`docs/Plan/m12/tools/`
   - 品質微調與原生評測：`docs/Plan/m13/tools/`（`m13_pipeline.sh`、`m13_train.py`、`m13_build.sh`、`m13_native_eval.py`）
5. 網站格式評測用的 envelope 檔**不在**本倉庫（見 `docs/Plan/m13/raw/ENVELOPES_REDACTED.md`）。若有網站 scripts 的本機路徑，可用 `m13_envelopes.mjs` 自行產生。
6. `gemma_native` 0.2.1 與網站 runtime 不在本倉庫；請使用既有的 Gemma4-Portable 安裝。

## 目錄配置

```
.
├── README.md                 # 本文件
├── docs/
│   ├── logs/                 # 工作紀錄與索引（必讀）
│   ├── Plan/                 # 各里程碑計畫、工具、raw 評測
│   │   ├── m11/tools/        # SBS I/O 與 exact patch/verify
│   │   ├── m12/raw/gemma_native_m12.diff
│   │   └── m13/tools/        # 品質微調與評測
│   └── README_initial_mlx.md # 專案初期的 MLX／HF 路線說明
├── src/
│   ├── g4qat/                # 套件骨架（CLI、grid、ct export）
│   ├── gemma/                # 上游指針；執行 scripts/fetch_upstream_gemma.sh
│   └── gemma-qat/            # 本專案的可合併 LoRA-QAT 改動
├── tools/                    # HF↔JAX 轉換、exact-grid 打包
├── qat_from_source/          # 早期 JAX QAT 實驗
├── scripts/fetch_upstream_gemma.sh
├── sample.jsonl
└── STATUS.txt                # 本次公開整理的進度（僅出現在整理用副本）
```

本機工作樹另外還有（皆不上傳）：`weights/`、`out/`（含建議 SBS）、虛擬環境。

## 授權

- **模型權重**：須遵守 [Gemma Terms of Use](https://ai.google.dev/gemma/terms)，本倉庫不附帶權重。
- **`src/gemma-qat/`**：修改自 google-deepmind/gemma，Apache License 2.0（見該目錄 `LICENSE`）。
- **本倉庫其餘自行撰寫的程式與文件**：尚未選定授權。請倉庫擁有者決定後再補 `LICENSE`。

## 公開範圍與安全

上傳前已排除權重、SBS、虛擬環境、快取、大於 50 MB 的檔案，以及網站政策 envelope 正文。詳見 [`docs/logs/2026-11-22_github_publish.md`](docs/logs/2026-11-22_github_publish.md)。文件與腳本中的本機絕對路徑已改寫為 `~/`；指向網站專案的路徑改為 `$WEBSITE_ROOT`（重現時請設成自己的網站專案路徑），網站政策與對話格式的內部名稱改為 `site-policy`、`site-chat-v1`。

## 聯絡 / Contact

@LexChien - Lex Chien, XsOin Inc.
