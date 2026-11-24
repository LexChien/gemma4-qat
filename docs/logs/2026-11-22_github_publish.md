# 2026-11-22 GitHub 公開整理（UTC+8）

依據：Lex 要求「整理一份，github 建立專案，上傳」，並選定公開（public）。倉庫名稱預定 `LexChien/gemma4-qat`。

## 做法

- 不直接改工作專案成 git 倉庫，以免動到進行中的 `out/`、`weights/`。
- 以 rsync 在 `~/Documents/AI/gemma4-qat-publish` 做一份乾淨副本，在該處初始化 git 並上傳。
- 原專案只新增本篇 log，並更新 `docs/logs/README.md` 索引。

## 排除（不上傳）

- `weights/`、`out/`（工作機上合計約 414 GB）
- 所有 `*.sbs`、`*.safetensors`、`*.bin`、`*.gguf`、`*.pt` 與檢查點
- `.venv`、`.venv-jax`、`.venv-m0`、`__pycache__`、`.pytest_cache`、`*.egg-info`、`.DS_Store`
- 任何大於 50 MB 的檔案（本次符合條件的皆已在上述目錄內）
- 上游 `src/gemma/` 完整 checkout（改以 `src/gemma/README.md`＋`scripts/fetch_upstream_gemma.sh` 標明 `https://github.com/google-deepmind/gemma.git` @ `9e1f74131ba9d45e5fe6ae7bb3941d9f2abb39c3`）
- 網站政策 envelope 正文：`docs/Plan/m13/raw/envelopes.json`、`docs/Plan/m13/raw/envelopes_txt.json`（policy `site-policy` 完整指示）。說明改放在 `docs/Plan/m13/raw/ENVELOPES_REDACTED.md`。這兩檔也已寫入 `.gitignore`。

保留：程式與工具、`docs/logs`、`docs/Plan`（含 `docs/Plan/m12/raw/gemma_native_m12.diff`）、`src/g4qat`、`src/gemma-qat`（本專案 LoRA-QAT 改動，不含其巢狀 `.git`）。

建議產物 `out/m13/gemma4-e2b-ft185-q4emb.sbs`（2,622,554,112 B，sha256 `3a22d31303b2117f99b6b2cd16813eff427e85f88f2ee2c67efb05bb59891050`）因超過 GitHub 單檔 2 GB 上限而未納入；README 只記錄大小、雜湊與重建路徑。

## 安全掃描

- 本機未安裝 `gitleaks`／`trufflehog`，以 ripgrep 與腳本掃描工作樹（排除 venv／weights／out／.git）。
- 未發現 `hf_`、`ghp_`、`github_pat_`、`sk-`、`AKIA`、AWS secret、Slack token、`BEGIN PRIVATE KEY`、`.env` 或 API key 指派。
- 完整網站政策正文僅出現在上述兩個被排除的 envelope 檔，以及本機 `out/m13/work/ft/env_out.json`（本就不會上傳）。工具與 M13 日誌只提到政策版本名稱與產生方式，不含政策正文。
- 電子郵件：`docs/Plan/m9/tools/m9_factcheck.py` 的維基 API User-Agent 含 `github.com/LexChien`（公開聯絡用，已保留並在此註記）。訓練／維基快取中另有 `user@domain.com`、`bob@b.org` 與維基條目裡的 `kiritofelay@gmail.com`，非客戶資料。
- 看似電話或內網 IP 的字串經核對為量化誤差浮點（例如 `0.0927734375`）以及維基文中的瀏覽器版本號，屬誤報。
- 部分腳本含本機絕對路徑（`~/Documents/...`），已註記，未當成密鑰。

## 授權

本倉庫頂層未放程式碼授權檔，需由 Lex 決定（例如 Apache-2.0、MIT，或先維持未聲明）。`src/gemma-qat` 沿用上游 Apache-2.0。權重遵守 Gemma Terms of Use。

## 上傳紀錄

- 本機 `gh` 未安裝；GitHub SSH（`git@github.com`）已能以 `LexChien` 驗證。建立空倉庫改用已登入的 GitHub connector（`create_repository`，`private=false`），再用 git SSH 推送。
- 公開倉庫：https://github.com/LexChien/gemma4-qat （visibility=public）
- 分支：`main`
- 初始 commit：`d825cc278d71b923c100e27bbf3f8286725f12a0`（2026-11-22 20:19:32 UTC+8）
  - 1038 個檔案，234,609 行新增
  - 工作樹（不含 `.git`）約 67.22 MiB；git pack 約 21.55 MiB
- 作者：@LexChien - Lex Chien, XsOin Inc.
