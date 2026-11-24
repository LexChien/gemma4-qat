# 未納入本倉庫的檔案

以下兩個檔案只存在於工作機，**不會**放進公開倉庫，因為它們含有網站對話 envelope 的完整政策原文（policy `site-policy`）：

- `envelopes.json`
- `envelopes_txt.json`

若要在本機重跑 M13 的網站格式評測，請用 `docs/Plan/m13/tools/m13_envelopes.mjs` 從網站專案既有的唯讀 helper 產生（腳本裡的絕對路徑指向本機的 XsOin Inc. 網站 scripts，不在本倉庫）。`m13_env_text.py` 則把已產生的 JSON envelope 轉成純文字對照組。

本目錄其餘 raw 評測紀錄只保留模型輸出與分數，不含上述政策正文。
