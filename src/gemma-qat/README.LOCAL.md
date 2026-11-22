# src/gemma-qat（本專案對上游的 LoRA-QAT 改動）

此目錄是 [google-deepmind/gemma](https://github.com/google-deepmind/gemma) 的本地修改副本，不是官方分支。

- 基底上游仍為 google-deepmind/gemma
- 本副本的 HEAD：`60e24b1362a4ffdb2caf014f3f9cd138cef85f60`
- describe：`v4.0.1-43-g60e24b1`
- 本地分支名：`m5`
- 代表 commit 說明：`gemma4: mergeable LoRA-QAT (y = x·Q(W + s·AB), STE), split/merge helpers + tests`

授權沿用上游 Apache License 2.0（見本目錄 `LICENSE`）。

請把對訓練用 Gemma 程式庫的改動維持在這裡，不要改 `src/gemma` 的上游 checkout。
