# src/gemma（上游，不納入本倉庫）

本目錄在工作機上是 [google-deepmind/gemma](https://github.com/google-deepmind/gemma) 的完整 checkout，屬於上游程式，不在本公開倉庫重複放置。

本專案當時使用的 revision：

- 遠端：`https://github.com/google-deepmind/gemma.git`
- commit：`9e1f74131ba9d45e5fe6ae7bb3941d9f2abb39c3`
- describe：`v4.0.1-36-g9e1f741`
- 分支：`main`

請勿修改上游檔案。本專案對 Gemma 4 E2B 的可合併 LoRA-QAT 改動在 `../gemma-qat/`（見該目錄的 `README.LOCAL.md`）。

## 取得上游原始碼

在倉庫根目錄執行：

```bash
bash scripts/fetch_upstream_gemma.sh
```

或手動：

```bash
git clone https://github.com/google-deepmind/gemma.git src/gemma
cd src/gemma
git checkout 9e1f74131ba9d45e5fe6ae7bb3941d9f2abb39c3
```

授權：Apache License 2.0（見上游 `LICENSE`）。
