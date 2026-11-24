# g4qat

Gemma 4 E2B QAT／LoRA-QAT／匯出工具包。這是專案重構（`docs/Plan/10_refactor_plan.md`）的套件根目錄。**不要改 `src/gemma` 或 `weights/`。** 舊腳本（`docs/Plan/m*/tools`、`tools/`、根目錄 `*.py`）仍可照舊呼叫。

## 安裝（專案的 `.venv-m0`）

```bash
cd ~/Documents/AI/gemma4-qat
.venv-m0/bin/pip install -e src/g4qat
```

## CLI

| 指令 | 作用 | 舊入口（仍可用） |
|---|---|---|
| `g4qat export-ct SRC OUT [--ple-group 256]` | MLX exact-grid → compressed-tensors | `python -m g4qat.export_ct` |
| `g4qat eval-hf MODEL TAG OUTDIR` | transformers 載入／PPL／50 題 | `python -m g4qat.hf_eval` |
| `g4qat eval-ppl NAME MODEL_DIR` | M0 MLX PPL＋速度 | `docs/Plan/m0/tools/m0_bench_mlx.py` |
| `g4qat eval-chat MODEL TAG …` | M7 延遲／對話 | `docs/Plan/m7/tools/m7_latency.py` |
| `g4qat grade TAG…` | M7 評分 | `docs/Plan/m7/tools/m7_grade.py` |
| `g4qat bench NAME MODEL_DIR` | `eval-ppl` 的別名 | 同上 |
| `g4qat version` | 套件版本 | — |

C-1 階段已就位：套件骨架、`grid.py`、`export/`、`eval/`、CLI。尚未實作（B 以後）：自寫 MLX 模型、訓練迴圈、JAX 重構。

## 模組

- `g4qat.grid`：Q4_0 g32 pack／unpack、固定格點 levels、STE 用的量化
- `g4qat.export.ct`：compressed-tensors 匯出
- `g4qat.eval.hf`：HF 評測
- `g4qat.ct_compat`：transformers 5.20.1 量化 Embedding 初始化修正
- `g4qat.mlx`／`g4qat.jax`／`g4qat.data`／`g4qat.io`：占位（B／A／後續）
