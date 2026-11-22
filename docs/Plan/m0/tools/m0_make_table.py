"""M0: build results table (markdown + json) from raw/bench_*.json and raw/sizes.json."""
import json, os, statistics as st
R = "docs/Plan/m0/raw/"; S = json.load(open(R + "sizes.json"))
J = lambda n: json.load(open(R + f"bench_{n}.json")) if os.path.exists(R + f"bench_{n}.json") else None
rows = [  # name, bench, size key, reference bench for PPL delta, note
 ("E2B-it bf16（HF 原檔）", "mlx-it-bf16", "hf-it-bf16", None, "含 vision/audio；MLX 只載文字"),
 ("QAT-q4_0-unquantized bf16（HF 原檔）", "mlx-qat-bf16", "hf-qat-q4_0-unquantized", None, "QAT 基準"),
 ("MLX it 4-bit g64", "mlx-it-q4g64", "mlx-it-q4g64", "mlx-it-bf16", "全部 4-bit（含 embed/PLE），純文字"),
 ("MLX it 4-bit g32", "mlx-it-q4g32", "mlx-it-q4g32", "mlx-it-bf16", "同上"),
 ("MLX QAT 4-bit g64", "mlx-qat-q4g64", "mlx-qat-q4g64", "mlx-qat-bf16", "= quantize_mlx_mixed safe（sha1 相同）"),
 ("MLX QAT 4-bit g32", "mlx-qat-q4g32", "mlx-qat-q4g32", "mlx-qat-bf16", ""),
 ("MLX QAT mobile 混合 2/4/8-bit", "mlx-qat-mobile-g64", "mlx-qat-mobile-g64", "mlx-qat-bf16", "embed 2-bit、MLP≥15 2-bit、PLE 4-bit g128"),
 ("官方 qat-w4a16-ct", "mlx-ct-w4a16-dequant-bf16", "official-w4a16-ct", "mlx-qat-bf16", "反量化後以 MLX bf16 評測（weight-only）"),
 ("官方 qat-mobile-ct", "mlx-ct-mobile-dequant-bf16", "official-mobile-ct", "mlx-qat-bf16", "反量化後評測；忽略 int8 activation"),
 ("E2B pt bf16（參考）", "mlx-pt-bf16", None, None, "基礎預訓練版，僅作 PPL 參考"),
]
out = []; md = ["| 版本 | 磁碟大小 GB (GiB) | <3GB | PPL wikitext | Δ | PPL 繁中 | Δ | 生成 tok/s | 載入 s | 生成峰值記憶體 GB | PPL 峰值記憶體 GB | 備註 |", "|" + "---|" * 12]
for lab, b, sk, ref, note in rows:
    d = J(b)
    if d is None: continue
    sp = [J(b + f"_speed{i}") for i in (1, 2)]; sp = [x for x in sp if x]
    tps = st.mean([x["gen_tps"] for x in sp]) if sp else d["gen_tps"]
    load = st.mean([x["load_s"] for x in sp]) if sp else d["load_s"]
    gmem = max([x["gen_peak_mem_GB"] for x in sp] + [d["gen_peak_mem_GB"]])
    size = S[sk] if sk else None
    r = {"label": lab, "bench": b, "size_GB": size and size["GB"], "size_GiB": size and size["GiB"],
         "ppl_wt": d["ppl_wikitext"], "ppl_zh": d["ppl_zh_tw"], "tps": tps, "tps_runs": [x["gen_tps"] for x in sp], "load_s": load,
         "gen_mem_GB": gmem, "ppl_mem_GB": d["ppl_peak_mem_GB"], "note": note}
    if ref:
        rr = J(ref); r["d_wt"] = d["ppl_wikitext"] / rr["ppl_wikitext"] - 1; r["d_zh"] = d["ppl_zh_tw"] / rr["ppl_zh_tw"] - 1
    out.append(r)
    f = lambda x: f"{x*100:+.1f}%" if x is not None else "—"
    sz = f"{r['size_GB']:.3f} ({r['size_GiB']:.2f})" if size else "—"
    ok = ("✅" if r["size_GB"] < 3 else "❌") if size else "—"
    perf = f"{tps:.1f} | {load:.2f} | {gmem:.2f} | {r['ppl_mem_GB']:.2f}" if not b.startswith("mlx-ct-") else "— | — | — | —"
    md.append(f"| {lab} | {sz} | {ok} | {r['ppl_wt']:.2f} | {f(r.get('d_wt'))} | {r['ppl_zh']:.2f} | {f(r.get('d_zh'))} | {perf} | {note} |")
json.dump(out, open(R + "results_table.json", "w"), ensure_ascii=False, indent=1)
open(R + "results_table.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
