"""Build markdown tables for the M6 report from raw/*.json -> raw/tables.md"""
import json, os, glob
R = "docs/Plan/m6/raw/"; J = lambda p: json.load(open(p)) if os.path.exists(p) else None
o = []
o.append("### PPL / 速度（M0 方法）\n\n| 模型 | 參數 GB | PPL wikitext | PPL zh-TW | 生成 tok/s（run1/run2） | prefill tok/s | 峰值 GB |\n|---|---|---|---|---|---|---|")
rows = [("官方 QAT g32 exact（M5 量測）", "docs/Plan/m5/raw/bench_m5-qat-g32exact.json", "m6-qat-g32exact"),
        ("M5 JAX LoRA-QAT g32 exact", "docs/Plan/m5/raw/bench_m5-jaxlora-g32exact.json", "m6-jaxlora-g32exact"),
        ("M6 MLX LoRA → snap g32 exact", R + "bench_m6-mlxlora-g32exact.json", "m6-mlxlora-g32exact"),
        ("M6 MLX LoRA → g64 affine", R + "bench_m6-mlxlora-g64.json", None),
        ("M6 MLX LoRA fused bf16（參考）", R + "bench_m6-mlxlora-fused-bf16.json", None)]
for name, p, sp in rows:
    d = J(p); s1 = J(R + f"bench_{sp}_speed1.json") if sp else None; s2 = J(R + f"bench_{sp}_speed2.json") if sp else None
    sp_s = f"{d['gen_tps']:.1f}" + (f" → {s1['gen_tps']:.1f} / {s2['gen_tps']:.1f}" if s1 and s2 else "")
    o.append(f"| {name} | {d['n_params_bytes_GB']:.2f} | {d['ppl_wikitext']:.2f} | {d['ppl_zh_tw']:.2f} | {sp_s} | {d['prompt_tps']:.0f} | {d['gen_peak_mem_GB']:.2f} |")
o.append("\n（第一個 tok/s 為含 PPL 的 run，箭頭後為本次 M6 兩次純速度 run；M6 的速度 run 期間 Chrome Helper 佔 ~100% CPU/GPU，數字約為 M5 的一半，只能互相比較。）\n")
o.append("### 中文對話品質（50 題，greedy）\n\n| 模型 / 模式 | 知識 | 指令 | 垃圾/不可答 | 繁體字比例 | 含簡體字答案數 | 平均 rep4 | rep4>0.3 | 平均字數 | 撞上限 |\n|---|---|---|---|---|---|---|---|---|---|")
for t, n in [("qat_off", "官方 QAT · 思考關"), ("jaxlora_off", "M5 JAX LoRA · 思考關"), ("mlxlora_off", "M6 MLX LoRA · 思考關"),
             ("qat_on", "官方 QAT · 思考開（25 題）"), ("jaxlora_on", "M5 · 思考開（25 題）"), ("mlxlora_on", "M6 · 思考開（25 題）"),
             ("lever_budget128_on", "官方 · 思考上限 128（25 題）"), ("lever_budget256_on", "官方 · 思考上限 256（25 題）"), ("lever_kv4_off", "官方 · KV 4-bit · 思考關")]:
    d = J(R + f"chat_{t}_summary.json")
    if d: o.append(f"| {n} | {d['knowledge_pass']} | {d['instruction_pass']} | {d['garbage_pass']} | {d['trad_ratio']:.3f} | {d['answers_with_any_simplified']} | {d['mean_rep4']:.3f} | {d['n_rep4_gt_0_3']} | {d['mean_chars']:.0f} | {d['n_hit_max']} |")
o.append("\n### 端到端延遲（warm，每格 median / P90；秒）\n\n| 設定 | run | prefill tok/s | TTFT | 思考秒數 | 思考 tokens | 首個答案 token | 答案秒數 | decode tok/s | 總時間 | cold 首題 TTFT/TTFA/總 |\n|---|---|---|---|---|---|---|---|---|---|---|")
for t in ["qat_off", "qat_on", "jaxlora_off", "jaxlora_on", "mlxlora_off", "mlxlora_on", "lever_kv8_off", "lever_kv4_off", "lever_g64_off", "lever_budget128_on", "lever_budget256_on"]:
    d = J(R + f"lat_{t}_summary.json")
    if not d: continue
    for run, s in d["runs"].items():
        f = lambda k: f"{s[k]['median']} / {s[k]['p90']}"
        c = s.get("cold_first_prompt"); cs = f"{c['ttft_s']} / {c['ttfa_s']} / {c['total_s']}" if c else ""
        o.append(f"| {t} | {run} | {f('prefill_tps')} | {f('ttft_s')} | {f('think_s')} | {f('think_tokens')} | {f('ttfa_s')} | {f('answer_s')} | {f('decode_tps')} | {f('total_s')} | {cs} |")
pc = J(R + "lever_prompt_cache_qat.json")
if pc: o.append(f"\n### Prompt/KV cache 重用\n\n```\n{json.dumps(pc, ensure_ascii=False, indent=1)}\n```")
open(R + "tables.md", "w").write("\n".join(o) + "\n"); print("\n".join(o))
