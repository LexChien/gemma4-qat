"""M9 checkpoint selection: quick PPL (first 10,240 tokens of each M0 PPL set) + thinking-off 50-prompt grade per candidate.
Feasible = PPL ratio <= 1.10 (both sets, vs official at the same limit) and knowledge >= 12/15 and 0 English replies and 50/50 compliance.
Pick the feasible one with the best (garbage + instruction + knowledge - answers_with_simplified), tie -> smaller worst PPL ratio.
If none is feasible: fewest violations, then the same score, then smaller PPL ratio. -> raw/select.json"""
import json, glob, os
R = "docs/Plan/m9/raw/"; C = R + "cand/"
o = json.load(open(R + "bench_m9-qat_q.json")); wt0, zh0 = o["ppl_wikitext"], o["ppl_zh_tw"]
rows = []
for f in sorted(glob.glob(R + "bench_m9c-*_q.json")):
    s = f.split("m9c-")[1][:6]; b = json.load(open(f)); g = json.load(open(C + f"chat_c{s}_off_summary.json"))
    n = lambda k: int(g[k].split("/")[0])
    r = dict(step=int(s), ppl_wt=round(b["ppl_wikitext"], 2), ppl_zh=round(b["ppl_zh_tw"], 2), r_wt=round(b["ppl_wikitext"] / wt0, 3), r_zh=round(b["ppl_zh_tw"] / zh0, 3),
             know=n("knowledge_pass"), inst=n("instruction_pass"), garb=n("garbage_pass"), eng=g["n_english_reply"], simp=g["answers_with_any_simplified"],
             comp=n("thinking_off_compliance"), trad=g["trad_ratio"], fails=[p["id"] for p in g["per_prompt"] if not p["ok"] and not p["id"].startswith("s")])
    r["viol"] = [k for k, bad in [("ppl", max(r["r_wt"], r["r_zh"]) > 1.10), ("know", r["know"] < 12), ("eng", r["eng"] > 0), ("comp", r["comp"] < 50)] if bad]
    r["score"] = r["garb"] + r["inst"] + r["know"] - r["simp"]; rows.append(r)
best = min(rows, key=lambda r: (len(r["viol"]), -r["score"], max(r["r_wt"], r["r_zh"])))
out = dict(official_quick=dict(ppl_wt=round(wt0, 2), ppl_zh=round(zh0, 2)), candidates=rows, chosen=best["step"], chosen_feasible=not best["viol"])
json.dump(out, open(R + "select.json", "w"), ensure_ascii=False, indent=1)
for r in rows: print(r)
print(f"CHOSEN {best['step']:06d} feasible={not best['viol']}")
