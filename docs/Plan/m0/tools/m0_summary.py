import json, glob
for f in sorted(glob.glob("docs/Plan/m0/raw/bench_*.json")):
    d = json.load(open(f))
    print(f.split("bench_")[1][:-5].ljust(34), "wt", round(d.get("ppl_wikitext", 0), 2), "zh", round(d.get("ppl_zh_tw", 0), 2),
          "tps", round(d.get("gen_tps", d.get("gen_tps_incl_prefill", 0)), 1), "load", round(d.get("load_s", 0), 2),
          "genmem", round(d.get("gen_peak_mem_GB", 0), 2), "pplmem", round(d.get("ppl_peak_mem_GB", 0), 2), "rss", round(d.get("max_rss_GB", 0), 2))
