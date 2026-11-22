"""M0: build fixed eval token sets (shared tokenizer of gemma-4-E2B-it; verified identical across variants).
Outputs docs/Plan/m0/raw/eval_tokens.json  {wikitext: [...], zh_tw: [...]} + source texts."""
import json, urllib.request, urllib.parse, hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
R = "docs/Plan/m0/raw/"
tok = AutoTokenizer.from_pretrained("weights/gemma-4-E2B-it")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
wt = "\n\n".join(ds["text"])
wt_ids = tok(wt, add_special_tokens=False)["input_ids"][:40960]
titles = ["臺灣", "臺北市", "玉山", "日月潭", "台灣積體電路製造", "國立故宮博物院", "臺灣高速鐵路", "阿里山"]
zh = []
for t in titles:
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "format": "json",
                                "titles": t, "variant": "zh-tw", "redirects": 1})
    req = urllib.request.Request("https://zh.wikipedia.org/w/api.php?" + q, headers={"User-Agent": "gemma4-qat-M0/0.1 (research)"})
    pages = json.load(urllib.request.urlopen(req, timeout=60))["query"]["pages"]
    for p in pages.values(): zh.append(p.get("extract", ""))
zht = "\n\n".join(zh)
open(R + "zh_tw_source.txt", "w").write(zht)
zh_ids = tok(zht, add_special_tokens=False)["input_ids"][:20480]
json.dump({"tokenizer": "weights/gemma-4-E2B-it", "bos_id": tok.bos_token_id,
           "wikitext": wt_ids, "zh_tw": zh_ids,
           "notes": "wikitext-2-raw-v1 test joined by \\n\\n, first 40960 tokens; zh-tw Wikipedia extracts (variant=zh-tw) of %s, first 20480 tokens" % titles},
          open(R + "eval_tokens.json", "w"))
print("wikitext tokens", len(wt_ids), "zh tokens", len(zh_ids), "zh chars", len(zht), "bos", tok.bos_token_id)
print("sha1 zh", hashlib.sha1(zht.encode()).hexdigest())
print(zht[:200])
