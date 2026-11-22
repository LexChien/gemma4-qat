"""M5 training data (held-out from M0 eval):
  * en: wikitext-2-raw-v1 TRAIN (M0 eval = test split), CC BY-SA 3.0
  * zh-TW: HF dataset zetavg/zh-tw-wikipedia (zh-TW Wikipedia, markdown), CC BY-SA 3.0/4.0; streaming shuffle (seed 0);
    the 8 M0 eval titles (and variants) are excluded.
Outputs docs/Plan/m5/raw/: train_tokens.npy (BOS-separated docs, en/zh interleaved), mlx_data/{train,valid}.jsonl
(chunks of ~CHUNK tokens as {"text"}), data_meta.json"""
import json, os, re, random, numpy as np
from datasets import load_dataset
import transformers
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")); R = ROOT + "/docs/Plan/m5/raw/"
ZH_CHARS = int(os.environ.get("ZH_CHARS", 3_000_000)); CHUNK = int(os.environ.get("CHUNK", 512))
EVAL = ["臺灣", "臺北市", "玉山", "日月潭", "台灣積體電路製造", "國立故宮博物院", "臺灣高速鐵路", "阿里山", "台灣", "台北市", "台積電", "臺灣積體電路製造"]
tok = transformers.AutoTokenizer.from_pretrained(ROOT + "/weights/gemma-4-E2B-it")
def clean(md):
    md = re.sub(r"\*\*|__", "", md); md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)
    md = re.sub(r"^#+\s*", "", md, flags=re.M); return md.strip()
zh, n_chars, n_skip = [], 0, 0
for r in load_dataset("zetavg/zh-tw-wikipedia", split="train", streaming=True).shuffle(seed=0, buffer_size=20000):
    t = r["original_title"] or ""
    if t in EVAL: n_skip += 1; continue
    x = clean(r["markdown"] or "")
    if len(x) < 1500: continue
    zh.append(x); n_chars += len(x)
    if n_chars >= ZH_CHARS: break
wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]
en, cur = [], []
for line in wt:  # split into articles at " = Title = " headings
    if re.match(r"^ = [^=].* = \n$", line) and cur: en.append("".join(cur)); cur = []
    cur.append(line)
en.append("".join(cur))
random.seed(0); docs = [("zh", d) for d in zh] + [("en", d) for d in en]; random.shuffle(docs)
ids, chunks = [], []
for lang, d in docs:
    t = tok(d, add_special_tokens=False)["input_ids"]
    ids += [tok.bos_token_id] + t
    for i in range(0, len(t), CHUNK): chunks.append(tok.decode(t[i:i + CHUNK]))
ids = np.array(ids, np.int32); np.save(R + "train_tokens.npy", ids)
os.makedirs(R + "mlx_data", exist_ok=True); random.shuffle(chunks)
nv = 64
with open(R + "mlx_data/valid.jsonl", "w") as f:
    for c in chunks[:nv]: f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
with open(R + "mlx_data/train.jsonl", "w") as f:
    for c in chunks[nv:]: f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
zh_tok = sum(len(tok(d, add_special_tokens=False)["input_ids"]) for d in zh[:50]) / max(1, sum(len(d) for d in zh[:50]))
meta = {"zh_source": "https://huggingface.co/datasets/zetavg/zh-tw-wikipedia (zh-TW Wikipedia, markdown->text)",
        "zh_license": "CC BY-SA (Wikipedia text; dataset card: cc-by-sa-3.0)", "en_source": "Salesforce/wikitext wikitext-2-raw-v1 train",
        "en_license": "CC BY-SA 3.0", "zh_docs": len(zh), "zh_chars": n_chars, "zh_eval_titles_skipped": n_skip,
        "en_docs": len(en), "total_tokens": int(len(ids)), "zh_tokens_per_char_est": zh_tok, "mlx_chunks_train": len(chunks) - nv,
        "mlx_chunks_valid": nv, "chunk_tokens": CHUNK, "eval_heldout": "M0 eval_tokens.json = wikitext TEST + 8 zh-TW Wikipedia articles (excluded)"}
json.dump(meta, open(R + "data_meta.json", "w"), ensure_ascii=False, indent=1); print(meta)
