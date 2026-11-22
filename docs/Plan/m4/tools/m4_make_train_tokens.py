"""Tokenize wikitext-2-raw-v1 TRAIN split (HF cache) with the gemma-4-E2B-it tokenizer -> m4/raw/train_tokens.npy"""
import numpy as np, datasets, transformers, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
tok = transformers.AutoTokenizer.from_pretrained(ROOT + "/weights/gemma-4-E2B-it")
text = "".join(ds["text"][:4000])
ids = tok(text, add_special_tokens=False)["input_ids"]
np.save(ROOT + "/docs/Plan/m4/raw/train_tokens.npy", np.array(ids, np.int32))
print(len(ids), "tokens; bos", tok.bos_token_id)
