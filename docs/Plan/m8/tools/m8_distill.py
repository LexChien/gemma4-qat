"""M8 distillation: official QAT g32 exact as teacher, THINKING ON + zh-TW guidance system prompt (context distillation;
the student is trained without the system prompt). Resumable: appends to raw/distill.jsonl, skips done ids.
Usage: m8_distill.py [--limit N] [--batch B]"""
import argparse, json, os, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import batch_generate
R = "docs/Plan/m8/raw/"
GUIDE = ("你是臺灣的繁體中文助理。請一律使用繁體中文與臺灣慣用語回答（例如：軟體、網路、資訊、影片、品質、程式）。"
         "回答要正確、精簡、直接，並嚴格遵守使用者指定的格式、項目數與字數。"
         "如果輸入是亂碼或沒有意義，或問題無法回答（例如未來的事件、即時資料、他人隱私、你不可能知道的個人事情、前提錯誤的問題），"
         "請用繁體中文簡短說明原因，並請使用者補充或澄清，不要編造答案。遇到違法或危險的要求，請婉拒並提供安全的替代建議。")
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--max-tokens", type=int, default=1400); a = ap.parse_args()
P = [json.loads(l) for l in open(R + "sft_prompts.jsonl")]
P = [p for p in P if p["target"] is None]
done = set()
if os.path.exists(R + "distill.jsonl"): done = {json.loads(l)["id"] for l in open(R + "distill.jsonl")}
todo = [p for p in P if p["id"] not in done][: a.limit]
print(f"prompts {len(P)} done {len(done)} todo {len(todo)}", flush=True)
model, tok = load("out/m5/mlx-qat-g32exact"); t0 = time.time(); ntok = 0
with open(R + "distill.jsonl", "a") as f:
    for i in range(0, len(todo), a.batch):
        ch = todo[i:i + a.batch]
        ids = [tok.apply_chat_template([{"role": "system", "content": GUIDE}, {"role": "user", "content": p["prompt"]}],
                                       add_generation_prompt=True, enable_thinking=True) for p in ch]
        r = batch_generate(model, tok, ids, max_tokens=a.max_tokens, return_token_ids=True)
        for p, toks in zip(ch, r.token_ids if hasattr(r, "token_ids") else r.tokens):
            toks = [int(t) for t in toks]; ntok += len(toks)
            f.write(json.dumps(dict(id=p["id"], cat=p["cat"], think=p["think"], prompt=p["prompt"], gen_ids=toks,
                                    gen_text=tok.decode(toks), n=len(toks), hit_max=len(toks) >= a.max_tokens), ensure_ascii=False) + "\n")
        f.flush(); el = time.time() - t0
        print(f"distill {min(i + a.batch, len(todo))}/{len(todo)} gen_tok {ntok} {ntok / el:.0f} tok/s elapsed {el / 60:.1f} min", flush=True)
print("DISTILL_DONE", flush=True)
