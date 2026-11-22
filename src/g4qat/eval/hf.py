"""Evaluate a compressed-tensors (or plain HF) Gemma 4 checkpoint with transformers: load time, peak RSS, PPL on the M0 sets
(ctx 1024 / stride 512, same as m0_bench_mlx.py) and the 50 zh-TW prompts (thinking off, greedy, 256 tokens) -> m7_grade format.
Usage: g4qat eval-hf MODEL_DIR TAG OUTDIR [--device mps|cpu] [--no-ppl] [--no-chat] [--limit N]
       python -m g4qat.hf_eval MODEL_DIR TAG OUTDIR [...]"""
import argparse, json, math, os, resource, sys, time
import torch
import g4qat.ct_compat  # noqa: F401  (transformers 5.20.1 quantized-Embedding init fix)
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig
def rss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("outdir")
    ap.add_argument("--device", default="mps"); ap.add_argument("--no-ppl", action="store_true"); ap.add_argument("--no-chat", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True); res = dict(model=a.model, tag=a.tag, device=a.device)
    t0 = time.time(); cfg = json.load(open(os.path.join(a.model, "config.json")))
    kw = dict(quantization_config=CompressedTensorsConfig(run_compressed=False)) if "quantization_config" in cfg else {}
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu", **kw).eval()
    res["load_cpu_s"] = round(time.time() - t0, 2)
    if a.device != "cpu":
        try:
            with torch.no_grad():
                for t in list(m.parameters()) + list(m.buffers()): t.data = t.data.contiguous().clone()  # decompressed views -> aligned storage
            m = m.to(a.device)
        except Exception as ex:
            print("MPS move failed, falling back to CPU:", str(ex)[:120], flush=True); m = m.to("cpu"); a.device = "cpu"; res["device"] = "cpu(fallback)"
    if a.device == "mps": torch.mps.synchronize()
    res["load_s"] = round(time.time() - t0, 2); res["rss_after_load_GB"] = round(rss_gb(), 3)
    tok = AutoTokenizer.from_pretrained(a.model)
    res["param_bytes_GB"] = round(sum(p.numel() * p.element_size() for p in m.parameters()) / 1e9, 3)
    print(json.dumps(res), flush=True)
    if not a.no_ppl:
        data = json.load(open("docs/Plan/m0/raw/eval_tokens.json")); bos = data["bos_id"]
        @torch.no_grad()
        def ppl(ids, ctx=1024, stride=512):
            ids = ids[:a.limit] if a.limit else ids; N = len(ids); nll = 0.0; cnt = 0; prev = 0; L = ctx - 1
            for b in range(0, N, stride):
                e = min(b + L, N); x = torch.tensor([[bos] + ids[b:e]], device=a.device)
                lo = m(x).logits[0, :-1].float().cpu(); lp = lo - torch.logsumexp(lo, -1, keepdim=True)  # score on CPU (MPS gather mixed-device error)
                tl = lp.gather(-1, torch.tensor(ids[b:e], dtype=torch.long)[:, None])[:, 0]; s = max(prev, b) - b
                nll += -tl[s:].sum().item(); cnt += e - b - s; prev = e
                if e == N: break
            return math.exp(nll / cnt), cnt
        t1 = time.time(); res["ppl_wikitext"], res["ppl_wikitext_tokens"] = ppl(data["wikitext"]); res["ppl_zh_tw"], res["ppl_zh_tw_tokens"] = ppl(data["zh_tw"])
        res["ppl_time_s"] = round(time.time() - t1, 1); print(json.dumps(res), flush=True)
    if not a.no_chat:
        sys.path.insert(0, "docs/Plan/m7/tools"); from m6_chat_prompts import P
        with open(os.path.join(a.outdir, f"lat_{a.tag}.jsonl"), "w") as f:
            for pid, cat, prompt, _ in P:
                x = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True).to(a.device)
                t1 = time.time()
                with torch.no_grad(): o = m.generate(**x, max_new_tokens=256, do_sample=False)
                g = o[0][x["input_ids"].shape[1]:].tolist(); raw = tok.decode(g); ans = tok.decode(g, skip_special_tokens=True).strip()
                f.write(json.dumps(dict(run=1, id=pid, cat=cat, answer=ans, raw=raw, n_ch_open=raw.count("<|channel>"), hit_max=len(g) >= 256 and 106 not in g, answer_tokens=len(g), total_s=round(time.time() - t1, 3)), ensure_ascii=False) + "\n"); f.flush()
                print(f"chat {pid} {len(g)} tok {time.time() - t1:.1f}s {ans[:40]!r}", flush=True)
    res["max_rss_GB"] = round(rss_gb(), 3)
    if a.device == "mps": res["mps_driver_peak_GB"] = round(torch.mps.driver_allocated_memory() / 1e9, 3)
    json.dump(res, open(os.path.join(a.outdir, f"hf_{a.tag}.json"), "w"), indent=1); print(json.dumps(res)); print("HF_EVAL_DONE", flush=True)
if __name__ == "__main__": main()
