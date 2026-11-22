"""M0 benchmark (PyTorch/transformers, MPS): used for official compressed-tensors checkpoints
(+ bf16 reference to cross-check against MLX). Same PPL protocol as m0_bench_mlx.py.
Usage: m0_bench_torch.py <name> <model_dir> [--device mps|cpu] [--smoke]"""
import time, json, math, resource, argparse, threading
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("path")
ap.add_argument("--device", default="mps"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--tag", default="")
ap.add_argument("--ctx", type=int, default=1024); ap.add_argument("--stride", type=int, default=512)
a = ap.parse_args(); R = "docs/Plan/m0/raw/"; dev = a.device
out = {"name": a.name, "path": a.path, "framework": "torch", "torch": torch.__version__,
       "transformers": transformers.__version__, "device": dev}
peak = [0]
def poll():
    while True:
        if dev == "mps": peak[0] = max(peak[0], torch.mps.driver_allocated_memory())
        time.sleep(0.05)
threading.Thread(target=poll, daemon=True).start()
t = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(a.path, dtype=torch.bfloat16, device_map=dev)
model.eval(); out["load_s"] = time.perf_counter() - t; out["model_class"] = type(model).__name__
if dev == "mps": torch.mps.synchronize(); out["mem_after_load_GB"] = torch.mps.driver_allocated_memory() / 1e9
out["param_bytes_GB"] = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
out["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
tok = AutoTokenizer.from_pretrained(a.path)
msgs = [{"role": "user", "content": "請用繁體中文詳細介紹台灣的地理、歷史與文化，至少寫五段。"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(dev)
with torch.no_grad():
    model.generate(ids, max_new_tokens=8, do_sample=False)
    n = 8 if a.smoke else 128
    if dev == "mps": torch.mps.synchronize()
    t = time.perf_counter(); g = model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)
    if dev == "mps": torch.mps.synchronize()
    dt = time.perf_counter() - t
out["gen_tokens"] = g.shape[1] - ids.shape[1]; out["gen_tps_incl_prefill"] = out["gen_tokens"] / dt
out["gen_text"] = tok.decode(g[0, ids.shape[1]:]); out["gen_peak_mem_GB"] = peak[0] / 1e9
data = json.load(open(R + "eval_tokens.json")); bos = data["bos_id"]
def ppl(ids, limit=None):
    if limit: ids = ids[:limit]
    N = len(ids); nll = 0.0; cnt = 0; prev = 0; L = a.ctx - 1
    for b in range(0, N, a.stride):
        e = min(b + L, N)
        x = torch.tensor([[bos] + ids[b:e]], device=dev)
        with torch.no_grad():
            logits = model(input_ids=x).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1)
        tl = lp.gather(-1, torch.tensor(ids[b:e], device=dev)[:, None])[:, 0]
        s = max(prev, b) - b
        nll += (-tl[s:].sum()).item(); cnt += (e - b - s); prev = e
        if e == N: break
    return math.exp(nll / cnt), cnt
peak[0] = 0; t = time.perf_counter()
lim = 2048 if a.smoke else a.limit
out["ppl_wikitext"], out["ppl_wikitext_tokens"] = ppl(data["wikitext"], lim)
out["ppl_zh_tw"], out["ppl_zh_tw_tokens"] = ppl(data["zh_tw"], lim)
out["ppl_time_s"] = time.perf_counter() - t; out["ppl_peak_mem_GB"] = peak[0] / 1e9
out["max_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
json.dump(out, open(R + f"bench_{a.name}{a.tag}{'_smoke' if a.smoke else ''}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "gen_text"}, ensure_ascii=False)); print(out["gen_text"][:200])
