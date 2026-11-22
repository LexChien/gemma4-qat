"""M6 wrapper around `mlx_lm lora`: optional Metal cache/memory limits (env CACHE_GB, MEM_GB) and resume from the
latest NNNNNNN_adapters.safetensors in adapter_path (remaining iters; LR schedule restarts - documented limitation)."""
import glob, os, re, sys, yaml, tempfile
import mlx.core as mx
cfg_path = sys.argv[1]; cfg = yaml.safe_load(open(cfg_path))
if os.environ.get("CACHE_GB"): mx.set_cache_limit(int(float(os.environ["CACHE_GB"]) * 2**30))
if os.environ.get("MEM_GB"): mx.set_memory_limit(int(float(os.environ["MEM_GB"]) * 2**30))
for k in ("BATCH", "SEQ", "ITERS"):
    if os.environ.get(k): cfg[{"BATCH": "batch_size", "SEQ": "max_seq_length", "ITERS": "iters"}[k]] = int(os.environ[k])
ck = sorted(glob.glob(os.path.join(cfg["adapter_path"], "[0-9]*_adapters.safetensors")))
if ck:
    done = int(re.findall(r"(\d+)_adapters", os.path.basename(ck[-1]))[0])
    cfg["resume_adapter_file"] = ck[-1]; total = cfg["iters"]; cfg["iters"] = max(1, total - done)
    ls = cfg.get("lr_schedule")
    if ls and ls.get("name") == "cosine_decay":  # continue the cosine curve instead of restarting warmup
        import math
        init, steps, end = ls["arguments"]; w = ls.get("warmup", 0); t = min(1.0, (done - w) / max(1, steps - w))
        lr_now = end + 0.5 * (init - end) * (1 + math.cos(math.pi * t))
        cfg["lr_schedule"] = {"name": "cosine_decay", "arguments": [lr_now, cfg["iters"], end]}
        print(f"lr schedule continued from {lr_now:.3g}", flush=True)
    print(f"RESUME from {ck[-1]} (iter {done}), remaining iters {cfg['iters']}", flush=True)
tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False); yaml.safe_dump(cfg, tmp); tmp.close()
print("config", cfg, flush=True)
sys.argv = ["mlx_lm.lora", "-c", tmp.name]
from mlx_lm import lora
lora.main()
