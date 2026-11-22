# M0: download official compressed-tensors checkpoints into weights/ (text+towers, as published)
import time
from huggingface_hub import snapshot_download
for rid in ["google/gemma-4-E2B-it-qat-mobile-ct", "google/gemma-4-E2B-it-qat-w4a16-ct"]:
    t = time.time()
    p = snapshot_download(rid, local_dir=f"weights/{rid.split('/')[1]}")
    print("DONE", rid, p, f"{time.time()-t:.0f}s", flush=True)
print("ALL DONE", flush=True)
