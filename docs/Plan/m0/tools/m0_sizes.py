"""M0: on-disk size of every variant: apparent bytes (sum st_size), du -sk (allocated KiB), weights-only bytes."""
import os, json, subprocess
V = {"hf-it-bf16": "weights/gemma-4-E2B-it", "hf-qat-q4_0-unquantized": "weights/gemma-4-E2B-it-qat-q4_0-unquantized",
     "official-w4a16-ct": "weights/gemma-4-E2B-it-qat-w4a16-ct", "official-mobile-ct": "weights/gemma-4-E2B-it-qat-mobile-ct",
     **{n: f"out/m0/{n}" for n in ["mlx-it-q4g64", "mlx-it-q4g32", "mlx-qat-q4g64", "mlx-qat-q4g32", "mlx-qat-safe-g64", "mlx-qat-mobile-g64"]}}
res = {}
for n, p in V.items():
    tot = wts = 0
    for root, ds, fs in os.walk(p):
        ds[:] = [d for d in ds if d != ".cache"]
        for f in fs:
            s = os.path.getsize(os.path.join(root, f)); tot += s
            if f.endswith(".safetensors"): wts += s
    du = int(subprocess.check_output(["du", "-sk", p]).split()[0])
    res[n] = {"path": p, "apparent_bytes": tot, "GB": round(tot / 1e9, 3), "GiB": round(tot / 2**30, 3), "weights_bytes": wts, "du_KiB": du}
    print(f"{n:28s} {tot/1e9:7.3f} GB {tot/2**30:7.3f} GiB  du {du/2**20:6.2f} GiB")
json.dump(res, open("docs/Plan/m0/raw/sizes.json", "w"), indent=1)
