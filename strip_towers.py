"""Optional: drop vision/audio tower tensors from a merged HF checkpoint for text-only deployment
(reduces size by ~450M params est.). Config may also need its vision/audio sections removed;
verify the stripped model loads before quantizing."""
import argparse, json, os, shutil
from safetensors.torch import load_file, save_file
p = argparse.ArgumentParser(); p.add_argument("--src", required=True); p.add_argument("--out", required=True)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
for f in os.listdir(a.src):
    s = os.path.join(a.src, f)
    if f.endswith(".safetensors"):
        sd = load_file(s); keep = {k: v for k, v in sd.items() if not any(t in k for t in ("vision", "audio"))}
        print(f, len(sd), "->", len(keep)); save_file(keep, os.path.join(a.out, f), metadata={"format": "pt"})
    elif f.endswith(".index.json"):
        idx = json.load(open(s)); idx["weight_map"] = {k: v for k, v in idx["weight_map"].items() if not any(t in k for t in ("vision", "audio"))}
        json.dump(idx, open(os.path.join(a.out, f), "w"), indent=2)
    elif os.path.isfile(s): shutil.copy(s, a.out)
