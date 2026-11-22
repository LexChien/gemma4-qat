"""Official QAT HF (multimodal) -> text-only Gemma4ForCausalLM dir (keys model.*), same layout as convert_jax_to_hf --text_only."""
import json, os, shutil, sys
from safetensors import safe_open
from safetensors.torch import save_file
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
f = safe_open(src + "/model.safetensors", "pt"); out = {}
for k in f.keys():
    if k.startswith("model.language_model."): out["model." + k[len("model.language_model."):]] = f.get_tensor(k)
save_file(out, dst + "/model.safetensors", metadata={"format": "pt"})
cfg = json.load(open(src + "/config.json")); tc = dict(cfg["text_config"]); tc["architectures"] = ["Gemma4ForCausalLM"]; tc.setdefault("dtype", "bfloat16")
json.dump(tc, open(dst + "/config.json", "w"), indent=2)
for x in os.listdir(src):
    if x.endswith((".json", ".jinja")) and x not in ("config.json", "processor_config.json", "preprocessor_config.json", "model.safetensors.index.json"):
        shutil.copy(os.path.join(src, x), os.path.join(dst, x))
print(len(out), "tensors ->", dst)
