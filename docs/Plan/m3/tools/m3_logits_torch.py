"""M3: transformers (MPS, bf16) logits check: converted checkpoints vs reference HF.
Feeds BOS + first 1023 tokens of wikitext and zh_tw (docs/Plan/m0/raw/eval_tokens.json); compares logits
bit-exact (torch.equal) and max_abs; also reports load warnings (missing/unexpected keys).
Usage: m3_logits_torch.py <out_json> name=dir [name=dir ...]   (first = reference)"""
import sys, json, time, torch, transformers
from transformers import AutoModelForCausalLM
from transformers.utils import logging as hl
outp = sys.argv[1]; models = [a.split("=", 1) for a in sys.argv[2:]]
D = json.load(open("docs/Plan/m0/raw/eval_tokens.json")); bos = D["bos_id"]
X = {n: torch.tensor([[bos] + D[n][:1023]]) for n in ("wikitext", "zh_tw")}
res = {"transformers": transformers.__version__, "torch": torch.__version__, "models": {}}
logits = {}
for name, path in models:
    t = time.time()
    dt = torch.float32 if name.endswith("_f32") else torch.bfloat16  # f32 ckpt: load natively (bf16 cast of 20GB on load was pathologically slow, >12 min)
    m, info = AutoModelForCausalLM.from_pretrained(path, dtype=dt, device_map="mps", output_loading_info=True)
    r = {"path": path, "dtype": str(dt), "class": type(m).__name__, "load_s": round(time.time() - t, 2),
         "loading_info": {k: sorted(v)[:20] if not isinstance(v, (int, float)) else v for k, v in info.items()},
         "n_missing": len(info.get("missing_keys", [])), "n_unexpected": len(info.get("unexpected_keys", []))}
    logits[name] = {}
    with torch.no_grad():
        for n, x in X.items():
            lg = m(input_ids=x.to("mps")).logits.float().cpu()[0]
            logits[name][n] = lg
            lp = torch.log_softmax(lg[:-1], -1); tgt = x[0, 1:]
            r[f"nll_{n}"] = -lp.gather(-1, tgt[:, None]).mean().item()
    res["models"][name] = r; del m; torch.mps.empty_cache()
    print(name, {k: v for k, v in r.items() if k != "loading_info"}, flush=True)
ref = models[0][0]
res["vs_ref"] = {}
for name, _ in models[1:]:
    res["vs_ref"][name] = {n: {"bit_exact": bool(torch.equal(logits[name][n], logits[ref][n])),
                              "max_abs": (logits[name][n] - logits[ref][n]).abs().max().item(),
                              "argmax_agree": (logits[name][n].argmax(-1) == logits[ref][n].argmax(-1)).float().mean().item()}
                           for n in X}
    print("vs", ref, name, res["vs_ref"][name])
json.dump(res, open(outp, "w"), indent=1)
