#!/usr/bin/env python
"""Gemma 4 E2B text: HF safetensors (e.g. official gemma-4-E2B-it-qat-q4_0-unquantized) -> gemma JAX param tree
(orbax/`gm.ckpts.load_params` layout, incl. mlp.{gating_einsum,linear} = {'w': ...}). Inverse of
tools/convert_jax_to_hf.py:text_map. KV-shared layers whose k/v/k_norm are absent in HF (official QAT drops them;
never used at runtime) are filled from an orbax checkpoint (--fill_orbax) or zeros.
API: load_hf_text_params(hf_dir, fill_orbax=None, dtype=ml_dtypes.bfloat16) -> nested dict of numpy arrays.
CLI self-test: python tools/convert_hf_to_jax.py --hf DIR [--fill_orbax DIR] -> round-trips through text_map, compares bit-exact."""
import argparse, json, os, sys
import numpy as np, ml_dtypes, torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _reader(hf_dir):
    f = safe_open(os.path.join(hf_dir, "model.safetensors"), "pt")
    keys = set(f.keys())
    pre = "model.language_model." if any(k.startswith("model.language_model.") for k in keys) else "model."
    def get(name):
        t = f.get_tensor(pre + name)
        if t.dtype == torch.bfloat16: return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
        return t.float().numpy()
    has = lambda name: (pre + name) in keys
    return get, has

def load_hf_text_params(hf_dir, fill_orbax=None, dtype=ml_dtypes.bfloat16):
    cfg = json.load(open(os.path.join(hf_dir, "config.json"))); tc = cfg.get("text_config", cfg)
    get, has = _reader(hf_dir); c = lambda x: np.ascontiguousarray(x).astype(dtype)
    O = None
    if fill_orbax:
        from convert_jax_to_hf import Orbax
        O = Orbax(fill_orbax)
    E = get("embed_tokens.weight"); V, D = E.shape; L = tc["num_hidden_layers"]
    Pl = get("embed_tokens_per_layer.weight"); P = Pl.shape[1] // L
    p = {"embedder": {"input_embedding": c(E), "per_layer_embeddings": c(Pl.reshape(V, L, P)),
                      "per_layer_model_projection": {"w": c(get("per_layer_model_projection.weight").T.reshape(D, L, P))},
                      "per_layer_projection_norm": {"scale": c(get("per_layer_projection_norm.weight"))}},
         "final_norm": {"scale": c(get("norm.weight"))}}
    del E, Pl
    for i in range(L):
        q = f"layers.{i}."; lp = {}
        wq = get(q + "self_attn.q_proj.weight"); qn = get(q + "self_attn.q_norm.weight"); H = qn.shape[0]; N = wq.shape[0] // H
        attn = {"q_einsum": {"w": c(wq.reshape(N, H, D).transpose(0, 2, 1))}, "query_norm": {"scale": c(qn)}}
        wo = get(q + "self_attn.o_proj.weight"); attn["attn_vec_einsum"] = {"w": c(wo.T.reshape(N, H, D))}
        if has(q + "self_attn.k_proj.weight"):
            ks = [get(q + f"self_attn.{n}_proj.weight") for n in ("k", "v")]; K = ks[0].shape[0] // H
            attn["kv_einsum"] = {"w": c(np.stack([k.reshape(K, H, D).transpose(0, 2, 1) for k in ks]))}
            attn["key_norm"] = {"scale": c(get(q + "self_attn.k_norm.weight"))}
        elif O is not None:
            attn["kv_einsum"] = {"w": c(O(f"layer_{i}.attn.kv_einsum.w"))}
            attn["key_norm"] = {"scale": c(O(f"layer_{i}.attn.key_norm.scale"))}
        else:
            K = tc.get("num_key_value_heads", 1)
            attn["kv_einsum"] = {"w": np.zeros((2, K, D, H), dtype)}; attn["key_norm"] = {"scale": np.zeros((H,), dtype)}
        lp["attn"] = attn
        lp["mlp"] = {"gating_einsum": {"w": c(np.stack([get(q + "mlp.gate_proj.weight"), get(q + "mlp.up_proj.weight")]))},
                     "linear": {"w": c(get(q + "mlp.down_proj.weight").T)}}
        for hf_n, jx in [("input_layernorm", "pre_attention_norm"), ("post_attention_layernorm", "post_attention_norm"),
                         ("pre_feedforward_layernorm", "pre_ffw_norm"), ("post_feedforward_layernorm", "post_ffw_norm"),
                         ("post_per_layer_input_norm", "post_per_layer_input_norm")]:
            lp[jx] = {"scale": c(get(q + hf_n + ".weight"))}
        lp["per_layer_input_gate"] = {"w": c(get(q + "per_layer_input_gate.weight").T)}
        lp["per_layer_projection"] = {"w": c(get(q + "per_layer_projection.weight").T)}
        lp["skip_scale"] = c(get(q + "layer_scalar"))
        p[f"layer_{i}"] = lp
    return p

def flat(p, pre=""):
    out = {}
    for k, v in p.items():
        if isinstance(v, dict): out.update(flat(v, pre + k + "."))
        else: out[pre + k] = v
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--hf", required=True); ap.add_argument("--fill_orbax"); a = ap.parse_args()
    from convert_jax_to_hf import text_map
    p = load_hf_text_params(a.hf, a.fill_orbax); F = flat(p)
    O = lambda path: F[path].astype(np.float32) if F[path].dtype == ml_dtypes.bfloat16 else F[path]
    cfg = json.load(open(os.path.join(a.hf, "config.json"))); get, has = _reader(a.hf)
    n = bad = 0
    for k, fn in text_map(O, cfg if "text_config" in cfg else {"text_config": cfg}, drop_shared_kv=True):
        name = k.replace("model.language_model.", ""); n += 1
        ref = get(name).astype(np.float32); x = np.asarray(fn(), np.float32).reshape(ref.shape)
        if not np.array_equal(x, ref): bad += 1; print("MISMATCH", k)
    print(json.dumps({"hf": a.hf, "n_leaves": len(F), "roundtrip_tensors": n, "mismatch": bad}))
