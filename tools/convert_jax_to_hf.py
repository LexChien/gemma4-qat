#!/usr/bin/env python
"""Gemma 4 E2B: JAX/orbax (google-deepmind/gemma checkpoint) -> HF safetensors (transformers Gemma4).

Mapping rules: docs/Plan/trace/04_weights_map.md + docs/Plan/m3 verification.
  * Einsum/Linear kernels -> HF [out,in]: transposed, EXCEPT text/vision mlp.gating_einsum (gate/up) which are
    already [F,D]; square matrices are transposed too.
  * RMSNorm scales copied as-is (HF multiplies by w, no +/-1). Embeddings / PLE unscaled (scaling at forward).
  * orbax dtypes: f32 everywhere except 151 audio tensors (bf16). Output cast with round-to-nearest-even.
  * vision layers are stacked on axis 0 (16 layers); clip_* arrays [16] -> per-layer scalars.
  * embedder.{mm,audio}_input_embedding_extra have no HF counterpart (dropped).
Usage:
  python tools/convert_jax_to_hf.py --orbax weights/orbax/checkpoints/gemma4-e2b-it --out out/m3/it-bf16 \
      [--hf_ref weights/gemma-4-E2B-it] [--dtype bf16|f32] [--text_only] [--drop_shared_kv]
  --text_only     : only the language model, Gemma4ForCausalLM layout (keys model.*, config = text_config)
  --drop_shared_kv: omit k_proj/v_proj/k_norm of KV-shared layers (like the official QAT HF checkpoint)
Read-only on inputs. Requires: numpy, tensorstore, torch, safetensors."""
import argparse, json, os, shutil, sys, time
import numpy as np, tensorstore as ts, torch
from safetensors.torch import save_file

def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orbax", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--hf_ref", default=None, help="HF dir to copy config/tokenizer from (default: guessed from orbax name)")
    ap.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    ap.add_argument("--text_only", action="store_true"); ap.add_argument("--drop_shared_kv", action="store_true")
    return ap.parse_args()

class Orbax:
    def __init__(self, base):
        self.base = os.path.abspath(base) + "/"
        meta_all = json.load(open(self.base + "_METADATA")); meta = meta_all["tree_metadata"]
        self.driver = "zarr3" if meta_all.get("use_zarr3") else "zarr"  # orbax >=0.11 saves zarr3 by default
        import ast
        self.keys = {".".join(ast.literal_eval(k)) for k in meta}
        self.used = set(); self.cache = {}
    def __call__(self, path):
        self.used.add(path)
        if path in self.cache: return self.cache[path]
        if "stacked_layers.block" in path:  # vision: 16 layers stacked on axis 0 -> read once
            self.cache[path] = self._read(path); return self.cache[path]
        return self._read(path)
    def _read(self, path):
        t = ts.open({"driver": self.driver, "kvstore": {"driver": "ocdbt", "base": "file://" + self.base, "path": path + "/"}}).result()
        return np.asarray(t.read().result())

def to_torch(x, dtype):
    shape = x.shape  # keep 0-d scalars 0-d (np.ascontiguousarray would make them [1])
    if x.dtype.name == "bfloat16":
        t = torch.from_numpy(np.ascontiguousarray(x).view(np.uint16).copy()).view(torch.bfloat16)
    else:
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    t = t.reshape(shape)
    return t.to(torch.bfloat16 if dtype == "bf16" else torch.float32).contiguous()

def text_map(O, cfg, drop_shared_kv):
    tc = cfg["text_config"]; L = "model.language_model."
    n_layers = tc["num_hidden_layers"]; n_shared = tc.get("num_kv_shared_layers", 0)
    yield L + "embed_tokens.weight", lambda: O("embedder.input_embedding")
    yield L + "embed_tokens_per_layer.weight", lambda: (lambda x: x.reshape(x.shape[0], -1))(O("embedder.per_layer_embeddings"))
    yield L + "per_layer_model_projection.weight", lambda: (lambda x: x.reshape(x.shape[0], -1).T)(O("embedder.per_layer_model_projection.w"))
    yield L + "per_layer_projection_norm.weight", lambda: O("embedder.per_layer_projection_norm.scale")
    yield L + "norm.weight", lambda: O("final_norm.scale")
    for i in range(n_layers):
        p = f"layer_{i}."; q = f"{L}layers.{i}."
        shared = i >= n_layers - n_shared
        def qp(p=p):
            w = O(p + "attn.q_einsum.w"); N, D, H = w.shape; return w.transpose(0, 2, 1).reshape(N * H, D)
        def kvp(c, p=p):
            w = O(p + "attn.kv_einsum.w")[c]; K, D, H = w.shape; return w.transpose(0, 2, 1).reshape(K * H, D)
        def op(p=p):
            w = O(p + "attn.attn_vec_einsum.w"); N, H, D = w.shape; return w.reshape(N * H, D).T
        yield q + "self_attn.q_proj.weight", qp
        if not (drop_shared_kv and shared):
            yield q + "self_attn.k_proj.weight", lambda f=kvp: f(0)
            yield q + "self_attn.v_proj.weight", lambda f=kvp: f(1)
        yield q + "self_attn.o_proj.weight", op
        yield q + "self_attn.q_norm.weight", lambda p=p: O(p + "attn.query_norm.scale")
        if not (drop_shared_kv and shared):
            yield q + "self_attn.k_norm.weight", lambda p=p: O(p + "attn.key_norm.scale")
        yield q + "mlp.gate_proj.weight", lambda p=p: O(p + "mlp.gating_einsum.w")[0]
        yield q + "mlp.up_proj.weight", lambda p=p: O(p + "mlp.gating_einsum.w")[1]
        yield q + "mlp.down_proj.weight", lambda p=p: O(p + "mlp.linear.w").T
        for hf_n, jx in [("input_layernorm", "pre_attention_norm"), ("post_attention_layernorm", "post_attention_norm"),
                         ("pre_feedforward_layernorm", "pre_ffw_norm"), ("post_feedforward_layernorm", "post_ffw_norm"),
                         ("post_per_layer_input_norm", "post_per_layer_input_norm")]:
            yield q + hf_n + ".weight", lambda p=p, jx=jx: O(p + jx + ".scale")
        yield q + "per_layer_input_gate.weight", lambda p=p: O(p + "per_layer_input_gate.w").T
        yield q + "per_layer_projection.weight", lambda p=p: O(p + "per_layer_projection.w").T
        yield q + "layer_scalar", lambda p=p: O(p + "skip_scale")

CLIPS = [("input_min", "clip_input_min"), ("input_max", "clip_input_max"), ("output_min", "clip_output_min"), ("output_max", "clip_output_max")]

def vision_map(O, cfg):
    V = "model.vision_tower."; B = "vision_encoder.transformer.stacked_layers.block."
    yield V + "patch_embedder.input_proj.weight", lambda: O("vision_encoder.entry.input_projection.w").T
    yield V + "patch_embedder.position_embedding_table", lambda: O("vision_encoder.entry.pos_emb").transpose(1, 0, 2)
    for l in range(cfg["vision_config"]["num_hidden_layers"]):
        q = f"{V}encoder.layers.{l}."
        def att(kind, c=None, l=l):
            if kind == "q": w = O(B + "attn.q_einsum.w")[l]; N, D, H = w.shape; return w.transpose(0, 2, 1).reshape(N * H, D)
            if kind == "kv": w = O(B + "attn.kv_einsum.w")[l, c]; N, D, H = w.shape; return w.transpose(0, 2, 1).reshape(N * H, D)
            if kind == "o": w = O(B + "attn.attn_vec_einsum.w")[l]; N, H, D = w.shape; return w.reshape(N * H, D).T
        lin = {"self_attn.q_proj": ("attn.q_einsum", lambda f=att: f("q")),
               "self_attn.k_proj": ("attn.kv_einsum", lambda f=att: f("kv", 0)),
               "self_attn.v_proj": ("attn.kv_einsum", lambda f=att: f("kv", 1)),
               "self_attn.o_proj": ("attn.attn_vec_einsum", lambda f=att: f("o")),
               "mlp.gate_proj": ("mlp.gating_einsum", lambda l=l: O(B + "mlp.gating_einsum.w")[l, 0]),
               "mlp.up_proj": ("mlp.gating_einsum", lambda l=l: O(B + "mlp.gating_einsum.w")[l, 1]),
               "mlp.down_proj": ("mlp.linear", lambda l=l: O(B + "mlp.linear.w")[l].T)}
        for hf_n, (jx, fn) in lin.items():
            yield q + hf_n + ".linear.weight", fn
            for hc, jc in CLIPS:
                yield q + f"{hf_n}.{hc}", lambda l=l, jx=jx, jc=jc: O(B + f"{jx}.{jc}")[l]
        yield q + "self_attn.q_norm.weight", lambda l=l: O(B + "attn.query_norm.scale")[l]
        yield q + "self_attn.k_norm.weight", lambda l=l: O(B + "attn.key_norm.scale")[l]
        for hf_n, jx in [("input_layernorm", "pre_attention_norm"), ("post_attention_layernorm", "post_attention_norm"),
                         ("pre_feedforward_layernorm", "pre_ffw_norm"), ("post_feedforward_layernorm", "post_ffw_norm")]:
            yield q + hf_n + ".weight", lambda l=l, jx=jx: O(B + jx + ".scale")[l]
    yield "model.embed_vision.embedding_projection.weight", lambda: O("embedder.mm_input_projection.w").T

def audio_map(O, cfg):
    A = "model.audio_tower."
    for l in range(cfg["audio_config"]["num_hidden_layers"]):
        p = f"audio_encoder.conformer/stacked_layers_{l}."; q = f"{A}layers.{l}."
        def linear(hf_pre, jx, tf=lambda w: w.T):
            yield hf_pre + ".linear.weight", lambda: tf(O(jx + ".kernel"))
            for hc, jc in CLIPS:
                yield hf_pre + "." + hc, lambda jc=jc: O(jx + "." + jc)
        for hf_ff, jff in [("feed_forward1", "fflayer_start"), ("feed_forward2", "fflayer_end")]:
            yield from linear(q + hf_ff + ".ffw_layer_1", p + jff + ".ffn_layer1")
            yield from linear(q + hf_ff + ".ffw_layer_2", p + jff + ".ffn_layer2")
            yield q + hf_ff + ".pre_layer_norm.weight", lambda p=p, j=jff: O(p + j + ".pre_layer_norm.scale")
            yield q + hf_ff + ".post_layer_norm.weight", lambda p=p, j=jff: O(p + j + ".post_layer_norm.scale")
        yield q + "lconv1d.conv_norm.weight", lambda p=p: O(p + "lconv.conv_norm.scale")
        yield q + "lconv1d.depthwise_conv1d.weight", lambda p=p: O(p + "lconv.depthwise_conv1d.kernel").transpose(2, 1, 0)
        yield from linear(q + "lconv1d.linear_start", p + "lconv.linear_start")
        yield from linear(q + "lconv1d.linear_end", p + "lconv.linear_end")
        yield q + "lconv1d.pre_layer_norm.weight", lambda p=p: O(p + "lconv.ln.scale")
        yield q + "norm_out.weight", lambda p=p: O(p + "final_ln.scale")
        yield q + "norm_post_attn.weight", lambda p=p: O(p + "trans_atten.post_norm.scale")
        yield q + "norm_pre_attn.weight", lambda p=p: O(p + "trans_atten.pre_norm.scale")
        for hf_n, jx in [("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value")]:
            yield from linear(q + "self_attn." + hf_n, p + "trans_atten.self_atten." + jx)
        yield q + "self_attn.per_dim_scale", lambda p=p: O(p + "trans_atten.self_atten.per_dim_scale")
        yield from linear(q + "self_attn.post", p + "trans_atten.post", tf=lambda w: w.reshape(-1, w.shape[-1]).T)
        yield q + "self_attn.relative_k_proj.weight", lambda p=p: (lambda w: w.reshape(w.shape[0], -1).T)(
            O(p + "trans_atten.self_atten.relative_position_embedding.pos_proj.kernel"))
    S = A + "subsample_conv_projection."
    yield S + "input_proj_linear.weight", lambda: (lambda w: w.reshape(-1, w.shape[-1]).T)(O("audio_encoder.feature.input_proj.kernel"))
    for i in (0, 1):
        yield S + f"layer{i}.conv.weight", lambda i=i: O(f"audio_encoder.feature.subsampling_{i}.kernel").transpose(3, 2, 0, 1)
        yield S + f"layer{i}.norm.weight", lambda i=i: O(f"audio_encoder.feature.norm_{i}.scale")
    yield A + "output_proj.weight", lambda: O("audio_encoder.output_projection.kernel").T
    yield A + "output_proj.bias", lambda: O("audio_encoder.output_projection.bias")
    yield "model.embed_audio.embedding_projection.weight", lambda: O("embedder.audio_input_projection.w").T

def main():
    a = parse(); t0 = time.time()
    ref = a.hf_ref or ("weights/gemma-4-E2B-it" if a.orbax.rstrip("/").endswith("-it") else "weights/gemma-4-E2B")
    cfg = json.load(open(os.path.join(ref, "config.json")))
    O = Orbax(a.orbax)
    items = list(text_map(O, cfg, a.drop_shared_kv))
    if not a.text_only: items += list(vision_map(O, cfg)) + list(audio_map(O, cfg))
    out = {}
    for n, (k, fn) in enumerate(items):
        x = fn(); out[k] = to_torch(x, a.dtype)
        if n % 200 == 0: print(f"[{n}/{len(items)}] {k} {tuple(out[k].shape)} {time.time()-t0:.0f}s", flush=True)
    if a.text_only:
        out = {k.replace("model.language_model.", "model."): v for k, v in out.items()}
    os.makedirs(a.out, exist_ok=True)
    save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    if a.text_only:
        tc = dict(cfg["text_config"]); tc["architectures"] = ["Gemma4ForCausalLM"]
        tc["dtype"] = "bfloat16" if a.dtype == "bf16" else "float32"
        json.dump(tc, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    else:
        c = dict(cfg); c["dtype"] = "bfloat16" if a.dtype == "bf16" else "float32"
        for sub in ("text_config", "vision_config", "audio_config"):
            if sub in c and isinstance(c[sub], dict): c[sub] = dict(c[sub], dtype=c["dtype"])
        json.dump(c, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in os.listdir(ref):
        if f.endswith((".json", ".jinja")) and f not in ("config.json", "model.safetensors.index.json"):
            if a.text_only and f in ("processor_config.json", "preprocessor_config.json"): continue
            shutil.copy(os.path.join(ref, f), os.path.join(a.out, f))
    unused = sorted(O.keys - O.used)
    rep = {"orbax": a.orbax, "hf_ref": ref, "dtype": a.dtype, "text_only": a.text_only, "drop_shared_kv": a.drop_shared_kv,
           "n_tensors": len(out), "n_params": int(sum(v.numel() for v in out.values())),
           "orbax_leaves": len(O.keys), "orbax_unused": unused, "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(a.out, "convert_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "orbax_unused"}), f"orbax leaves unused: {len(unused)}", unused[:4])

if __name__ == "__main__":
    main()
