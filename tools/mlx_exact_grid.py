#!/usr/bin/env python
"""Rewrite an `mlx_lm convert -q --q-bits 4 --q-group-size 32` model so every quantized tensor reproduces the
symmetric Q4 block-32 grid of the HF source *exactly* (levels L in [-8,7], one scale d per block of 32 along the
input axis): MLX affine w = scale*q + bias with q = L+8, scale = bf16(d), bias = -8*scale (exact in bf16).
d is the least-squares scale given the levels (bf16 min/max lose the true scale). Blocks that are not on a grid
(residual > tol) keep MLX's own affine quantization.
Usage: python tools/mlx_exact_grid.py --hf HF_DIR --mlx MLX_G32_DIR --out OUT_DIR"""
import argparse, glob, json, os, shutil
import mlx.core as mx

def grid_fit(w, tol=0.05, snap=False):
    b = w.astype(mx.float32).reshape(-1, 32)
    d0 = mx.maximum(-b.min(-1, keepdims=True) / 8, b.max(-1, keepdims=True) / 7)
    L = mx.clip(mx.round(b / mx.where(d0 == 0, 1, d0)), -8, 7)
    if snap:  # PTQ: force every block onto the symmetric grid (Q4_0_MINMAX formula, scale stored in bf16)
        d0 = d0.astype(mx.bfloat16).astype(mx.float32)
        L = mx.clip(mx.round(b / mx.where(d0 == 0, 1, d0)), -8, 7)
        return L, d0, mx.ones((b.shape[0],), dtype=mx.bool_)
    d = (b * L).sum(-1, keepdims=True) / mx.maximum((L * L).sum(-1, keepdims=True), 1e-30)
    res = mx.abs(b / mx.where(d == 0, 1, d) - L).max(-1)
    ok = (res < tol) | (mx.abs(b).max(-1) == 0)
    return L, d, ok

def pack(L, shape):
    q = (L + 8).astype(mx.uint32).reshape(shape[0], -1, 8)
    out = mx.zeros(q.shape[:-1], mx.uint32)
    for i in range(8): out = out | (q[..., i] << (4 * i))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hf", required=True); ap.add_argument("--mlx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snap", action="store_true", help="source is off-grid (e.g. fused LoRA): snap with Q4_0_MINMAX (PTQ)")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.mlx, "config.json"))); assert cfg["quantization"]["group_size"] == 32
    hf = {}
    for f in glob.glob(os.path.join(a.hf, "*.safetensors")): hf.update(mx.load(f))
    def hf_name(k):
        for c in (k.replace("language_model.model.", "model.language_model."), k.replace("language_model.model.", "model."),
                  k, "model." + k.split("model.", 1)[-1]):
            if c in hf: return c
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.mlx):
        if not f.endswith(".safetensors"): shutil.copy(os.path.join(a.mlx, f), os.path.join(a.out, f))
    rep = {"snap": a.snap, "tensors": 0, "blocks": 0, "blocks_on_grid": 0, "fallback_blocks": 0, "max_abs_err_vs_hf": 0.0, "missing_hf": []}
    for f in sorted(glob.glob(os.path.join(a.mlx, "*.safetensors"))):
        W = mx.load(f); out = dict(W)
        for k in [k for k in W if k.endswith(".scales")]:
            base = k[:-len(".scales")]; n = hf_name(base + ".weight")
            if n is None: rep["missing_hf"].append(base); continue
            w = hf[n]; shape = w.shape if w.ndim == 2 else (w.shape[0], -1)
            L, d, ok = grid_fit(w, snap=a.snap)
            s = d.astype(mx.bfloat16); s32 = s.astype(mx.float32)
            ok_all = bool(ok.all())
            if not ok_all:
                rep["fallback_blocks"] += int((~ok).sum())
            packed = pack(L, (w.shape[0], w.size // w.shape[0]))
            scales = s.reshape(w.shape[0], -1); biases = (-8 * s32).astype(mx.bfloat16).reshape(w.shape[0], -1)
            if not ok_all:  # keep MLX affine for off-grid blocks (row-group granularity)
                okr = ok.reshape(w.shape[0], -1)
                scales = mx.where(okr, scales, W[base + ".scales"]); biases = mx.where(okr, biases, W[base + ".biases"])
                okp = mx.repeat(okr, 4, axis=1)  # 32 values = 4 uint32 per group
                packed = mx.where(okp, packed, W[base + ".weight"])
            deq = mx.dequantize(packed, scales, biases, group_size=32, bits=4)
            err = float(mx.abs(deq.astype(mx.float32) - w.astype(mx.float32).reshape(deq.shape)).max())
            if not a.snap: rep["max_abs_err_vs_hf"] = max(rep["max_abs_err_vs_hf"], err)
            out[base + ".weight"], out[base + ".scales"], out[base + ".biases"] = packed, scales, biases
            rep["tensors"] += 1; rep["blocks"] += int(ok.size); rep["blocks_on_grid"] += int(ok.sum())
            mx.eval(out[base + ".weight"], out[base + ".scales"], out[base + ".biases"])
        mx.save_safetensors(os.path.join(a.out, os.path.basename(f)), out, metadata={"format": "mlx"})
    rep["on_grid_frac"] = rep["blocks_on_grid"] / max(1, rep["blocks"])
    json.dump(rep, open(os.path.join(a.out, "exact_grid_report.json"), "w"), indent=1); print(json.dumps(rep))

if __name__ == "__main__":
    main()
