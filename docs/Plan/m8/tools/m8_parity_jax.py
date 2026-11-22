"""M8 parity / losslessness checks.
  mlx  : (a) MLX q4_minmax_bf16(W+sAB) data for JAX; (b) fixed-grid fake-quant grads; writes /tmp/m8_par.npz
  jax  : compare (a) with gemma.peft._block_quantization.block_quantize(minmax, bf16 scale) - the M5 recipe
  real : on the real model: Q_fixed(W_official) == W_official for all 205 targets (base is a fixed point)"""
import sys, numpy as np
sys.path.insert(0, "docs/Plan/m8/tools")
if sys.argv[1] == "mlx":
    import mlx.core as mx, mlx.nn as nn
    from m8_qat_lora import q4_minmax_bf16, QATLoRALinear
    rng = np.random.default_rng(0)
    W = np.asarray(mx.array((rng.standard_normal((96, 256)) * 0.02).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))
    A = (rng.standard_normal((256, 8)) * 0.05).astype(np.float32); B = (rng.standard_normal((8, 96)) * 0.05).astype(np.float32); s = 2.0
    x = rng.standard_normal((4, 256)).astype(np.float32)
    wp = mx.array(W) + s * (mx.array(A) @ mx.array(B)).T
    qm = np.asarray(q4_minmax_bf16(wp).astype(mx.bfloat16).astype(mx.float32)); y = x @ qm.T
    # fixed-grid module: gradient reaches A/B through the STE
    d = mx.array(np.abs(W).reshape(96, 8, 32).max(-1) / 8).astype(mx.bfloat16)
    lin = nn.Linear(256, 96, bias=False); lin.weight = mx.array(W).astype(mx.bfloat16)
    m = QATLoRALinear(lin, d, 8, s); B2 = mx.array(B)
    def f(a, b): m.lora_a, m.lora_b = a, b; return m(mx.array(x)).sum()
    ga, gb = mx.grad(f, argnums=(0, 1))(m.lora_a, B2)
    np.savez("/tmp/m8_par.npz", W=W, A=A, B=B, s=s, x=x, qm=qm, y=y)
    print("STE grad |A|:", float(mx.abs(ga).sum()), "|B|:", float(mx.abs(gb).sum()))
elif sys.argv[1] == "jax":
    import jax.numpy as jnp
    from gemma.peft import _block_quantization as bq
    z = np.load("/tmp/m8_par.npz"); W, A, B, s, x = z["W"], z["A"], z["B"], float(z["s"]), z["x"]
    wp = jnp.asarray(W) + s * (jnp.asarray(A) @ jnp.asarray(B)).T
    qj = np.asarray(jnp.asarray(bq.block_quantize(wp, (1,), block_size=32, formula="minmax", scale_dtype=jnp.bfloat16)).astype(jnp.bfloat16).astype(jnp.float32))
    print("max|Q_mlx-Q_jax|:", float(np.abs(z["qm"] - qj).max()), "blocks differing:", int((np.abs(z["qm"] - qj).reshape(-1, 32).max(-1) > 0).sum()), "/", qj.size // 32,
          "max|y_mlx-y_jax|:", float(np.abs(z["y"] - x @ qj.T).max()))
else:
    import mlx.core as mx, mlx.nn as nn
    from mlx_lm import load
    from m8_qat_lora import apply_qat_lora, QATLoRALinear, load_pack, load_deployed_dense
    m, _ = load("out/m5/qat-text-hf"); P = load_pack(); nd = load_deployed_dense(m, P); n = apply_qat_lora(m, P=P)
    print("dense tensors replaced by deployed dequant:", nd)
    bad = blocks = 0; mxe = 0.0
    for name, mod in m.named_modules():
        if isinstance(mod, QATLoRALinear):
            q = mod.quantized_weight().astype(mx.float32); w = mod.weight.astype(mx.float32)
            e = mx.abs(q - w); mxe = max(mxe, float(e.max())); bad += int((e.reshape(-1, 32).max(-1) > 0).sum()); blocks += e.size // 32
    print("targets", n, "blocks", blocks, "changed_by_Q_fixed", bad, "max_abs", mxe,
          "trainable", sum(v.size for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(m.trainable_parameters())))
