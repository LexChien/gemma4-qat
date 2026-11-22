"""M4 smoke: load official IT orbax into gemma4-qat Gemma4_E2B(text_only), check tree layout, tiny QAT training
on CPU (frozen embedder, trainable decoder), export snapped params as orbax (converter-compatible layout).
Env: STEPS (20) SEQ (128) LR (3e-4) METHOD (Q4_0_ABSMAX) NSEQ (4) SAVE (1)
Writes docs/Plan/m4/raw/smoke_train.json, out/m4/smoke_orbax/"""
import json, os, time, ast, resource, threading
import numpy as np, psutil
import jax, jax.numpy as jnp, optax
import orbax.checkpoint as ocp
from gemma import gm, peft
from gemma.gm.nn.gemma4 import _qat
from gemma.gm.nn import _quantization as gq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
ORBAX = ROOT + "/weights/orbax/checkpoints/gemma4-e2b-it"
STEPS = int(os.environ.get("STEPS", 20)); SEQ = int(os.environ.get("SEQ", 128)); LR = float(os.environ.get("LR", 3e-4))
METHOD = os.environ.get("METHOD", "Q4_0_ABSMAX"); NSEQ = int(os.environ.get("NSEQ", 4)); SAVE = os.environ.get("SAVE", "1") == "1"
TAG = os.environ.get("TAG", "")  # e.g. "_step0" -> smoke_train_step0.json, out/m4/smoke_step0_orbax
R = {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()], "steps": STEPS, "seq": SEQ, "lr": LR,
     "method": METHOD, "nseq": NSEQ, "batch": 1}
proc = psutil.Process(); peak = [0]
def mon():
    while True:
        peak[0] = max(peak[0], proc.memory_info().rss); time.sleep(0.5)
threading.Thread(target=mon, daemon=True).start()
def gb(x): return round(x / 2**30, 2)
def dump(): json.dump(R, open(ROOT + f"/docs/Plan/m4/raw/smoke_train{TAG}.json", "w"), indent=1)
t0 = time.time()

# ---- 1. load + tree layout check
model = gm.nn.Gemma4_E2B(text_only=True)
tok0 = jnp.zeros((1, 8), jnp.int32)
abstract = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), tok0)["params"])
params = gm.ckpts.load_params(ORBAX, params=abstract, donate=False, text_only=True, quantize=True)
sd = lambda t: jax.tree.map(lambda a: (tuple(a.shape), str(a.dtype)), t)
R["load_s"] = round(time.time() - t0, 1)
R["tree_equal_to_model_init"] = jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(abstract) and sd(params) == sd(abstract)
fa = {jax.tree_util.keystr(k): (tuple(v.shape), str(v.dtype)) for k, v in jax.tree_util.tree_leaves_with_path(abstract)}
fl = {jax.tree_util.keystr(k): (tuple(v.shape), str(v.dtype)) for k, v in jax.tree_util.tree_leaves_with_path(params)}
R["tree_keys_equal"] = set(fa) == set(fl); R["tree_shapes_equal"] = all(fa[k][0] == fl[k][0] for k in fa if k in fl)
R["tree_dtype_diff"] = sorted({f"{fa[k][1]}->{fl[k][1]}" for k in fa if k in fl and fa[k][1] != fl[k][1]})
R["tree_key_diff"] = sorted(set(fa) ^ set(fl))[:6]
R["n_leaves"] = len(jax.tree_util.tree_leaves(params)); R["n_params"] = int(sum(a.size for a in jax.tree_util.tree_leaves(params)))
R["rss_after_load_gb"] = gb(proc.memory_info().rss); print("loaded", R, flush=True); dump()

# ---- 2. data
toks = np.load(ROOT + "/docs/Plan/m4/raw/train_tokens.npy")
def seq(i):
    s = toks[i * SEQ:(i + 1) * SEQ]; return np.concatenate([[2], s]).astype(np.int32)  # BOS + SEQ tokens
train = np.stack([seq(i) for i in range(NSEQ)]); held = np.stack([seq(NSEQ + 50)])

# ---- 3. model / loss
qat = gm.nn.QuantizationAwareWrapper(model=model, method=getattr(peft.QuantizationMethod, METHOD), log_replacements=True)
frozen = {"embedder": params["embedder"]}
trainable = jax.tree.map(lambda a: a.astype(jnp.float32), {k: v for k, v in params.items() if k != "embedder"})
del params
def ce(logits, tgt):
    lp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    return -jnp.take_along_axis(lp, tgt[..., None], -1).mean()
def loss_fn(tr, fr, x, m):
    out = m.apply({"params": {**fr, **tr}}, x[:, :-1])
    return ce(out.logits, x[:, 1:])
opt = optax.adafactor(learning_rate=LR)
opt_state = opt.init(trainable)
@jax.jit
def step(tr, st, fr, x):
    l, g = jax.value_and_grad(loss_fn)(tr, fr, x, qat)
    up, st = opt.update(g, st, tr); return optax.apply_updates(tr, up), st, l
eval_q = jax.jit(lambda tr, fr, x: loss_fn(tr, fr, x, qat))
eval_fp = jax.jit(lambda tr, fr, x: loss_fn(tr, fr, x, model))
R["n_wrapped_einsum"] = None
t1 = time.time(); R["held_loss_fp_before"] = float(eval_fp(trainable, frozen, held)); R["eval_compile_plus_run_s"] = round(time.time() - t1, 1)
R["held_loss_qat_before"] = float(eval_q(trainable, frozen, held))
R["train_loss_qat_before_seq0"] = float(eval_q(trainable, frozen, train[:1]))
print({k: R[k] for k in ("held_loss_fp_before", "held_loss_qat_before", "train_loss_qat_before_seq0")}, flush=True); dump()

# ---- 4. train
R["losses"] = []; R["step_s"] = []
for s in range(STEPS):
    ts = time.time()
    trainable, opt_state, l = step(trainable, opt_state, frozen, train[s % NSEQ][None])
    l = float(l); R["losses"].append(l); R["step_s"].append(round(time.time() - ts, 2))
    R["peak_rss_gb"] = gb(peak[0]); dump()
    print(f"step {s} loss {l:.4f} {R['step_s'][-1]}s peak {R['peak_rss_gb']}GB", flush=True)
if not STEPS: R["note"] = "STEPS=0: control = official IT snapped to the grid, no training"
R["held_loss_qat_after"] = float(eval_q(trainable, frozen, held))
R["held_loss_fp_after"] = float(eval_fp(trainable, frozen, held))
R["train_loss_qat_after_seq0"] = float(eval_q(trainable, frozen, train[:1]))
R["n_wrapped_einsum"] = sum(1 for _, k, _ in gq.LAST_REPLACEMENTS if k == "Einsum")
R["peak_rss_gb"] = gb(peak[0]); R["ru_maxrss_gb"] = gb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print({k: R[k] for k in ("held_loss_qat_after", "held_loss_fp_after", "train_loss_qat_after_seq0")}, flush=True); dump()

# ---- 5. export: snap to Q4 grid (same formula as training), orbax layout of the official ckpt
if SAVE:
    formula = {"Q4_0_ABSMAX": "absmax", "Q4_0_MINMAX": "minmax", "Q4_0": "q4_0"}[METHOD]
    full = {**frozen, **jax.tree.map(lambda a: a.astype(jnp.bfloat16), trainable)}
    del trainable, opt_state
    snapped = jax.jit(lambda p: _qat.snap_params(p, formula=formula))(full)
    snapped = jax.tree.map(lambda a: a.astype(jnp.bfloat16), snapped)
    def to_ckpt_layout(p):  # flax 'mlp/gating_einsum' (array) -> orbax 'mlp/gating_einsum/w'
        out = {}
        for k, v in p.items():
            if isinstance(v, dict): out[k] = to_ckpt_layout(v)
            elif k in ("gating_einsum", "linear"): out[k] = {"w": v}
            else: out[k] = v
        return out
    ck = to_ckpt_layout(snapped)
    flat = {".".join(str(getattr(q, "key", q)) for q in k) for k, _ in jax.tree_util.tree_leaves_with_path(ck)}
    meta = json.load(open(ORBAX + "/_METADATA"))["tree_metadata"]
    off = {".".join(ast.literal_eval(k)) for k in meta}
    off_text = {k for k in off if not k.startswith(("vision_encoder", "audio_encoder", "embedder.mm_", "embedder.audio_"))}
    R["export_keys_equal_official_text_keys"] = flat == off_text
    R["export_missing"] = sorted(off_text - flat)[:10]; R["export_extra"] = sorted(flat - off_text)[:10]
    outdir = ROOT + f"/out/m4/smoke{TAG}_orbax"
    if os.path.exists(outdir):
        import shutil; shutil.rmtree(outdir)
    c = ocp.StandardCheckpointer(); c.save(outdir, ck); c.wait_until_finished()
    R["export_dir"] = outdir
R["total_s"] = round(time.time() - t0, 1); R["peak_rss_gb"] = gb(peak[0]); dump(); print("DONE", flush=True)
