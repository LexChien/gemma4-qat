"""M5-A: JAX LoRA-QAT on the OFFICIAL QAT weights (HF qat-q4_0-unquantized -> JAX), mergeable formulation
y = x·Q(W + s·AB) (gm.nn.gemma4._qat_lora.QATLoRA), embed/PLE frozen (still fake-quantized), CPU.
Resumable: checkpoints LoRA+optimizer to out/m5/jax_lora{TAG}/ckpt.pkl every CKPT_EVERY steps.
Env: METHOD (Q4_0_MINMAX) RANK (16) ALPHA (32) LR (2e-4) SEQ (256) BATCH (1) STEPS (100000) BUDGET_S (3600)
     EVAL_EVERY (25) CKPT_EVERY (10) TAG ('') EXPORT (1)
Writes docs/Plan/m5/raw/jax_lora{TAG}.json; export: out/m5/jax_lora{TAG}/orbax (merged + snapped Q4_0_MINMAX)"""
import json, os, pickle, sys, time, threading
import numpy as np, psutil
import jax, jax.numpy as jnp, optax
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")); sys.path.insert(0, ROOT + "/tools")
from convert_hf_to_jax import load_hf_text_params
from gemma import gm, peft
from gemma.gm.nn.gemma4 import _qat, _qat_lora
E = lambda k, d: type(d)(os.environ.get(k, d))
METHOD = E("METHOD", "Q4_0_MINMAX"); RANK = E("RANK", 16); ALPHA = E("ALPHA", 32.0); LR = E("LR", 2e-4); SEQ = E("SEQ", 256)
BATCH = E("BATCH", 1); STEPS = E("STEPS", 100000); BUDGET = E("BUDGET_S", 3600.0); EVAL_EVERY = E("EVAL_EVERY", 25)
CKPT_EVERY = E("CKPT_EVERY", 10); TAG = E("TAG", ""); EXPORT = E("EXPORT", "1") == "1"
OUT = ROOT + f"/out/m5/jax_lora{TAG}"; os.makedirs(OUT, exist_ok=True); RJ = ROOT + f"/docs/Plan/m5/raw/jax_lora{TAG}.json"
proc = psutil.Process(); peak = [0]
threading.Thread(target=lambda: [peak.__setitem__(0, max(peak[0], proc.memory_info().rss)) or time.sleep(0.5) for _ in iter(int, 1)], daemon=True).start()
gb = lambda x: round(x / 2**30, 2)
R = json.load(open(RJ)) if os.path.exists(RJ) and os.path.exists(OUT + "/ckpt.pkl") else {
    "method": METHOD, "rank": RANK, "alpha": ALPHA, "lr": LR, "seq": SEQ, "batch": BATCH, "budget_s": BUDGET,
    "base": "weights/gemma-4-E2B-it-qat-q4_0-unquantized", "losses": [], "val": [], "step_s": []}
dump = lambda: json.dump(R, open(RJ, "w"), indent=1)
t0 = time.time()
base_np = load_hf_text_params(ROOT + "/weights/gemma-4-E2B-it-qat-q4_0-unquantized", ROOT + "/weights/orbax/checkpoints/gemma4-e2b-it")
base = jax.tree.map(jnp.asarray, base_np); del base_np
R["load_s"] = round(time.time() - t0, 1)
model = gm.nn.Gemma4_E2B(text_only=True)
m = _qat_lora.QATLoRA(model=model, rank=RANK, alpha=ALPHA, method=getattr(peft.QuantizationMethod, METHOD))
tok0 = jnp.zeros((1, 8), jnp.int32)
lo_shape, _ = _qat_lora.split_lora(jax.eval_shape(lambda: m.init(jax.random.PRNGKey(0), tok0)["params"]))
def init_lora(key):
    leaves = jax.tree_util.tree_leaves_with_path(lo_shape); keys = jax.random.split(key, len(leaves))
    def mk(path, s, k):
        n = jax.tree_util.keystr(path)
        return (jax.random.normal(k, s.shape, jnp.float32) / np.sqrt(s.shape[0])) if "lora_a" in n else jnp.zeros(s.shape, jnp.float32)
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(lo_shape), [mk(p, s, k) for (p, s), k in zip(leaves, keys)])
def merge(a, b):
    out = dict(b)
    for k, v in a.items(): out[k] = merge(v, b.get(k, {})) if isinstance(v, dict) else v
    return out
toks = np.load(ROOT + "/docs/Plan/m5/raw/train_tokens.npy"); nval = 64 * (SEQ + 1)
tr_toks, va_toks = toks[:-nval], toks[-nval:]
BOS = 2
val = np.concatenate([np.full((4, 1), BOS, np.int32), va_toks[: 4 * SEQ].reshape(4, SEQ)], 1)  # BOS + SEQ tokens
def ce(logits, tgt):
    lp = jax.nn.log_softmax(logits.astype(jnp.float32), -1); return -jnp.take_along_axis(lp, tgt[..., None], -1).mean()
def loss_fn(lo, bs, x): return ce(m.apply({"params": merge(lo, bs)}, x[:, :-1]).logits, x[:, 1:])
sched = optax.warmup_cosine_decay_schedule(0.0, LR, 20, 2000, LR * 0.1)
opt = optax.adamw(sched, weight_decay=0.0)
@jax.jit
def step(lo, st, bs, x):
    l, g = jax.value_and_grad(loss_fn)(lo, bs, x); up, st = opt.update(g, st, lo); return optax.apply_updates(lo, up), st, l
eval_l = jax.jit(loss_fn)
eval_fp = jax.jit(lambda bs, x: ce(model.apply({"params": bs}, x[:, :-1]).logits, x[:, 1:]))
def val_loss(lo): return float(np.mean([float(eval_l(lo, base, val[i:i + 1])) for i in range(len(val))]))
if os.path.exists(OUT + "/ckpt.pkl"):
    ck = pickle.load(open(OUT + "/ckpt.pkl", "rb")); lora = jax.tree.map(jnp.asarray, ck["lora"]); st = jax.tree.map(jnp.asarray, ck["opt"])
    s0, rng = ck["step"], np.random.RandomState(); rng.set_state(ck["rng"]); print("resumed at step", s0, flush=True)
    R.setdefault("resumes", []).append(s0)
else:
    lora = init_lora(jax.random.PRNGKey(0)); st = opt.init(lora); s0 = 0; rng = np.random.RandomState(0)
    R["n_lora_params"] = int(sum(a.size for a in jax.tree_util.tree_leaves(lora)))
    tv = time.time(); R["val_step0"] = val_loss(lora); R["val_eval_s"] = round(time.time() - tv, 1); R["val_step0_noquant"] = float(np.mean([float(eval_fp(base, val[i:i + 1])) for i in range(len(val))])); R["val"].append([0, R["val_step0"]])
    print("val step0", R["val_step0"], flush=True); dump()
def save(s):
    pickle.dump({"lora": jax.tree.map(np.asarray, lora), "opt": jax.tree.map(np.asarray, st), "step": s, "rng": rng.get_state()},
                open(OUT + "/ckpt.tmp", "wb")); os.replace(OUT + "/ckpt.tmp", OUT + "/ckpt.pkl")
tstart = time.time(); s = s0; elapsed0 = R.get("train_s", 0.0)
while s < STEPS and elapsed0 + (time.time() - tstart) < BUDGET:
    idx = rng.randint(0, len(tr_toks) - SEQ - 1, BATCH); x = np.stack([np.concatenate([[BOS], tr_toks[i:i + SEQ]]) for i in idx]).astype(np.int32)  # Gemma needs BOS at pos 0
    ts = time.time(); lora, st, l = step(lora, st, base, x); l = float(l); s += 1
    R["losses"].append(l); R["step_s"].append(round(time.time() - ts, 2)); R["train_s"] = elapsed0 + (time.time() - tstart)
    R["steps_done"] = s; R["tokens_seen"] = s * BATCH * SEQ; R["peak_rss_gb"] = gb(peak[0])
    if s % EVAL_EVERY == 0: R["val"].append([s, val_loss(lora)]); print("val", R["val"][-1], flush=True)
    if s % CKPT_EVERY == 0: save(s)
    dump(); print(f"step {s} loss {l:.4f} {R['step_s'][-1]}s peak {R['peak_rss_gb']}GB", flush=True)
save(s); R["val_final"] = val_loss(lora); R["val"].append([s, R["val_final"]]); dump()
if EXPORT:
    import orbax.checkpoint as ocp, shutil
    full = merge(lora, base); del base
    merged = _qat_lora.merge_lora(full, rank=RANK, alpha=ALPHA)
    snapped = jax.tree.map(lambda a: a.astype(jnp.bfloat16), _qat.snap_params(merged, formula="minmax"))
    d = OUT + "/orbax"
    if os.path.exists(d): shutil.rmtree(d)
    c = ocp.StandardCheckpointer(); c.save(d, snapped); c.wait_until_finished(); R["export_dir"] = d
R["total_s"] = round(time.time() - t0, 1); R["peak_rss_gb"] = gb(peak[0]); dump(); print("DONE", flush=True)
