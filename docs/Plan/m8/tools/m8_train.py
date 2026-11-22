"""M8 SFT with on-grid mergeable LoRA-QAT in MLX (m8_qat_lora.py). Resumable with GLOBAL step numbering:
out/m8/ckpt/step_{global:06d}_{lora,opt}.safetensors + step_{global:06d}.json (step, lr state, data order seed).
Data order is a deterministic per-epoch permutation, so a resumed run sees exactly the batches it would have seen.
Exit code 0 + raw/train.DONE when finished; any Metal crash -> nonzero exit -> m8_train_loop.sh restarts from the latest ckpt."""
import argparse, glob, json, math, os, random, sys, time
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
sys.path.insert(0, os.path.dirname(__file__))
from m8_qat_lora import apply_qat_lora, load_pack, load_deployed_dense, lora_tree
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=float, default=2.0); ap.add_argument("--batch", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--rank", type=int, default=16); ap.add_argument("--scale", type=float, default=2.0)
ap.add_argument("--save-every", type=int, default=25); ap.add_argument("--eval-every", type=int, default=100); ap.add_argument("--max-steps", type=int, default=None)
ap.add_argument("--ckpt", default="out/m8/ckpt"); ap.add_argument("--cache-gb", type=float, default=4.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--micro", type=int, default=1, help="micro-batch size (grad accumulation) - keeps each Metal command buffer small")
a = ap.parse_args()
R = "docs/Plan/m8/raw/"; os.makedirs(a.ckpt, exist_ok=True)
mx.set_cache_limit(int(a.cache_gb * 2**30))
train = [json.loads(l) for l in open(R + "sft_train.jsonl")]; valid = [json.loads(l) for l in open(R + "sft_valid.jsonl")]
spe = math.ceil(len(train) / a.batch); total = a.max_steps or int(spe * a.epochs)
def batch_at(step):  # deterministic: epoch permutation seeded by (seed, epoch)
    ep, k = divmod(step, spe); order = list(range(len(train))); random.Random(a.seed * 1000 + ep).shuffle(order)
    return [train[i] for i in order[k * a.batch:(k + 1) * a.batch]] or [train[order[0]]]
def collate(exs):
    L = max(len(e["ids"]) for e in exs); x = mx.zeros((len(exs), L), mx.int32).tolist(); m = [[0.0] * L for _ in exs]
    for b, e in enumerate(exs):
        ids = e["ids"]; x[b][:len(ids)] = ids
        for t in range(e["n_prompt"] - 1, len(ids) - 1): m[b][t] = 1.0  # position t predicts token t+1 (target tokens)
    return mx.array(x), mx.array(m)
t0 = time.time()
model, tok = load("out/m5/qat-text-hf"); P = load_pack(); nd = load_deployed_dense(model, P); nq = apply_qat_lora(model, a.rank, a.scale, P=P); del P
grad_checkpoint(model.layers[0])
sched = optim.join_schedules([optim.linear_schedule(1e-7, a.lr, a.warmup), optim.cosine_decay(a.lr, max(1, total - a.warmup), a.lr * 0.1)], [a.warmup])
opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)
start = 0
ck = sorted(glob.glob(f"{a.ckpt}/step_*.json"))
if ck:
    meta = json.load(open(ck[-1])); start = meta["step"]; tag = ck[-1][:-5]
    model.load_weights(list(mx.load(tag + "_lora.safetensors").items()), strict=False)
    opt.init(model.trainable_parameters())
    opt.state = tree_unflatten(list(mx.load(tag + "_opt.safetensors").items()))
    print(f"RESUME from global step {start} ({tag})", flush=True)
print(f"setup {time.time() - t0:.1f}s deployed-dense {nd} qat-lora {nq} trainable {sum(v.size for _, v in tree_flatten(model.trainable_parameters()))} "
      f"train {len(train)} valid {len(valid)} steps/epoch {spe} total {total}", flush=True)
TM = getattr(model, "language_model", model)  # text Model (has .model, tie_word_embeddings, final_logit_softcapping)
def loss_fn(model, x, m, denom, C=128):
    # Chunked lm_head + CE: the 262k-vocab head (and its split-k backward) is the known source of Metal 0000000e GPU timeouts.
    h = TM.model(x[:, :-1]); tgt = x[:, 1:]; mm = m[:, :-1]; tot = mx.array(0.0, mx.float32)
    for i in range(0, h.shape[1], C):
        lo = TM.model.embed_tokens.as_linear(h[:, i:i + C]) if TM.tie_word_embeddings else TM.lm_head(h[:, i:i + C])
        if TM.final_logit_softcapping is not None: lo = mx.tanh(lo / TM.final_logit_softcapping) * TM.final_logit_softcapping
        tot = tot + (nn.losses.cross_entropy(lo.astype(mx.float32), tgt[:, i:i + C]) * mm[:, i:i + C]).sum()
    return tot / denom, mm.sum()
lg = nn.value_and_grad(model, loss_fn)
def evaluate():
    tot = n = 0.0
    for i in range(0, len(valid)):
        x, m = collate(valid[i:i + 1]); l, k = loss_fn(model, x, m, 1.0); mx.eval(l, k); tot += l.item(); n += k.item()
    return tot / n
def save(step):
    tag = f"{a.ckpt}/step_{step:06d}"
    mx.save_safetensors(tag + "_lora.safetensors", lora_tree(model))
    mx.save_safetensors(tag + "_opt.safetensors", dict(tree_flatten(opt.state)))
    json.dump(dict(step=step, total=total, time=time.strftime("%F %T")), open(tag + ".json", "w"))  # json last = commit marker
    for old in sorted(glob.glob(f"{a.ckpt}/step_*.json"))[:-3]:  # keep last 3
        o = old[:-5]; [os.remove(p) for p in (old, o + "_lora.safetensors", o + "_opt.safetensors") if os.path.exists(p)]
if start == 0: print(f"step 0 val {evaluate():.4f}", flush=True)
tl = tn = 0.0; tt = time.time(); ntok = 0
for step in range(start, total):
    exs = batch_at(step); denom = float(sum(e["n_target"] for e in exs)); gacc = None; lsum = 0.0
    for j in range(0, len(exs), a.micro):
        x, m = collate(exs[j:j + a.micro]); (l, k), g = lg(model, x, m, denom)
        gacc = g if gacc is None else tree_unflatten([(kk, v + w) for (kk, v), (_, w) in zip(tree_flatten(gacc), tree_flatten(g))])
        mx.eval(gacc, l); lsum += l.item(); mx.clear_cache()
    opt.update(model, gacc); mx.eval(model.trainable_parameters(), opt.state)
    tl += lsum; tn += 1; ntok += int(denom)
    gs = step + 1
    if gs % 10 == 0:
        el = time.time() - tt
        print(f"step {gs}/{total} loss {tl / tn:.4f} lr {sched(mx.array(step)).item():.2e} tgt_tok/s {ntok / el:.0f} peak {mx.get_peak_memory() / 1e9:.1f}GB", flush=True)
        tl = tn = 0.0; tt = time.time(); ntok = 0
    if gs % a.save_every == 0 or gs == total: save(gs)
    if gs % a.eval_every == 0 or gs == total: print(f"step {gs} val {evaluate():.4f}", flush=True)
    mx.clear_cache()
open(R + "train.DONE", "w").write(time.strftime("%F %T")); print("TRAIN_DONE", flush=True)
