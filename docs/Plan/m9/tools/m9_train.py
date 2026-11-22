"""M9 lighter on-grid LoRA-QAT: SFT CE on assistant tokens (cleaned M9 set) + beta * KL(teacher || student) on general text
(wikitext-2 train + zh-wiki topic extracts, 512-token chunk per step). Teacher = the same network with LoRA bypassed; the base is the
official deployed-dense pack (a fixed point of Q), so the bypassed network IS the official QAT model -> no second model in memory.
Chunked lm_head/softcap/CE/KL (C=128) against Metal 0000000e GPU timeouts; micro-batch 1 grad accumulation.
Checkpoints out/m9/ckpt/step_{global:06d}_{lora,opt}.safetensors + .json (commit marker), NEVER pruned. Resume from latest json."""
import argparse, glob, json, math, os, random, sys, time
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
sys.path.insert(0, "docs/Plan/m8/tools")
import m8_qat_lora as Q
from m8_qat_lora import apply_qat_lora, load_pack, load_deployed_dense, lora_tree
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--batch", type=int, default=4); ap.add_argument("--lr", type=float, default=4e-5)
ap.add_argument("--warmup", type=int, default=15); ap.add_argument("--rank", type=int, default=8); ap.add_argument("--scale", type=float, default=2.0)
ap.add_argument("--beta", type=float, default=1.0); ap.add_argument("--save-every", type=int, default=25); ap.add_argument("--eval-every", type=int, default=50)
ap.add_argument("--max-steps", type=int, default=None); ap.add_argument("--ckpt", default="out/m9/ckpt"); ap.add_argument("--cache-gb", type=float, default=4.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--micro", type=int, default=1)
a = ap.parse_args()
R = "docs/Plan/m9/raw/"; os.makedirs(a.ckpt, exist_ok=True); mx.set_cache_limit(int(a.cache_gb * 2**30))
train = [json.loads(l) for l in open(R + "sft_train.jsonl")]; valid = [json.loads(l) for l in open(R + "sft_valid.jsonl")]
GEN = [json.loads(l)["ids"] for l in open(R + "general.jsonl")]
spe = math.ceil(len(train) / a.batch); total = a.max_steps or int(spe * a.epochs)
def batch_at(step):
    ep, k = divmod(step, spe); order = list(range(len(train))); random.Random(a.seed * 1000 + ep).shuffle(order)
    return [train[i] for i in order[k * a.batch:(k + 1) * a.batch]] or [train[order[0]]]
def collate(exs):
    L = max(len(e["ids"]) for e in exs); x = [[0] * L for _ in exs]; m = [[0.0] * L for _ in exs]
    for b, e in enumerate(exs):
        ids = e["ids"]; x[b][:len(ids)] = ids
        for t in range(e["n_prompt"] - 1, len(ids) - 1): m[b][t] = 1.0
    return mx.array(x), mx.array(m)
TEACH = [False]; _orig = Q.QATLoRALinear.__call__
def _call(self, x):
    if TEACH[0]:
        y = x @ self.weight.astype(x.dtype).T
        return y + self.bias if "bias" in self else y
    return _orig(self, x)
Q.QATLoRALinear.__call__ = _call
t0 = time.time()
model, tok = load("out/m5/qat-text-hf"); P = load_pack(); nd = load_deployed_dense(model, P); nq = apply_qat_lora(model, a.rank, a.scale, P=P); del P
grad_checkpoint(model.layers[0])
sched = optim.join_schedules([optim.linear_schedule(1e-7, a.lr, a.warmup), optim.cosine_decay(a.lr, max(1, total - a.warmup), a.lr * 0.1)], [a.warmup])
opt = optim.AdamW(learning_rate=sched, weight_decay=0.0); start = 0
ck = sorted(glob.glob(f"{a.ckpt}/step_*.json"))
if ck:
    meta = json.load(open(ck[-1])); start = meta["step"]; tag = ck[-1][:-5]
    model.load_weights(list(mx.load(tag + "_lora.safetensors").items()), strict=False)
    opt.init(model.trainable_parameters()); opt.state = tree_unflatten(list(mx.load(tag + "_opt.safetensors").items()))
    print(f"RESUME from global step {start} ({tag})", flush=True)
print(f"setup {time.time() - t0:.1f}s deployed-dense {nd} qat-lora {nq} rank {a.rank} trainable {sum(v.size for _, v in tree_flatten(model.trainable_parameters()))} "
      f"train {len(train)} valid {len(valid)} general {len(GEN)} steps/epoch {spe} total {total} lr {a.lr} beta {a.beta}", flush=True)
TM = getattr(model, "language_model", model)
def head(h):
    lo = TM.model.embed_tokens.as_linear(h) if TM.tie_word_embeddings else TM.lm_head(h)
    if TM.final_logit_softcapping is not None: lo = mx.tanh(lo / TM.final_logit_softcapping) * TM.final_logit_softcapping
    return lo.astype(mx.float32)
def loss_sft(model, x, m, denom, C=128):
    h = TM.model(x[:, :-1]); tgt = x[:, 1:]; mm = m[:, :-1]; tot = mx.array(0.0, mx.float32)
    for i in range(0, h.shape[1], C):
        tot = tot + (nn.losses.cross_entropy(head(h[:, i:i + C]), tgt[:, i:i + C]) * mm[:, i:i + C]).sum()
    return tot / denom, mm.sum()
def loss_kl(model, xg, ht, C=128):
    h = TM.model(xg[:, :-1]); tgt = xg[:, 1:]; n = h.shape[1]; kl = mx.array(0.0, mx.float32); ce = mx.array(0.0, mx.float32)
    for i in range(0, n, C):
        ls = head(h[:, i:i + C]); lt = mx.stop_gradient(head(ht[:, i:i + C]))
        lps = ls - mx.logsumexp(ls, axis=-1, keepdims=True); lpt = lt - mx.logsumexp(lt, axis=-1, keepdims=True)
        kl = kl + (mx.exp(lpt) * (lpt - lps)).sum()
        ce = ce + mx.stop_gradient(-mx.take_along_axis(lps, tgt[:, i:i + C, None], axis=-1).sum())
    return a.beta * kl / n, ce / n
lg_sft = nn.value_and_grad(model, loss_sft); lg_kl = nn.value_and_grad(model, loss_kl)
def add(g1, g2): return g2 if g1 is None else tree_unflatten([(k, v + w) for (k, v), (_, w) in zip(tree_flatten(g1), tree_flatten(g2))])
def evaluate():
    tot = n = 0.0
    for i in range(len(valid)):
        x, m = collate(valid[i:i + 1]); l, k = loss_sft(model, x, m, 1.0); mx.eval(l, k); tot += l.item(); n += k.item(); mx.clear_cache()
    return tot / n
def save(step):
    tag = f"{a.ckpt}/step_{step:06d}"
    mx.save_safetensors(tag + "_lora.safetensors", lora_tree(model)); mx.save_safetensors(tag + "_opt.safetensors", dict(tree_flatten(opt.state)))
    json.dump(dict(step=step, total=total, rank=a.rank, scale=a.scale, lr=a.lr, beta=a.beta, time=time.strftime("%F %T")), open(tag + ".json", "w"))
if start == 0: print(f"step 0 val {evaluate():.4f}", flush=True)
acc = dict(sft=0.0, kl=0.0, ce=0.0, n=0); tt = time.time(); ntok = 0
for step in range(start, total):
    exs = batch_at(step); denom = float(sum(e["n_target"] for e in exs)); g = None; ls = 0.0
    for j in range(0, len(exs), a.micro):
        x, m = collate(exs[j:j + a.micro]); (l, _), gg = lg_sft(model, x, m, denom); g = add(g, gg); mx.eval(g, l); ls += l.item(); mx.clear_cache()
    xg = mx.array([GEN[step % len(GEN)]])
    TEACH[0] = True; ht = mx.stop_gradient(TM.model(xg[:, :-1])); mx.eval(ht); TEACH[0] = False
    (kl, ce), gg = lg_kl(model, xg, ht); g = add(g, gg); mx.eval(g, kl, ce); mx.clear_cache()
    opt.update(model, g); mx.eval(model.trainable_parameters(), opt.state)
    acc["sft"] += ls; acc["kl"] += kl.item(); acc["ce"] += ce.item(); acc["n"] += 1; ntok += int(denom) + xg.shape[1]
    gs = step + 1
    if gs % 5 == 0 or gs == start + 1:
        el = time.time() - tt; n = acc["n"]
        print(f"step {gs}/{total} sft {acc['sft'] / n:.4f} kl {acc['kl'] / n:.4f} gen_ce {acc['ce'] / n:.3f} lr {sched(mx.array(step)).item():.2e} "
              f"tok/s {ntok / el:.0f} s/step {el / n:.1f} peak {mx.get_peak_memory() / 1e9:.1f}GB", flush=True)
        acc = dict(sft=0.0, kl=0.0, ce=0.0, n=0); tt = time.time(); ntok = 0
    if gs % a.save_every == 0 or gs == total: save(gs)
    if gs % a.eval_every == 0 or gs == total: print(f"step {gs} val {evaluate():.4f}", flush=True)
    mx.clear_cache()
open(R + "train.DONE", "w").write(time.strftime("%F %T")); print("TRAIN_DONE", flush=True)
