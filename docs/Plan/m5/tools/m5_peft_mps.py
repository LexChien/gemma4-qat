"""M5-B: HF PEFT LoRA (PyTorch MPS, bf16) on the official QAT weights (text-only), no fake-quant (plain LoRA).
Resumable (adapter + optimizer + step every CKPT_EVERY). Then merge_and_unload -> out/m5/peft{TAG}-merged-hf.
Env: RANK 16 ALPHA 32 LR 1e-4 SEQ 512 BATCH 2 BUDGET_S 2700 CKPT_EVERY 50 EVAL_EVERY 100 TAG ''"""
import json, os, time, math
import numpy as np, torch, psutil
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
E = lambda k, d: type(d)(os.environ.get(k, d))
RANK = E("RANK", 16); ALPHA = E("ALPHA", 32); LR = E("LR", 1e-4); SEQ = E("SEQ", 512); BATCH = E("BATCH", 2)
BUDGET = E("BUDGET_S", 2700.0); CKPT = E("CKPT_EVERY", 50); EVAL_EVERY = E("EVAL_EVERY", 100); TAG = E("TAG", "")
OUT = ROOT + f"/out/m5/peft{TAG}"; os.makedirs(OUT, exist_ok=True); RJ = ROOT + f"/docs/Plan/m5/raw/peft{TAG}.json"
dev = torch.device("mps"); torch.manual_seed(0)
R = json.load(open(RJ)) if os.path.exists(OUT + "/state.pt") else {"rank": RANK, "alpha": ALPHA, "lr": LR, "seq": SEQ, "batch": BATCH,
    "budget_s": BUDGET, "base": "out/m5/qat-text-hf (official QAT, bf16)", "losses": [], "val": [], "step_s": []}
dump = lambda: json.dump(R, open(RJ, "w"), indent=1)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(ROOT + "/out/m5/qat-text-hf", dtype=getattr(torch, os.environ.get("DTYPE", "bfloat16"))).to(dev)
model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
cfg = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
if os.path.exists(OUT + "/adapter/adapter_config.json"):
    model = PeftModel.from_pretrained(model, OUT + "/adapter", is_trainable=True)
else:
    model = get_peft_model(model, cfg)
R["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad); R["load_s"] = round(time.time() - t0, 1)
toks = np.load(ROOT + "/docs/Plan/m5/raw/train_tokens.npy"); nval = 64 * (SEQ + 1); tr, va = toks[:-nval], toks[-nval:]
BOS = 2
val = torch.tensor(np.concatenate([np.full((4, 1), BOS), va[: 4 * (SEQ - 1)].reshape(4, SEQ - 1)], 1).astype(np.int64))
params = [p for p in model.parameters() if p.requires_grad]
for p_ in params: p_.data = p_.data.float()  # fp32 LoRA weights (bf16 AdamW -> NaN at step 3, attempt 1)
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
sched = lambda s: LR * min(1.0, s / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / 2000))))
rng = np.random.RandomState(0); s = 0
if os.path.exists(OUT + "/state.pt"):
    stt = torch.load(OUT + "/state.pt", weights_only=False); opt.load_state_dict(stt["opt"]); s = stt["step"]; rng.set_state(stt["rng"])
    R.setdefault("resumes", []).append(s); print("resumed", s, flush=True)
@torch.no_grad()
def val_loss():
    model.eval(); ls = [float(model(input_ids=val[i:i+1].to(dev), labels=val[i:i+1].to(dev)).loss) for i in range(len(val))]; model.train(); return float(np.mean(ls))
if s == 0: R["val"].append([0, val_loss()]); print("val0", R["val"][-1], flush=True)
def save():
    model.save_pretrained(OUT + "/adapter"); torch.save({"opt": opt.state_dict(), "step": s, "rng": rng.get_state()}, OUT + "/state.tmp"); os.replace(OUT + "/state.tmp", OUT + "/state.pt")
model.train(); proc = psutil.Process(); tstart = time.time(); el0 = R.get("train_s", 0.0)
while el0 + time.time() - tstart < BUDGET:
    idx = rng.randint(0, len(tr) - SEQ - 1, BATCH); x = torch.tensor(np.stack([np.concatenate([[BOS], tr[i:i + SEQ - 1]]) for i in idx]).astype(np.int64)).to(dev)
    ts = time.time()
    for g in opt.param_groups: g["lr"] = sched(s)
    loss = model(input_ids=x, labels=x).loss
    if not torch.isfinite(loss): R["nonfinite_steps"] = R.get("nonfinite_steps", 0) + 1; opt.zero_grad(set_to_none=True); s += 1; continue
    loss.backward(); gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    if not torch.isfinite(gn): R["nonfinite_grad_steps"] = R.get("nonfinite_grad_steps", 0) + 1; opt.zero_grad(set_to_none=True); s += 1; continue
    opt.step(); opt.zero_grad(set_to_none=True); torch.mps.synchronize()
    s += 1; R["losses"].append(float(loss.detach())); R["step_s"].append(round(time.time() - ts, 2)); R["steps_done"] = s; R["tokens_seen"] = s * BATCH * SEQ
    R["train_s"] = el0 + time.time() - tstart; R["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
    R["peak_rss_gb"] = max(R.get("peak_rss_gb", 0), round(proc.memory_info().rss / 2**30, 2))
    if s % EVAL_EVERY == 0: R["val"].append([s, val_loss()]); print("val", R["val"][-1], flush=True)
    if s % CKPT == 0: save()
    if s % 10 == 0: dump(); print(f"step {s} loss {float(loss):.4f} {R['step_s'][-1]}s mps {R['mps_driver_gb']}GB", flush=True)
save(); R["val"].append([s, val_loss()]); dump()
merged = model.merge_and_unload(); merged.save_pretrained(OUT + "-merged-hf", safe_serialization=True)
import shutil
for x in os.listdir(ROOT + "/out/m5/qat-text-hf"):
    if x.endswith((".json", ".jinja")) and x not in ("config.json",): shutil.copy(ROOT + "/out/m5/qat-text-hf/" + x, OUT + "-merged-hf/" + x)
R["total_s"] = round(time.time() - t0, 1); dump(); print("DONE", flush=True)
