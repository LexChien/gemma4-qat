"""Full fine-tune (all params). Keep lr low (1e-5) to stay near the QAT-friendly weight grid,
otherwise 4-bit quality after re-quantization degrades. Mac: needs >=32GB unified memory (est.)."""
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from common import MODEL_ID, device, dtype_for
p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--out", default="out/full")
p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--max_len", type=int, default=1024)
p.add_argument("--freeze_embeddings", action="store_true", default=True)
a = p.parse_args()
dev = device()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype_for(dev)).to(dev)
if a.freeze_embeddings:  # PLE tables are huge (~2.8B params); freezing saves lots of memory
    for n, prm in model.named_parameters():
        if "embed" in n: prm.requires_grad = False
ds = load_dataset("json", data_files=a.data, split="train")
cfg = SFTConfig(output_dir=a.out, per_device_train_batch_size=1, gradient_accumulation_steps=16,
    learning_rate=a.lr, num_train_epochs=1, max_length=a.max_len, gradient_checkpointing=True,
    bf16=(dev == "cuda"), optim="adafactor", report_to=[], save_strategy="no")
SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok).train()
model.save_pretrained(a.out, safe_serialization=True); tok.save_pretrained(a.out)
