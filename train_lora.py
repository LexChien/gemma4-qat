"""PEFT + TRL LoRA. Primary path: Mac MPS. Also works on CUDA.
python train_lora.py --data data.jsonl --out out/lora  (jsonl: {"messages":[{"role":..,"content":..}]})"""
import argparse, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from common import MODEL_ID, device, dtype_for
p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--out", default="out/lora")
p.add_argument("--r", type=int, default=16); p.add_argument("--epochs", type=float, default=1)
p.add_argument("--max_len", type=int, default=1024); p.add_argument("--lr", type=float, default=2e-4)
a = p.parse_args()
dev = device(); dt = dtype_for(dev)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
# text-only fine-tune; if AutoModelForCausalLM rejects the multimodal config, switch to AutoModelForMultimodalLM
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt).to(dev)
ds = load_dataset("json", data_files=a.data, split="train")
peft_cfg = LoraConfig(r=a.r, lora_alpha=2*a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=1,
    gradient_accumulation_steps=8, learning_rate=a.lr, max_length=a.max_len,
    bf16=(dev == "cuda"), gradient_checkpointing=True, logging_steps=10, save_strategy="epoch",
    report_to=[], dataloader_pin_memory=(dev == "cuda"))
SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg).train()
model.save_pretrained(a.out); tok.save_pretrained(a.out)
