"""Unsloth LoRA — CUDA (NVIDIA) only; Unsloth does not run on Apple MPS.
Gemma 4 support in Unsloth: verify on unsloth docs before use (UNVERIFIED)."""
import argparse
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from common import MODEL_ID
p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--out", default="out/lora_unsloth")
p.add_argument("--load_in_4bit", action="store_true", help="QLoRA (bitsandbytes, CUDA)")
a = p.parse_args()
model, tok = FastLanguageModel.from_pretrained(MODEL_ID, max_seq_length=2048, load_in_4bit=a.load_in_4bit)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth")
ds = load_dataset("json", data_files=a.data, split="train")
SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=SFTConfig(output_dir=a.out,
    per_device_train_batch_size=2, gradient_accumulation_steps=4, num_train_epochs=1,
    learning_rate=2e-4, bf16=True, report_to=[])).train()
model.save_pretrained(a.out); tok.save_pretrained(a.out)
