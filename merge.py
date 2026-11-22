"""Merge LoRA adapter into base (bf16). Runs on Mac (CPU/MPS) or CUDA."""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from common import MODEL_ID
p = argparse.ArgumentParser(); p.add_argument("--adapter", required=True); p.add_argument("--out", default="out/merged")
a = p.parse_args()
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
m.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(a.out)
