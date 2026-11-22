import torch
MODEL_ID = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
def device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"
def dtype_for(dev):
    # MPS: bf16 supported on macOS 14+; fp32 fallback is safest for training stability
    return torch.bfloat16 if dev in ("cuda", "mps") else torch.float32
