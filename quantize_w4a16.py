"""compressed-tensors route (llm-compressor). Output targets vLLM/CUDA.
--recipe w4a16  : Google's official qat-w4a16-ct scheme (Linear int4, group 32, symmetric;
                  embeddings/PLE left bf16 -> official file is 8.32GB; NOT <3GB)
--recipe compact: + embed_tokens & lm_head int4 channel, embed_tokens_per_layer int4 group 256
                  (same as Google's qat-mobile-ct group_2), vision/audio ignored  -> est. ~2.4-2.9GB
Embedding quantization with llm-compressor is UNVERIFIED (Google's mobile-ct proves the
compressed-tensors FORMAT supports it; whether oneshot() targets nn.Embedding must be tested).
RTN (QuantizationModifier) is used; weights are already QAT-snapped, so GPTQ is optional."""
import argparse
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--out", default="out/ct")
p.add_argument("--recipe", choices=["w4a16", "compact"], default="compact")
a = p.parse_args()
W = lambda **k: QuantizationArgs(num_bits=4, type="int", symmetric=True, **k)
groups = {"linear": QuantizationScheme(targets=["re:model\\.language_model\\.layers\\..*"],
                                       weights=W(strategy="group", group_size=32))}
ignore = ["re:.*vision.*", "re:.*audio.*", "re:.*embed_vision.*", "re:.*embed_audio.*"]
if a.recipe == "compact":
    groups["embed"] = QuantizationScheme(targets=["model.language_model.embed_tokens", "re:.*lm_head$"],
                                         weights=W(strategy="channel"))
    groups["ple"] = QuantizationScheme(targets=["model.language_model.embed_tokens_per_layer"],
                                       weights=W(strategy="group", group_size=256))
else:
    ignore += ["lm_head"]
oneshot(model=a.model, recipe=QuantizationModifier(config_groups=groups, ignore=ignore), output_dir=a.out)
