#!/usr/bin/env bash
# Apple Silicon setup. JAX on Mac: CPU backend works; jax-metal is experimental and
# historically lags JAX releases / lacks ops — UNVERIFIED for this repo. Expect CPU-speed QAT.
set -e
python3.12 -m venv .venv-jax && source .venv-jax/bin/activate
pip install -U pip
git clone --depth 1 https://github.com/google-deepmind/gemma gemma-src || true
pip install -e gemma-src          # pulls jax, flax, kauldron>=1.4.4, orbax-checkpoint
pip install safetensors kagglehub tensorflow-datasets mlx-lm
# Optional experimental GPU: pip install jax-metal   (then check: python -c "import jax;print(jax.devices())")
python -c "from gemma import gm, peft; print(gm.nn.Gemma4_E2B, peft.QuantizationMethod.Q4_0)"
