"""Backward-compatible entry: python -m g4qat.hf_eval → g4qat.eval.hf."""
from g4qat.eval.hf import *  # noqa: F401
from g4qat.eval.hf import main
if __name__ == "__main__":
    main()
