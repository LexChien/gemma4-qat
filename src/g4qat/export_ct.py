"""Backward-compatible entry: python -m g4qat.export_ct → g4qat.export.ct."""
from g4qat.export.ct import *  # noqa: F401
from g4qat.export.ct import main
if __name__ == "__main__":
    main()
