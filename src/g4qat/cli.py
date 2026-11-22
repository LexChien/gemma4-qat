"""g4qat CLI. C-1: export-ct, eval-hf, and wrappers around the existing M0/M7 scripts.
Old scripts are not moved; this just forwards argv so existing pipelines keep working.
Usage: g4qat <command> ...
"""
from __future__ import annotations
import argparse, os, runpy, sys

def repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))  # .../src/g4qat
    cand = os.path.abspath(os.path.join(here, "..", ".."))
    if os.path.isdir(os.path.join(cand, "docs", "Plan")):
        return cand
    return os.getcwd()

def _run_path(rel: str, argv: list[str]) -> int:
    root = repo_root()
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        print(f"g4qat: missing script {path}", file=sys.stderr)
        return 2
    # Make sibling imports (e.g. m6_chat_prompts) work, and keep cwd at the repo root.
    sys.path.insert(0, os.path.dirname(path))
    os.chdir(root)
    sys.argv = [path] + list(argv)
    runpy.run_path(path, run_name="__main__")
    return 0

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="g4qat", description="Gemma 4 E2B QAT toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="print package version")
    e = sub.add_parser("export-ct", help="MLX exact-grid pack → compressed-tensors (G3)")
    e.add_argument("src"); e.add_argument("out"); e.add_argument("--ple-group", type=int, default=256)
    h = sub.add_parser("eval-hf", help="transformers load / PPL / 50-prompt chat")
    h.add_argument("model"); h.add_argument("tag"); h.add_argument("outdir")
    h.add_argument("--device", default="mps"); h.add_argument("--no-ppl", action="store_true")
    h.add_argument("--no-chat", action="store_true"); h.add_argument("--limit", type=int, default=None)
    for name, help_ in (("eval-ppl", "M0 MLX PPL + speed (wraps m0_bench_mlx.py)"),
                        ("bench", "alias for eval-ppl"),
                        ("eval-chat", "M7 latency / chat (wraps m7_latency.py)"),
                        ("grade", "M7 grade (wraps m7_grade.py)")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("rest", nargs=argparse.REMAINDER)

    # Parse only the subcommand + its known flags; remainder goes to wrapped scripts.
    if not argv:
        p.print_help(); return 0
    cmd = argv[0]
    rest = argv[1:]
    if cmd in {"-h", "--help"}:
        p.print_help(); return 0
    if cmd == "version":
        from g4qat import __version__
        print(__version__); return 0
    if cmd == "export-ct":
        sys.argv = ["g4qat-export-ct"] + rest
        from g4qat.export.ct import main as m; m(); return 0
    if cmd == "eval-hf":
        sys.argv = ["g4qat-eval-hf"] + rest
        from g4qat.eval.hf import main as m; m(); return 0
    if cmd in {"eval-ppl", "bench"}:
        return _run_path("docs/Plan/m0/tools/m0_bench_mlx.py", rest)
    if cmd == "eval-chat":
        return _run_path("docs/Plan/m7/tools/m7_latency.py", rest)
    if cmd == "grade":
        return _run_path("docs/Plan/m7/tools/m7_grade.py", rest)
    p.error(f"unknown command {cmd!r}")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
