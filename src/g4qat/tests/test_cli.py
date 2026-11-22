"""CLI smoke: --help and version; does not load models."""
import subprocess, sys, shutil, os

def _g4qat():
    exe = shutil.which("g4qat")
    if exe:
        return [exe]
    return [sys.executable, "-m", "g4qat.cli"]

def test_help_and_version():
    r = subprocess.run(_g4qat() + ["--help"], capture_output=True, text=True, check=True)
    assert "export-ct" in r.stdout and "eval-hf" in r.stdout
    r = subprocess.run(_g4qat() + ["version"], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "0.1.0"

def test_export_ct_help():
    r = subprocess.run(_g4qat() + ["export-ct", "--help"], capture_output=True, text=True, check=True)
    assert "--ple-group" in r.stdout
