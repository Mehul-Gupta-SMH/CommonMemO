"""The examples run, offline, and show what they say they show."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run(name: str) -> str:
    done = subprocess.run([sys.executable, str(EXAMPLES / name)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_plain_python():
    out = run("plain_python.py")
    assert "I'm vegetarian" in out and "dmy_or_mdy_date -> iso_date" in out


def test_harness_hooks():
    assert "make ship" in run("harness_hooks.py")


def test_google_adk_through_commonadk():
    pytest.importorskip("commonadk")
    out = run("google_adk.py")
    assert "only buy vegan" in out and "Notes from past calls of this tool" in out
