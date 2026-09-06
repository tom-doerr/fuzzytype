"""Argument parsing, and the promise that --help is instant."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from fuzzytype.cli import build_parser

SRC = Path(__file__).resolve().parents[1] / "src"


def test_defaults_describe_a_usable_run():
    args = build_parser().parse_args([])
    assert args.command is None  # no subcommand means the TUI
    assert args.layout == "colemak-dh"
    assert args.top > 0


def test_predict_subcommand_takes_context_and_keystrokes():
    args = build_parser().parse_args(
        ["predict", "--context", "I went to buy", "--typed", "brd"]
    )
    assert args.command == "predict"
    assert args.typed == "brd"


def test_unknown_layout_is_rejected_at_parse_time():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--layout", "dvorak-ish"])


def test_parsing_arguments_does_not_import_torch():
    """`fuzzytype --help` must not pay for a CUDA context.

    Importing torch here costs seconds and can fail outright on a fragmented
    machine, so the backend is imported inside the subcommand that needs it.
    Only a subprocess can prove the import never happened.
    """
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from fuzzytype.cli import build_parser;"
        "build_parser().parse_args(['--help'])" % str(SRC)
    )
    done = subprocess.run(
        [sys.executable, "-c", code + "\n"], capture_output=True, text=True
    )
    # --help exits 0 after printing usage.
    assert done.returncode == 0, done.stderr
    assert "fuzzytype" in done.stdout

    probe = (
        "import sys; sys.path.insert(0, %r);"
        "from fuzzytype.cli import build_parser;"
        "build_parser();"
        "sys.exit(1 if 'torch' in sys.modules else 0)" % str(SRC)
    )
    assert subprocess.run([sys.executable, "-c", probe]).returncode == 0
