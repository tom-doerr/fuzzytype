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


def test_common_options_work_on_either_side_of_the_subcommand():
    """`fuzzytype predict -k 4` and `fuzzytype -k 4 predict` must agree.

    argparse silently loses the second form unless the subparser's copies
    default to SUPPRESS: the subparser's own default overwrites whatever was
    parsed before the subcommand.
    """
    parser = build_parser()
    assert parser.parse_args(["predict", "-k", "4"]).top == 4
    assert parser.parse_args(["-k", "4", "predict"]).top == 4
    assert parser.parse_args(["predict"]).top == 8  # the shared default


def test_search_width_defaults_match_the_library():
    """A narrower CLI default silently under-searched: "th wthr hs bn" found
    "The week has been" instead of "The weather has been"."""
    from fuzzytype.search import PredictConfig

    args = build_parser().parse_args([])
    assert args.child_top_k == PredictConfig().child_top_k
    assert args.max_rounds == PredictConfig().max_rounds
    assert args.max_chars == PredictConfig().max_chars
