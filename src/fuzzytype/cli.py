"""Command line entry points.

Argument parsing stays torch-free so ``fuzzytype --help`` is instant: the
backend is imported inside the subcommand that needs it, never at module
scope. A test asserts that.
"""

from __future__ import annotations

import argparse
import time

from .channel import LAYOUTS, ChannelCosts
from .engine import DEFAULT_PREAMBLE, Engine, EngineConfig
from .lm import DEFAULT_MODEL
from .search import PredictConfig
from .rank import DEFAULT_LENGTH_BONUS

__all__ = ["build_parser", "main"]


def _add_common(parser: argparse.ArgumentParser, *, suppress: bool = False) -> None:
    """Options that make sense both before and after the subcommand.

    They are declared twice: once on the top level with real defaults, and
    once on each subcommand with ``SUPPRESS`` defaults. Without SUPPRESS a
    subparser's default silently overwrites a value the user gave *before*
    the subcommand, so ``fuzzytype -k 4 predict`` would quietly ignore the 4.
    """
    def default(value):
        return argparse.SUPPRESS if suppress else value

    parser.add_argument("--model", default=default(DEFAULT_MODEL), help="HF model id")
    parser.add_argument(
        "--device", default=default(None), help="cuda, cpu (default: auto)"
    )
    parser.add_argument(
        "--layout",
        default=default("colemak-dh"),
        choices=sorted(LAYOUTS),
        help="keyboard layout, so neighbouring-key typos cost less",
    )
    parser.add_argument(
        "--length-bonus",
        type=float,
        default=default(DEFAULT_LENGTH_BONUS),
        help="nats per character added back to offset the prior's length penalty",
    )
    parser.add_argument(
        "-k", "--top", type=int, default=default(8), help="rows to show"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=default(64),
        help="longest candidate to decode",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=default(12),
        help="batched forward passes per decode; the wall-clock lever",
    )
    parser.add_argument(
        "--child-top-k",
        type=int,
        default=default(64),
        help=(
            "next tokens considered per node; generous because top-p truncates "
            "first, so it costs almost nothing and widens what is reachable"
        ),
    )
    parser.add_argument("--preamble", default=default(DEFAULT_PREAMBLE))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fuzzytype",
        description=(
            "Predictive text input: a base LM decodes likely continuations, "
            "a noisy-channel fuzzy match decides which one you are typing."
        ),
    )
    _add_common(parser)
    common = argparse.ArgumentParser(add_help=False)
    _add_common(common, suppress=True)

    sub = parser.add_subparsers(dest="command")

    p_predict = sub.add_parser(
        "predict",
        parents=[common],
        help="rank the strings a set of keystrokes may have meant",
    )
    p_predict.add_argument("--context", default="", help="text already written")
    p_predict.add_argument("--typed", default="", help="keystrokes since the last word")
    p_predict.add_argument(
        "--stats", action="store_true", help="also print what the search did"
    )

    p_bench = sub.add_parser(
        "bench", parents=[common], help="time a decode and a keystroke re-rank"
    )
    p_bench.add_argument("--context", default="I went to the shop to buy some")
    p_bench.add_argument("--typed", default="")
    p_bench.add_argument("--repeat", type=int, default=3)

    return parser


def _engine(args) -> Engine:
    from .hf import HFLanguageModel  # imported here to keep --help torch-free

    lm = HFLanguageModel(args.model, device=args.device)
    return Engine(
        lm=lm,
        config=EngineConfig(
            preamble=args.preamble, k=args.top, length_bonus=args.length_bonus
        ),
        predict_config=PredictConfig(
            k=max(args.top * 20, 160),
            max_chars=args.max_chars,
            max_rounds=args.max_rounds,
            child_top_k=args.child_top_k,
        ),
        costs=ChannelCosts.for_layout(args.layout),
    )


def _print_table(suggestions, coverage: float) -> None:
    if not suggestions:
        print("(no candidate explains those keystrokes within the error budget)")
        return
    print(f"{'#':>2}  {'P(meant)':>8}  {'prior':>9}  {'-logP(keys)':>11}  suggestion")
    print(f"{'-' * 2}  {'-' * 8}  {'-' * 9}  {'-' * 11}  {'-' * 30}")
    for i, s in enumerate(suggestions, 1):
        shown = f"{s.matched}|{s.predicted}" if s.consumed else s.text
        print(
            f"{i:>2}  {s.probability:>7.1%}  {s.prior_display:>9}  "
            f"{s.cost:>11.2f}  {shown}"
        )
    print(f"\ncoverage of found mass: {coverage:.1%}")


def cmd_predict(args) -> int:
    engine = _engine(args)
    engine.text = args.context
    t0 = time.time()
    stats = engine.refresh(args.typed)
    decode_s = time.time() - t0
    t1 = time.time()
    suggestions, coverage = engine.suggest(args.typed)
    rerank_ms = (time.time() - t1) * 1000

    print(f"context: {args.context!r}")
    print(f"typed:   {args.typed!r}\n")
    _print_table(suggestions, coverage)
    print(f"\ndecode {decode_s:.2f}s   re-rank {rerank_ms:.1f}ms")
    if args.stats:
        print("search:", stats.as_dict())
    return 0


def cmd_bench(args) -> int:
    engine = _engine(args)
    engine.text = args.context
    for i in range(args.repeat):
        t0 = time.time()
        stats = engine.refresh(args.typed)
        decode = time.time() - t0
        t1 = time.time()
        for n in range(1, 9):
            engine.suggest(args.typed[:n] or "")
        rerank = (time.time() - t1) / 8 * 1000
        print(
            f"run {i + 1}: decode {decode:.2f}s "
            f"({stats.rounds} rounds, {stats.expansions} expansions, "
            f"{stats.distinct} candidates, {stats.pruned_by_channel} pruned) "
            f"re-rank {rerank:.1f}ms/keystroke  pool={len(engine.pool)}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "predict":
        return cmd_predict(args)
    if args.command == "bench":
        return cmd_bench(args)
    from .tui import run_tui

    return run_tui(args)
