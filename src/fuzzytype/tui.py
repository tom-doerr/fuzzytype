"""The interactive loop.

The point of the interface is that you can *steer*. Every keystroke re-ranks
the candidates immediately and shows both what the system now thinks you mean
and how well it is matching you -- so when the match is weak you type another
letter or two, and when it is strong you stop typing and take the sentence.
That feedback only works if the two numbers behind the ranking are on screen,
so each row shows its probability and its match quality rather than a single
opaque score.

Two clocks, as everywhere in this package. Re-ranking the cached pool is pure
Python and runs on the UI thread in about a millisecond. Decoding a new pool
takes on the order of a second and runs in a thread worker, so typing never
blocks on the GPU; the previous pool stays on screen and simply gets better
when the new one lands. The model itself loads in the background too -- you
can start typing during the twenty seconds it takes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Markdown, Static

from .channel import ChannelCosts
from .engine import Engine, EngineConfig
from .rank import Suggestion
from .search import PredictConfig, PredictStats

__all__ = ["FuzzyTypeApp", "run_tui", "quality_label"]

#: How the channel cost reads in words. The typist steers on this: "exact"
#: means stop typing, "loose" means give the system another letter.
_QUALITY = (
    (0.01, "exact"),
    (1.0, "case"),
    (4.0, "1 slip"),
    (8.0, "2 slips"),
    (13.0, "loose"),
)

HELP = """\
# fuzzytype

A base language model decodes the likely continuations of what you have
written. Your keystrokes are read as a *noisy transmission* of one of them.
The two combine into `P(you meant this)`, which ranks the list and also
decides which branches are worth decoding further.

## The idea

You are not completing a prefix, you are **describing** the sentence you
want. Type as much or as little as you like:

| You type | You get |
| --- | --- |
| `clo` | `clothes` -- an ordinary prefix |
| `clth` | `clothes` -- letters left out |
| `brd` | `bread` -- an abbreviation |
| `gt bck` | `get back to you as soon as possible` -- a whole sentence |
| `aprico` | `apricots` -- a word the model would never have guessed |

Watch the **match** column. `exact` means the system has you and you can
stop typing. `loose` means it is stretching to explain your keystrokes --
add a letter, or delete one that was a typo.

## Keys

| Key | Does |
| --- | --- |
| letters, space | type -- space is part of your query, not a commit |
| `up` / `down` | move the selection |
| `enter` / `tab` | accept the selected suggestion |
| `alt+1` ... `alt+9` | accept that row directly |
| `left` / `right` | move the cursor; past the keystrokes it walks into the text you have already accepted, and the suggestions follow it |
| `home` / `end` | jump to either end of the keystrokes |
| `backspace` / `delete` | delete behind or ahead of the cursor |
| `ctrl+l` | commit exactly what you typed, uncorrected |
| `ctrl+s` | cycle how much longer sentences are preferred |
| `ctrl+up` / `ctrl+down` (or `f3` / `f2`) | think harder or less hard -- more rounds finds more and longer phrases, and costs time |
| `ctrl+r` | force a fresh decode |
| `f1` | this help |
| `ctrl+q` | quit |

## What the columns mean

* **P(meant)** -- posterior over the candidates that were found, so it is a
  share of what the search saw, not of all English.
* **match** -- the channel cost in words. Exact costs nothing; every slip or
  omitted letter costs nats, and enough of them prune the branch entirely.
* **prior** -- `log P(text | context)` from the model alone, before your
  keystrokes are taken into account.
"""

#: ctrl+s cycles these: nats per character added back to offset the prior's
#: bias towards short candidates.
_LENGTH_BONUSES = (0.0, 1.5, 3.0)
_LENGTH_LABELS = ("words", "phrases", "sentences")

#: Animated while the model is decoding, so it is obvious that more
#: suggestions are still on their way rather than the list being final.
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

#: How hard to think, in batched forward passes per decode. Adjustable live
#: because it is the one setting whose right value depends entirely on what
#: is being written: a single word needs a handful of rounds, a long sentence
#: from heavy shorthand wants all of them.
_ROUND_STEPS = (4, 8, 12, 20, 32, 48, 80, 120)

#: Spinner period, and how many of those to skip while the model loads.
#: Repainting during the load was suspected of starving the loader thread,
#: but measured over both orderings the same configuration varies more
#: (18.9-21.9s) than any configuration differs from another -- the apparent
#: effect was a cold page cache on whichever ran first. The throttle stays
#: because a once-a-second elapsed counter is all a load needs, not because
#: it buys time.
_TICK_SECONDS = 0.2
_LOADING_TICKS = 5


def quality_label(cost: float) -> str:
    """Plain words for a channel cost, so the typist can act on it."""
    for limit, label in _QUALITY:
        if cost < limit:
            return label
    return "stretch"


class HelpScreen(ModalScreen):
    BINDINGS = [("escape,f1,q", "dismiss", "close")]

    def compose(self) -> ComposeResult:
        yield Markdown(HELP)


@dataclass
class _Status:
    loading: bool = True
    decoding: bool = False
    stats: PredictStats | None = None
    coverage: float = 0.0
    error: str = ""


class FuzzyTypeApp(App):
    """Type loosely; pick the sentence you meant."""

    CSS = """
    Screen { layers: base; }
    #document {
        height: auto; min-height: 3; padding: 1 2;
        border: round $primary; background: $surface;
    }
    #suggestions { height: 1fr; border: round $accent; }
    #status { height: 1; padding: 0 2; color: $text-muted; }
    HelpScreen { align: center middle; }
    HelpScreen > Markdown {
        width: 80%; height: 80%; padding: 1 2;
        border: round $accent; background: $surface;
    }
    """

    BINDINGS = [
        ("enter,tab", "accept", "accept"),
        ("up", "move(-1)", "up"),
        ("down", "move(1)", "down"),
        ("left", "cursor(-1)", "left"),
        ("right", "cursor(1)", "right"),
        ("home", "cursor_end(0)", ""),
        ("end", "cursor_end(1)", ""),
        ("delete", "delete", ""),
        ("ctrl+l", "accept_literal", "literal"),
        ("ctrl+s", "cycle_length", "length"),
        ("ctrl+up,f3", "rounds(1)", "+think"),
        ("ctrl+down,f2", "rounds(-1)", "-think"),
        ("ctrl+r", "force_refresh", "re-decode"),
        ("f1", "help", "help"),
        ("ctrl+q", "quit", "quit"),
    ] + [(f"alt+{n}", f"accept_row({n - 1})", "") for n in range(1, 10)]

    query: reactive[str] = reactive("")
    selected: reactive[int] = reactive(0)
    #: Where in the keystrokes the cursor sits. The keystrokes themselves sit
    #: at the document cursor, so walking off either end of them moves the
    #: document cursor instead -- left and right simply keep going.
    cursor: reactive[int] = reactive(0)

    def __init__(self, engine: Engine, model_loader=None) -> None:
        super().__init__()
        self.engine = engine
        self._model_loader = model_loader
        self._status = _Status(loading=model_loader is not None)
        self._suggestions: list[Suggestion] = []
        self._decoding = False
        self._decode_wanted = False
        #: Set to abandon the running decode. A keystroke the current pool
        #: cannot explain matters more than finishing a stale search.
        self._abandon = threading.Event()
        # Start the cycle where the configuration already is, so ctrl+s moves
        # away from the user's chosen default rather than resetting it.
        self._tick = 0
        self._bonus_index = min(
            range(len(_LENGTH_BONUSES)),
            key=lambda i: abs(_LENGTH_BONUSES[i] - engine.config.length_bonus),
        )

    # -- layout ----------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(id="document")
            yield DataTable(id="suggestions", cursor_type="row", zebra_stripes=True)
            yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(_TICK_SECONDS, self._tick_spinner)
        table = self.query_one("#suggestions", DataTable)
        table.add_columns("#", "P(meant)", "match", "suggestion")
        # This is an input method: the app owns every key. A focusable table
        # would otherwise swallow enter and the arrows for its own cursor.
        table.can_focus = False
        self.title = "fuzzytype"
        self.sub_title = "type loosely, pick the sentence you meant"
        if self._model_loader is not None:
            self._load_model()
        else:
            self._request_decode()
        self._render()

    def _tick_spinner(self) -> None:
        """Advance the spinner, but only repaint while something is running."""
        if self._decoding:
            self._tick += 1
            self._render_status()
        elif self._status.loading:
            self._tick += 1
            if self._tick % _LOADING_TICKS == 0:
                self._render_status()

    # -- background work -------------------------------------------------
    @work(thread=True, group="load")
    def _load_model(self) -> None:
        try:
            lm = self._model_loader()
        except Exception as exc:  # surfaced, never swallowed
            self.call_from_thread(self._model_failed, str(exc))
            return
        self.call_from_thread(self._model_ready, lm)

    def _model_ready(self, lm) -> None:
        self.engine.lm = lm
        self._status.loading = False
        self._request_decode()
        self._render()

    def _model_failed(self, message: str) -> None:
        self._status.loading = False
        self._status.error = message
        self._render()

    def _request_decode(self) -> None:
        """Ask for a fresh pool, one at a time, abandoning anything stale."""
        if self._status.loading or self._status.error:
            return
        if self._decoding:
            # Do not queue behind a search whose answer is already out of
            # date -- tell it to stop and start again from the new query.
            self._decode_wanted = True
            self._abandon.set()
            return
        self._decoding = True
        self._abandon.clear()
        self._status.decoding = True
        self._status.stats = None
        self._decode(self.query)
        self._render()

    def _request_rescore(self) -> None:
        """Re-price the pool under the new keystrokes, one pass at a time."""
        if self._decoding:
            self._decode_wanted = True
            self._abandon.set()
            return
        self._decoding = True
        self._abandon.clear()
        self._status.decoding = True
        self._rescore(self.query)
        self._render()

    @work(thread=True, group="decode")
    def _rescore(self, query: str) -> None:
        try:
            self.engine.rescore(query)
        except Exception as exc:
            self.call_from_thread(self._decode_failed, str(exc))
            return
        self.call_from_thread(self._rescore_done)

    def _rescore_done(self) -> None:
        self._decoding = False
        self._status.decoding = False
        self._render()
        # Nothing new was discovered, only re-priced, so look for more once
        # the typist has stopped -- interrupted again by the next keystroke.
        if self._decode_wanted:
            self._decode_wanted = False
            self._after_typing()
        else:
            self._request_decode()

    @work(thread=True, group="decode")
    def _decode(self, query: str) -> None:
        try:
            stats = self.engine.refresh(
                query,
                on_partial=lambda st: self.call_from_thread(self._decode_partial, st),
                should_stop=self._abandon.is_set,
            )
        except Exception as exc:
            self.call_from_thread(self._decode_failed, str(exc))
            return
        self.call_from_thread(self._decode_done, stats)

    def _decode_partial(self, stats: PredictStats) -> None:
        """A batch of new suggestions arrived while the search continues."""
        self._status.stats = stats
        self._render()

    def _decode_done(self, stats: PredictStats) -> None:
        self._decoding = False
        self._status.decoding = False
        self._status.stats = stats
        self._render()
        if self._decode_wanted:
            self._decode_wanted = False
            self._request_decode()

    def _decode_failed(self, message: str) -> None:
        self._decoding = False
        self._status.decoding = False
        self._status.error = message
        self._render()

    # -- typing ----------------------------------------------------------
    def on_key(self, event) -> None:
        """Own the keyboard: this is an input method, not a form field."""
        if event.key == "backspace":
            if self.cursor > 0:
                self.query = self.query[: self.cursor - 1] + self.query[self.cursor :]
                self.cursor -= 1
            else:
                self.engine.backspace_text()
                self._request_decode()
            self._after_typing()
            event.stop()
            return
        if event.is_printable and event.character:
            self.query = (
                self.query[: self.cursor] + event.character + self.query[self.cursor :]
            )
            self.cursor += 1
            self._after_typing()
            event.stop()

    def _after_typing(self) -> None:
        self.selected = 0
        self._render()
        if self._status.loading:
            return
        if self.engine.config.mode == "prompt" and self.engine.pool:
            # Prompt mode's priors are conditioned on the keystrokes, so a new
            # character invalidates the numbers but not the strings. Re-pricing
            # what is already known is far cheaper than searching for it again
            # -- measured at 2.7x -- and a decode follows once the typist
            # pauses, to keep finding new phrases.
            self._request_rescore()
            return
        # Channel mode's priors do not depend on the keystrokes, so the pool
        # only has to be extended when it stops explaining them.
        if self.engine.needs_refresh(self.query):
            self._request_decode()

    # -- actions ---------------------------------------------------------
    def action_cursor(self, delta: int) -> None:
        """Move the cursor, walking into committed text at either end.

        Stepping past the keystrokes moves the document cursor, so the
        suggestions become suggestions for *that* point in the sentence --
        going back to fix an earlier word predicts from there rather than
        from the end.
        """
        target = self.cursor + delta
        if 0 <= target <= len(self.query):
            self.cursor = target
        elif delta < 0:
            if self.engine.move_left():
                self._request_decode()
        elif self.engine.move_right():
            self._request_decode()
        self._render()

    def action_cursor_end(self, end: int) -> None:
        self.cursor = len(self.query) if end else 0
        self._render()

    def action_delete(self) -> None:
        """Delete forwards: the rest of the keystrokes first, then the text."""
        if self.cursor < len(self.query):
            self.query = self.query[: self.cursor] + self.query[self.cursor + 1 :]
            self._after_typing()
            return
        self.engine.delete_text()
        self._request_decode()
        self._render()

    def action_move(self, delta: int) -> None:
        if not self._suggestions:
            return
        self.selected = max(0, min(len(self._suggestions) - 1, self.selected + delta))
        self._render()

    def action_accept(self) -> None:
        self.action_accept_row(self.selected)

    def action_accept_row(self, index: int) -> None:
        """Accept a suggestion, consuming only the keystrokes it accounts for.

        A long shorthand is taken in pieces. Typing the whole sentence and
        accepting "this is a test" leaves "ofthenewtextinputsystem" still
        typed, ready for the next suggestion, rather than throwing away the
        part that has not been read yet.
        """
        if index >= len(self._suggestions):
            return
        chosen = self._suggestions[index]
        self.engine.commit(chosen.raw)
        self.query = self.query[chosen.keystrokes :]
        self.cursor = 0
        self.selected = 0
        self._request_decode()
        self._render()

    def action_accept_literal(self) -> None:
        if not self.query:
            return
        self.engine.commit_literal(self.query)
        self.query = ""
        self.cursor = 0
        self.selected = 0
        self._request_decode()
        self._render()

    def action_rounds(self, delta: int) -> None:
        """Spend more or less time searching, from now on.

        Applies to the next decode and starts one immediately, so the effect
        is visible rather than deferred to the next keystroke.
        """
        config = self.engine.predict_config
        steps = list(_ROUND_STEPS)
        current = min(range(len(steps)), key=lambda i: abs(steps[i] - config.max_rounds))
        index = max(0, min(len(steps) - 1, current + delta))
        if steps[index] == config.max_rounds:
            return
        config.max_rounds = steps[index]
        self._request_decode()
        self._render()

    def action_cycle_length(self) -> None:
        self._bonus_index = (self._bonus_index + 1) % len(_LENGTH_BONUSES)
        self.engine.config.length_bonus = _LENGTH_BONUSES[self._bonus_index]
        self._render()

    def action_force_refresh(self) -> None:
        """Decode again -- or retry the model, if loading it failed.

        Worth having on this hardware: a fragmented host can refuse a CUDA
        context, and an app whose only response is a permanent error line has
        to be restarted for something that usually succeeds on a second try.
        """
        if self._status.error and self._model_loader is not None:
            self._status.error = ""
            self._status.loading = True
            self._tick = 0
            self._load_model()
            self._render()
            return
        self._request_decode()

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    # -- rendering -------------------------------------------------------
    def _render(self) -> None:
        self._suggestions, self._status.coverage = (
            self.engine.suggest(self.query) if self.engine.pool else ([], 0.0)
        )
        if self.selected >= len(self._suggestions):
            self.selected = max(0, len(self._suggestions) - 1)
        self._render_document()
        self._render_table()
        self._render_status()

    def _render_document(self) -> None:
        engine = self.engine
        if not engine.text and not engine.after and not self.query:
            self.query_one("#document", Static).update(
                Text("start typing...", style="dim italic")
            )
            return
        body = Text(engine.text, style="dim")
        body.append(self.query[: self.cursor], style="bold yellow")
        body.append("▏", style="bold #ffcc00 reverse")
        body.append(self.query[self.cursor :], style="bold yellow")
        body.append(engine.after, style="dim")
        self.query_one("#document", Static).update(body)

    def _render_table(self) -> None:
        table = self.query_one("#suggestions", DataTable)
        table.clear()
        tail = self.engine.text[-48:]
        for i, s in enumerate(self._suggestions):
            line = Text()
            if tail:
                line.append(tail, style="dim")
                line.append(" ")
            # The typed part and the predicted part are visually separated so
            # it is obvious how much of the sentence is being guessed for you.
            line.append(s.matched, style="bold underline")
            line.append(s.predicted, style="bold green")
            table.add_row(
                Text(str(i + 1), style="dim"),
                Text(f"{s.probability:6.1%}", style=_bar_style(s.probability)),
                Text(quality_label(s.cost), style=_cost_style(s.cost)),
                line,
            )
        if self._suggestions:
            table.move_cursor(row=self.selected)

    def _render_status(self) -> None:
        self.query_one("#status", Static).update(self._status_line())

    def _status_line(self) -> Text:
        """The state line. The typist needs to know whether more is coming."""
        stats = self._status.stats
        line = Text()
        if self._status.error:
            line.append(f"error: {self._status.error}", style="bold red")
        elif self._status.loading:
            waited = int(self._tick * _TICK_SECONDS)
            line.append(f"● loading model {waited}s", style="bold yellow")
            line.append("  (you can type already)", style="dim")
        elif self._status.decoding:
            rounds = stats.rounds if stats else 0
            total = self.engine.predict_config.max_rounds
            line.append(
                f"{_SPINNER[self._tick % len(_SPINNER)]} thinking",
                style="bold cyan",
            )
            line.append(f" {rounds}/{total}", style="cyan")
            if stats is not None:
                line.append(f"  {stats.distinct} found so far", style="dim")
        else:
            line.append("✓ done", style="bold green")
            if stats is not None:
                line.append(
                    f"  {stats.distinct} phrases in {stats.seconds_total:.1f}s",
                    style="dim",
                )
        line.append(
            f"   |   prefer {_LENGTH_LABELS[self._bonus_index]}", style="dim"
        )
        line.append(
            f"   |   think {self.engine.predict_config.max_rounds}", style="dim"
        )
        line.append(f"   |   {self.engine.config.mode}", style="dim")
        if self._suggestions:
            line.append(f"   |   coverage {self._status.coverage:.0%}", style="dim")
        line.append("   |   f1 help", style="dim")
        return line


def _bar_style(probability: float) -> str:
    if probability >= 0.5:
        return "bold green"
    if probability >= 0.15:
        return "yellow"
    return "dim"


def _cost_style(cost: float) -> str:
    if cost < 1.0:
        return "green"
    if cost < 8.0:
        return "yellow"
    return "red"


def run_tui(args) -> int:
    """Build the engine and hand it to the app, loading the model lazily."""

    def loader():
        from .hf import HFLanguageModel

        return HFLanguageModel(args.model, device=args.device)

    engine = Engine(
        lm=None,  # filled in by the loader worker
        config=EngineConfig(
            preamble=args.preamble,
            k=args.top,
            length_bonus=args.length_bonus,
            mode=args.mode,
            channel_assist=args.channel_assist,
            channel_weight=args.channel_weight,
        ),
        predict_config=PredictConfig(
            k=max(args.top * 20, 160),
            max_chars=args.max_chars,
            max_rounds=args.max_rounds,
            child_top_k=args.child_top_k,
        ),
        costs=ChannelCosts.for_layout(args.layout),
    )
    FuzzyTypeApp(engine, model_loader=loader).run()
    return 0
