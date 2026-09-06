"""Orchestration: what is committed, what is cached, and when to decode again.

The interactive loop has two clocks. A decode takes ~1 s; a keystroke has
~20 ms before it feels laggy. So this holds a *pool* of candidates decoded in
the background and answers every keystroke from it, going back to the model
only when the pool stops explaining what is being typed.

The refresh trigger is the channel cost itself. If the best candidate in the
pool explains the keystrokes for near zero nats, the pool is still right and
no GPU work is needed. Once the cheapest explanation gets expensive -- a rare
word, a name, a turn the model did not anticipate -- that is the signal to
decode again, this time *with* the keystrokes as evidence so the search is
steered toward what is actually being typed. The typist keeps seeing the old
pool while that runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .channel import ChannelCosts
from .lm import LanguageModel
from .search import Candidate, PredictConfig, PredictStats, predict
from .rank import DEFAULT_LENGTH_BONUS, Suggestion, rerank

__all__ = ["EngineConfig", "Engine", "DEFAULT_PREAMBLE"]

#: A base model continues text; with an empty document it has nothing to
#: continue. What it is given matters more than it looks. Measured on this
#: model with an empty document: a bare newline produces Java import
#: statements, and a *description* of the task ("The following is a note
#: written in plain English") is continued by describing the task further --
#: it came back at 99.9% confidence repeating its own preamble. Two sentences
#: of ordinary prose, ended at a sentence boundary, are continued the way a
#: person would continue them.
DEFAULT_PREAMBLE = (
    "I have been meaning to write this down for a while. The week went by "
    "quickly and there is a lot to catch up on. "
)


@dataclass
class EngineConfig:
    preamble: str = DEFAULT_PREAMBLE
    #: Displayed rows.
    k: int = 8
    length_bonus: float = DEFAULT_LENGTH_BONUS
    #: Channel cost, in nats, above which the pool no longer explains the
    #: keystrokes well enough and a fresh decode is worth ~1 s of GPU.
    refresh_cost: float = 1.0
    #: Context fed to the model, in tokens. Bounds the cost of every forward.
    max_context_tokens: int = 192


@dataclass
class Engine:
    """Committed text, the cached candidate pool, and the policy between them."""

    lm: LanguageModel
    config: EngineConfig = field(default_factory=EngineConfig)
    predict_config: PredictConfig = field(default_factory=PredictConfig)
    costs: ChannelCosts = field(default_factory=ChannelCosts)
    text: str = ""
    pool: list[Candidate] = field(default_factory=list)
    #: The keystrokes the pool was decoded under; "" means unconstrained.
    pool_query: str = ""
    last_stats: PredictStats | None = None

    def context_ids(self) -> list[int]:
        ids = self.lm.encode(self.config.preamble + self.text)
        limit = self.config.max_context_tokens
        return ids[-limit:] if len(ids) > limit else ids

    def seeds(self, query: str) -> list[str]:
        """Literal texts the search should start from as well as the root.

        Only this layer knows whether a leading space belongs in front of the
        keystrokes, because only it knows what has been committed.
        """
        if not query:
            return []
        lead = "" if self._at_word_start() else " "
        variants = [query]
        # A typist does not reach for shift on a name. The channel forgives the
        # case when *ranking*, but a lowercase seed can only ever grow into a
        # lowercase word -- "alic" never reaches "Alice" -- so the capitalised
        # spelling has to be offered to the search as its own starting path.
        if query[:1].islower():
            variants.append(query[:1].upper() + query[1:])
        return [lead + v for v in variants]

    def refresh(
        self,
        query: str = "",
        on_partial: "Callable[[PredictStats], None] | None" = None,
        should_stop: "Callable[[], bool] | None" = None,
    ) -> PredictStats:
        """Decode a fresh pool. Slow; call off the UI thread.

        The pool is replaced as the search finds things rather than only at
        the end, so a longer decode shows more suggestions instead of a longer
        wait. ``should_stop`` lets a keystroke abandon a decode that has
        already been overtaken.
        """

        def publish(candidates: list[Candidate], stats: PredictStats) -> None:
            self.pool = candidates
            self.pool_query = query
            self.last_stats = stats
            if on_partial is not None:
                on_partial(stats)

        candidates, stats = predict(
            self.lm,
            self.context_ids(),
            query=query,
            config=self.predict_config,
            costs=self.costs,
            seeds=self.seeds(query),
            on_candidates=publish,
            should_stop=should_stop,
        )
        self.pool = candidates
        self.pool_query = query
        self.last_stats = stats
        return stats

    def suggest(self, query: str, k: int | None = None) -> tuple[list[Suggestion], float]:
        """Rank the cached pool against the keystrokes. Fast; safe on the UI thread."""
        return rerank(
            self.pool,
            query,
            self.costs,
            length_bonus=self.config.length_bonus,
            k=self.config.k if k is None else k,
        )

    def needs_refresh(self, query: str) -> bool:
        """True when the pool no longer explains what is being typed."""
        if not self.pool:
            return True
        suggestions, _ = self.suggest(query, k=1)
        if not suggestions:
            return True
        return suggestions[0].cost > self.config.refresh_cost

    def commit(self, raw: str) -> str:
        """Accept a suggestion. Returns the text inserted.

        Candidates carry the leading space that joins them to the previous
        word, so it has to come off again at the very start of a document.
        """
        if not self.text:
            raw = raw.lstrip()
        self.text += raw
        self.pool = []
        self.pool_query = ""
        return raw

    def commit_literal(self, typed: str) -> str:
        """Accept exactly what was typed, correcting nothing.

        The leading space is supplied here because candidates carry their own
        and the typist never types one at a word boundary.
        """
        return self.commit(typed if self._at_word_start() else " " + typed)

    def _at_word_start(self) -> bool:
        """True when the committed text already ends where a word may start."""
        return self.text == "" or self.text[-1].isspace()

    def backspace_text(self) -> None:
        """Delete one character of committed text and invalidate the pool."""
        if self.text:
            self.text = self.text[:-1]
            self.pool = []
            self.pool_query = ""
