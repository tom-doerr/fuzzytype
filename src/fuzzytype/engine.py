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
from dataclasses import dataclass, field, replace

from .channel import ChannelCosts
from .lm import LanguageModel
from .shorthand import DEFAULT_EXAMPLES, Example, build_prompt
from .search import Candidate, PredictConfig, PredictStats, predict
from .rank import DEFAULT_LENGTH_BONUS, Suggestion, rerank

__all__ = ["EngineConfig", "Engine", "DEFAULT_PREAMBLE"]

#: No primer. A forward pass needs at least one token, so an empty document
#: starts from the model's own document-boundary token instead of invented
#: prose -- and what gets suggested comes from looking up the vocabulary for
#: what was actually typed, which needs no register cue at all: from a bare
#: document start, "thi" retrieves "This", "thing", "this", "third", "think".
#:
#: Any prose put here is a prior on everything that follows, and a strong one.
#: A diary-ish opener made "This is a test" *less* likely than "This is my
#: latest update", so the phrase could not surface however long the search
#: ran. Set it deliberately or not at all.
DEFAULT_PREAMBLE = ""


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
    #: Drop the preamble once the document itself is this long, in characters.
    #: The preamble exists only to give the model something to continue when
    #: there is nothing; kept beyond that it is just an arbitrary prior on
    #: everything that follows. It decides the first sentence outright -- a
    #: diary-ish opener made "This is a test" *less* likely than "This is my
    #: latest update" and no amount of searching could surface it -- so it
    #: should stop applying as soon as the writing can speak for itself.
    preamble_until: int = 80
    #: How many candidates to keep across decodes. Bigger means less is
    #: rediscovered, but every keystroke re-scores the whole pool.
    max_pool: int = 600
    #: How many known candidates to hand the next decode as starting paths.
    #: Off by default on measurement: typing "gt bck" one character at a time,
    #: accumulating the pool alone gave 267 candidates in 16.8s, while adding
    #: twelve resume seeds gave 293 in 20.7s -- a quarter more GPU for a tenth
    #: more candidates, because each seed is priced by its own forward pass.
    resume_seeds: int = 0
    #: "channel" ranks by a hand-calibrated typing model; "prompt" asks the
    #: language model to expand the shorthand itself. See fuzzytype.shorthand.
    mode: str = "channel"
    #: In prompt mode, also apply the channel when ranking. On by default: the
    #: prompt alone does not insist that a candidate account for *all* the
    #: keystrokes, so "thiisatest" came back as "this test" at 57% with
    #: nothing to say the rest had been ignored -- and with no channel cost
    #: there is no match quality to show and nothing to highlight either.
    channel_assist: bool = True
    #: How much of the channel to apply when assisting. The prompt has already
    #: seen the shorthand, so a full-strength channel counts it twice and
    #: literal echoes win: at weight 1.0 "thisisatest" and "theisatest" take
    #: 12% and 8%. At 0.6 the ranking prefers "this is test" and "this is a
    #: test" while the echoes stay down.
    channel_weight: float = 0.6
    #: Prompt mode's context window. The prompt carries worked examples, so it
    #: needs considerably more room than a bare continuation.
    max_prompt_tokens: int = 640
    #: How many candidates a re-pricing pass covers. The cost is linear in
    #: this -- 465 candidates took 10.1s, against 27.3s to re-walk for them --
    #: so the pool is trimmed to its most probable entries rather than
    #: re-pricing a long tail nobody will see.
    max_rescore: int = 160


@dataclass
class Engine:
    """Committed text, the cached candidate pool, and the policy between them."""

    lm: LanguageModel
    config: EngineConfig = field(default_factory=EngineConfig)
    predict_config: PredictConfig = field(default_factory=PredictConfig)
    costs: ChannelCosts = field(default_factory=ChannelCosts)
    #: Committed text to the *left* of the cursor. This is what the model
    #: continues from, so moving the cursor changes what gets predicted.
    text: str = ""
    #: Committed text to the right of the cursor. Carried along untouched: a
    #: causal model cannot condition on it, so it is preserved rather than
    #: predicted around.
    after: str = ""
    pool: list[Candidate] = field(default_factory=list)
    #: The keystrokes the pool was decoded under; "" means unconstrained.
    pool_query: str = ""
    last_stats: PredictStats | None = None

    #: Worked examples used by prompt mode.
    examples: tuple = DEFAULT_EXAMPLES

    def prefix_ids(self, query: str) -> list[int]:
        """The tokens the search continues from.

        In channel mode that is simply the document. In prompt mode it is a
        shorthand-expansion prompt ending at "full text:", so the model's own
        next-token distribution is already conditioned on the keystrokes and
        no hand-built error model is involved.
        """
        if self.config.mode != "prompt":
            return self.context_ids()
        prompt = build_prompt(self.text[-240:].rstrip(" \t"), query, self.examples)
        ids = self.lm.encode(prompt)
        limit = self.config.max_prompt_tokens
        return ids[-limit:] if len(ids) > limit else ids

    def context_ids(self) -> list[int]:
        # A context ending in a space is off-distribution for this tokenizer:
        # the space belongs to the *following* token, so " bread" is one token
        # and a context already holding the space forces the rarer "bread".
        # Measured on "...to buy some ", the model's likeliest continuations
        # become digits -- "1", "2", "3" -- and word fragments; with the space
        # removed they are " new", " more", " clothes". The space is dropped
        # here and supplied by the candidate, which carries its own.
        preamble = (
            self.config.preamble
            if len(self.text) < self.config.preamble_until
            else ""
        )
        ids = self.lm.encode((preamble + self.text).rstrip(" \t"))
        if not ids:
            # Nothing written yet: begin where the model believes a document
            # begins, rather than in the middle of invented prose.
            ids = [self.lm.document_start_id]
        limit = self.config.max_context_tokens
        return ids[-limit:] if len(ids) > limit else ids

    def seeds(self, query: str) -> list[str]:
        """Literal texts the search should start from as well as the root.

        Includes what has already been found. A candidate's prior is
        ``P(text | context)`` and does not depend on the keystrokes at all, so
        everything discovered under a previous query is still valid -- handing
        the best of it back means the next decode extends "get back to you"
        into "get back to you as soon as possible" instead of rediscovering
        the first four words.

        Only this layer knows whether a leading space belongs in front of the
        keystrokes, because only it knows what has been committed.
        """
        resume = [c.raw for c in self.pool[: self.config.resume_seeds] if c.raw]
        if not query:
            return resume
        lead = "" if self._at_word_start() else " "
        variants = [query]
        # A typist does not reach for shift on a name. The channel forgives the
        # case when *ranking*, but a lowercase seed can only ever grow into a
        # lowercase word -- "alic" never reaches "Alice" -- so the capitalised
        # spelling has to be offered to the search as its own starting path.
        if query[:1].islower():
            variants.append(query[:1].upper() + query[1:])
        return [lead + v for v in variants] + resume

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
            self.pool = self._merged_pool(candidates)
            self.pool_query = query
            self.last_stats = stats
            if on_partial is not None:
                on_partial(stats)

        # Prompt mode puts the keystrokes in the prompt instead, so the
        # search runs unconstrained and the model does the fuzzy work.
        search_query = "" if self.config.mode == "prompt" else query
        candidates, stats = predict(
            self.lm,
            self.prefix_ids(query),
            query=search_query,
            config=self.predict_config,
            costs=self.costs,
            seeds=[] if self.config.mode == "prompt" else self.seeds(query),
            on_candidates=publish,
            should_stop=should_stop,
        )
        self.pool = self._merged_pool(candidates)
        self.pool_query = query
        self.last_stats = stats
        return stats

    def rescore(self, query: str) -> int:
        """Re-price everything already found under the new keystrokes.

        Prompt mode's priors depend on the shorthand, so a changed keystroke
        invalidates them -- but only their *numbers*, not the strings. Pricing
        the known candidates against the new prompt is a handful of batched
        forward passes, against a full walk of many seconds, which is what
        makes it practical to keep typing into a long sentence rather than
        rediscovering it from nothing at every character.

        Returns how many were re-priced. Anything past ``max_rescore`` is
        dropped rather than left carrying a price from an older keystroke.
        """
        if self.config.mode != "prompt" or not self.pool:
            return 0
        prefix = tuple(self.prefix_ids(query))
        if not prefix:
            return 0
        items, targets = [], []
        for candidate in self.pool[: self.config.max_rescore]:
            tokens = tuple(self.lm.encode(candidate.raw))
            if tokens:
                items.append((prefix, tokens))
                targets.append(candidate)
        if not items:
            return 0
        logprobs = self.lm.sequence_logprobs(items)
        self.pool = sorted(
            (replace(c, logprob=lp) for c, lp in zip(targets, logprobs)),
            key=lambda c: -c.logprob,
        )
        self.pool_query = query
        return len(items)

    def _merged_pool(self, found: list[Candidate]) -> list[Candidate]:
        """Accumulate rather than replace.

        Priors do not depend on the keystrokes *in channel mode*, so a
        candidate found under an earlier query is still exactly as probable
        now. Throwing the pool away on every edit meant re-deriving the same
        phrases from nothing each time, which is what made changing one
        character cost a whole decode -- and what put a ceiling on how long a
        sentence could be built up.

        Prompt mode's priors do depend on the keystrokes, so there the pool is
        kept but re-priced; see :meth:`rescore`.
        """
        best: dict[str, Candidate] = {c.text: c for c in self.pool}
        for candidate in found:
            known = best.get(candidate.text)
            if known is None or candidate.logprob > known.logprob:
                best[candidate.text] = candidate
        pool = sorted(best.values(), key=lambda c: -c.logprob)
        return pool[: self.config.max_pool]

    def suggest(self, query: str, k: int | None = None) -> tuple[list[Suggestion], float]:
        """Rank the cached pool against the keystrokes. Fast; safe on the UI thread."""
        prompt_mode = self.config.mode == "prompt"
        scored = query if not prompt_mode or self.config.channel_assist else ""
        return rerank(
            self.pool,
            scored,
            self.costs,
            length_bonus=self.config.length_bonus,
            k=self.config.k if k is None else k,
            # The channel is the only account of the keystrokes in channel
            # mode, and a partial second opinion in prompt mode.
            cost_weight=self.config.channel_weight if prompt_mode else 1.0,
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
        elif self.text.endswith((" ", "\t")) and raw.startswith(" "):
            # The cursor can sit just after a space; candidates carry their own.
            raw = raw.lstrip(" ")
        self.text += raw
        self._invalidate()
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
            self._invalidate()

    def delete_text(self) -> None:
        """Delete the character just after the cursor."""
        if self.after:
            self.after = self.after[1:]

    def move_left(self) -> bool:
        """Step the cursor one character back through committed text.

        The character moves from the left side to the right side, which is
        what makes the prediction follow the cursor: the model continues from
        whatever is now behind it, so going back to fix an earlier word gives
        suggestions for *that* point in the sentence rather than the end.
        """
        if not self.text:
            return False
        self.text, self.after = self.text[:-1], self.text[-1] + self.after
        self._invalidate()
        return True

    def move_right(self) -> bool:
        if not self.after:
            return False
        self.text, self.after = self.text + self.after[0], self.after[1:]
        self._invalidate()
        return True

    def _invalidate(self) -> None:
        """The context changed, so nothing found under the old one still holds."""
        self.pool = []
        self.pool_query = ""
