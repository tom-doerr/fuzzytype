"""Turning scores into probabilities, and re-ranking without the GPU.

Two jobs.

**Probabilities.** ``predict`` returns unnormalised log posteriors. Softmaxing
them over the candidate set gives ``P(this is what you meant)`` -- a real
number to show the user instead of an opaque score. It is a posterior over
the strings that were *found*, so ``coverage`` reports how much of the found
mass the displayed rows carry; the search's own frontier bound says how much
might still be missing.

**Re-ranking on every keystroke.** A decode costs ~1 s; a keystroke has ~20 ms
before it feels laggy. So the LM runs in the background to produce a pool of
candidates, and each keystroke only re-scores that cached pool through the
channel. The prior is already known per candidate, so a keystroke costs one
Levenshtein grid per candidate and nothing else.

**The length bonus.** Candidates of different lengths compete on joint
probability, and a longer string is always less probable than its own prefix
-- so an honest posterior systematically prefers the shortest completion.
Since the LM spends roughly a fixed number of nats per character of ordinary
text, adding back a fixed bonus per character puts long and short candidates
on comparable footing. It is a display-time preference, not a probability,
which is why it lives here and not in the search.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .channel import ChannelCosts, match
from .search import Candidate

__all__ = ["Suggestion", "rerank", "DEFAULT_LENGTH_BONUS"]

#: Nats per character, added back to offset the prior's length penalty.
#: Chosen by measurement rather than taste: on "Thanks for the update. I will"
#: a bonus of 0.0 offers only "look into it"-length completions, 0.8 collapses
#: onto a single long sentence at 83%, and 0.4 keeps both lengths on the list
#: ("keep you posted", "get back to you", "look into it") -- which is what
#: lets a typist choose how much sentence to accept.
DEFAULT_LENGTH_BONUS = 0.4


@dataclass(frozen=True)
class Suggestion:
    """A ranked candidate, with the numbers that produced its rank."""

    text: str
    raw: str
    logprob: float  # log P(text | context)
    cost: float  # -log P(keystrokes | text)
    consumed: int  # characters of `text` the keystrokes explain
    score: float  # log posterior, including any length bonus
    probability: float  # normalised over the candidate set
    n_paths: int

    @property
    def prior(self) -> float:
        """P(text | context) from the language model alone."""
        return math.exp(self.logprob)

    @property
    def prior_display(self) -> str:
        p = self.prior
        return f"{p:.2e}" if p < 1e-3 else f"{p:.5f}"

    @property
    def predicted(self) -> str:
        """The part being predicted rather than typed."""
        return self.text[self.consumed :]

    @property
    def matched(self) -> str:
        return self.text[: self.consumed]


def rerank(
    candidates: Sequence[Candidate],
    query: str,
    costs: ChannelCosts | None = None,
    *,
    length_bonus: float = DEFAULT_LENGTH_BONUS,
    k: int | None = None,
    drop_over_budget: bool = True,
) -> tuple[list[Suggestion], float]:
    """Re-score a cached pool against the keystrokes typed so far.

    Returns the ranked suggestions and the fraction of the found posterior
    mass they carry.
    """
    ch = costs or ChannelCosts()
    budget = ch.budget(len(query))

    scored: list[tuple[float, Candidate, float, int]] = []
    for cand in candidates:
        result = match(query, cand.text, ch)
        if drop_over_budget and result.cost > budget:
            continue
        score = cand.logprob - result.cost + length_bonus * len(cand.text)
        scored.append((score, cand, result.cost, result.consumed))

    if not scored:
        return [], 0.0

    top = max(s for s, _, _, _ in scored)
    total = sum(math.exp(s - top) for s, _, _, _ in scored)
    scored.sort(key=lambda item: (-item[0], item[1].text))

    out = [
        Suggestion(
            text=cand.text,
            raw=cand.raw,
            logprob=cand.logprob,
            cost=cost,
            consumed=min(consumed, len(cand.text)),
            score=score,
            probability=math.exp(score - top) / total,
            n_paths=cand.n_paths,
        )
        for score, cand, cost, consumed in scored
    ]
    shown = out if k is None else out[:k]
    return shown, sum(s.probability for s in shown)
