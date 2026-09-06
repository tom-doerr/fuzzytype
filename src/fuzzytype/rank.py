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

**The length credit.** An honest posterior always prefers the shortest
completion, so some credit for length has to be added back or every
suggestion is one word long. It saturates, so that a preference can never
become the ranking -- see :func:`length_credit`. It is a display-time
preference rather than a probability, which is why it lives here and not in
the search.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .channel import ChannelCosts, match
from .search import Candidate

__all__ = ["Suggestion", "rerank", "DEFAULT_LENGTH_BONUS", "length_credit"]

#: Coefficient of the saturating length term (see :func:`length_credit`).
#: 1.5 by measurement: it is small enough that "hello" still beats a
#: 27-character candidate that ignores a deliberately typed "l", and large
#: enough that "get back to you as soon as possible" still outranks "get back
#: to you" when both match equally well.
DEFAULT_LENGTH_BONUS = 1.5


def length_credit(text: str, coefficient: float) -> float:
    """How much a candidate's length is worth, in nats.

    Candidates of different lengths compete on joint probability, and a
    longer string is always less probable than its own prefix -- so an honest
    posterior systematically prefers the shortest completion, and a list of
    one-word suggestions is not what anyone wants. Some credit for length has
    to be added back.

    It has to *saturate*, though, and this was a real bug rather than a
    refinement. A credit linear in length is unbounded, so past some point it
    simply decides the ranking: against "hel", a 27-character "Here is what I
    have written" collected 10.8 nats -- far more than the cost of ignoring
    the "l" entirely -- so adding a letter to disambiguate made the wrong
    answer *more* confident. Capping it flat does not work either: a cap low
    enough to protect the match is also low enough that every sentence hits it
    and length stops ordering anything.

    A logarithm does both. Going from five characters to twenty-five is worth
    a lot; from forty to sixty, very little -- which is also how useful the
    extra text actually is to a typist. And because it grows without bound but
    ever more slowly, the gap between any two candidates stays small enough
    that a genuinely better match wins.
    """
    if coefficient == 0.0 or not text:
        return 0.0
    return coefficient * math.log1p(len(text))


@dataclass(frozen=True)
class Suggestion:
    """A ranked candidate, with the numbers that produced its rank."""

    text: str
    raw: str
    logprob: float  # log P(text | context)
    cost: float  # -log P(keystrokes | text)
    consumed: int  # characters of `text` the keystrokes explain
    #: Keystrokes accounted for. Accepting consumes exactly these.
    keystrokes: int
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
    cost_weight: float = 1.0,
) -> tuple[list[Suggestion], float]:
    """Re-score a cached pool against the keystrokes typed so far.

    ``cost_weight`` scales the channel's contribution. It is 1 when the
    channel is the only account of the keystrokes, and less when the prior
    already knows about them -- prompt mode puts the shorthand in the prompt,
    so charging the full channel on top counts the same evidence twice.

    Returns the ranked suggestions and the fraction of the found posterior
    mass they carry.
    """
    ch = costs or ChannelCosts()
    budget = ch.budget(len(query))

    scored: list[tuple[float, Candidate, float, int, int]] = []
    for cand in candidates:
        result = match(query, cand.text, ch)
        if query and result.keystrokes == 0:
            continue  # accounts for nothing that was typed
        errors = result.cost - ch.tail_charge(len(query) - result.keystrokes)
        if drop_over_budget and errors > ch.budget(result.keystrokes):
            continue
        score = (
            cand.logprob
            - cost_weight * result.cost
            + length_credit(cand.text, length_bonus)
        )
        scored.append(
            (score, cand, result.cost, result.consumed, result.keystrokes)
        )

    if not scored:
        return [], 0.0

    top = max(item[0] for item in scored)
    total = sum(math.exp(item[0] - top) for item in scored)
    scored.sort(key=lambda item: (-item[0], item[1].text))

    out = [
        Suggestion(
            text=cand.text,
            raw=cand.raw,
            logprob=cand.logprob,
            cost=cost,
            consumed=min(consumed, len(cand.text)),
            keystrokes=keystrokes,
            score=score,
            probability=math.exp(score - top) / total,
            n_paths=cand.n_paths,
        )
        for score, cand, cost, consumed, keystrokes in scored
    ]
    shown = out if k is None else out[:k]
    return shown, sum(s.probability for s in shown)
