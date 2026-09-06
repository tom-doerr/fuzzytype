"""The noisy channel: log P(keystrokes | intended string).

The typist is modelled as a noisy transmission of the string they meant. Each
elementary error has a probability, and the cost of an alignment is the sum of
their negative log-probabilities -- so a "match score" here is a genuine
log-likelihood in nats and can be added straight to the language model's
log-prior. That is what makes ranking a posterior rather than a heuristic.

Three error types, because they are the three things a typist actually does:

* **substitute** -- hit the wrong key. Cheaper when the wrong key is a
  physical neighbour on the layout being used, which is most typos.
* **delete** -- an extra keystroke that nothing in the intended string
  explains (a stutter, or a word this candidate simply does not contain).
* **skip** -- a character of the intended string that was never typed. This
  is what makes abbreviation work: "wte" -> "write" is two skips.

Characters of the candidate *past* what was typed are free. That is the whole
point -- the untyped tail is the prediction, not an error.

**The prefix-closure property.** The dynamic program is the standard
Levenshtein grid ``d[i][j]`` = cost of turning the first ``i`` keystrokes into
the first ``j`` characters of the candidate. Because every elementary cost is
non-negative, the *minimum of a column* is non-decreasing as the candidate
grows. So once a partial candidate's column minimum exceeds a budget, no
extension of it can ever come back under -- which is what lets the search
prune a branch instead of decoding it. Everything else in this file exists to
make that one bound cheap to maintain incrementally.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "ChannelCosts",
    "MatchResult",
    "LAYOUTS",
    "neighbor_map",
    "initial_column",
    "push_candidate_char",
    "initial_row",
    "push_query_char",
    "match",
]

# Physical key grids. A typo is usually a finger landing one key off, so the
# substitution cost is discounted between neighbours -- which requires knowing
# the layout the user actually types on, not the letters' alphabetical order.
LAYOUTS: dict[str, tuple[str, ...]] = {
    "qwerty": ("qwertyuiop", "asdfghjkl", "zxcvbnm"),
    # Colemak-DH, the layout this was built on.
    "colemak-dh": ("qwfpbjluy", "arstgmnei o".replace(" ", ""), "zxcdvkh"),
    "none": (),
}


def neighbor_map(layout: str) -> dict[str, frozenset[str]]:
    """Adjacency (horizontal, vertical and diagonal) on a keyboard grid.

    Rows are treated as aligned columns, which is exactly true on the
    ortholinear board this was written for and close enough on a staggered
    one to be worth having.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; known: {sorted(LAYOUTS)}")
    rows = LAYOUTS[layout]
    out: dict[str, set[str]] = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            near = out.setdefault(ch, set())
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < len(rows) and 0 <= cc < len(rows[rr]):
                        near.add(rows[rr][cc])
    return {k: frozenset(v) for k, v in out.items()}


@dataclass(frozen=True)
class ChannelCosts:
    """Negative log-probabilities, in nats, of each elementary typing error.

    The defaults say: a keystroke is right ~97% of the time, a neighbour-key
    slip is a few times likelier than an arbitrary wrong letter, and leaving a
    character out entirely (abbreviating) is the most forgivable thing a
    hurried typist does. They are deliberately readable as probabilities --
    ``exp(-cost)`` -- so they can be argued with.
    """

    substitute: float = 4.0  # ~1.8% -- an unrelated wrong letter
    substitute_near: float = 2.6  # ~7%  -- a neighbouring key
    delete: float = 3.5  # ~3%  -- a keystroke nothing explains
    skip: float = 2.3  # ~10% -- a character the typist did not type
    case: float = 0.4  # right letter, wrong case
    #: How much total error to tolerate before a branch is abandoned. It has
    #: to grow with the query at roughly the cost of an omitted character,
    #: because that is what heavy abbreviation actually spends: "th wthr hs
    #: bn" leaves seven characters out of "the weather has been", which is
    #: ~16 nats of skip on its own. A budget that grows more slowly than the
    #: typist abbreviates rejects the very candidate they meant.
    budget_base: float = 8.0
    budget_per_char: float = 1.5
    neighbors: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def for_layout(cls, layout: str, **kwargs: float) -> "ChannelCosts":
        return cls(neighbors=neighbor_map(layout), **kwargs)

    def budget(self, query_len: int) -> float:
        return self.budget_base + self.budget_per_char * query_len

    def substitution(self, typed: str, intended: str) -> float:
        """Cost of the typist producing ``typed`` when meaning ``intended``."""
        if typed == intended:
            return 0.0
        lo_t, lo_i = typed.lower(), intended.lower()
        if lo_t == lo_i:
            return self.case
        if lo_i in self.neighbors.get(lo_t, ()):
            return self.substitute_near
        return self.substitute


@dataclass(frozen=True)
class MatchResult:
    """How well a candidate explains the keystrokes.

    ``consumed`` is how many characters of the candidate the typing accounts
    for; the rest is the part being predicted, which is what the UI
    highlights.
    """

    cost: float
    consumed: int

    @property
    def likelihood(self) -> float:
        """P(keystrokes | candidate), for reading the cost as a probability."""
        return math.exp(-self.cost)


# -- the grid, extended along either axis -------------------------------
#
# ``d[i][j]``: cost of explaining the first i keystrokes with the first j
# characters of the candidate. Two entry points because the two callers grow
# the grid along different axes: the search grows the *candidate* one token at
# a time with the query fixed, while the UI grows the *query* one keystroke at
# a time with the candidate fixed. Both maintain the same numbers.


def initial_column(query_len: int, costs: ChannelCosts) -> tuple[float, ...]:
    """Column j=0: an empty candidate explains nothing, so every keystroke is
    a deletion."""
    return tuple(i * costs.delete for i in range(query_len + 1))


def push_candidate_char(
    column: tuple[float, ...], query: str, ch: str, costs: ChannelCosts
) -> tuple[float, ...]:
    """Extend the candidate by one character (search direction)."""
    new = [column[0] + costs.skip]
    for i in range(1, len(column)):
        new.append(
            min(
                new[i - 1] + costs.delete,
                column[i] + costs.skip,
                column[i - 1] + costs.substitution(query[i - 1], ch),
            )
        )
    return tuple(new)


def initial_row(candidate_len: int, costs: ChannelCosts) -> tuple[float, ...]:
    """Row i=0: nothing typed yet, so every candidate character is untyped."""
    return tuple(j * costs.skip for j in range(candidate_len + 1))


def push_query_char(
    row: tuple[float, ...], candidate: str, ch: str, costs: ChannelCosts
) -> tuple[float, ...]:
    """Extend the query by one keystroke (interactive direction)."""
    new = [row[0] + costs.delete]
    for j in range(1, len(row)):
        new.append(
            min(
                row[j] + costs.delete,
                new[j - 1] + costs.skip,
                row[j - 1] + costs.substitution(ch, candidate[j - 1]),
            )
        )
    return tuple(new)


def _best(row: tuple[float, ...]) -> MatchResult:
    """Best alignment of the whole query against any *prefix* of the candidate.

    Taking the minimum over j is what makes the untyped tail free.
    """
    best_j = 0
    best = row[0]
    for j in range(1, len(row)):
        if row[j] < best:
            best, best_j = row[j], j
    return MatchResult(cost=best, consumed=best_j)


def match(query: str, candidate: str, costs: ChannelCosts) -> MatchResult:
    """-log P(query | candidate), plus how much of the candidate was typed."""
    row = initial_row(len(candidate), costs)
    for ch in query:
        row = push_query_char(row, candidate, ch, costs)
    return _best(row)
