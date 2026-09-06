"""The noisy channel: log P(keystrokes | intended string).

The typist is modelled as a noisy transmission of the string they meant. Each
elementary error has a probability, and the cost of an alignment is the sum of
their negative log-probabilities -- so a "match score" here is a genuine
log-likelihood in nats and can be added straight to the language model's
log-prior. That is what makes ranking a posterior rather than a heuristic.

The errors a typist actually makes:

* **substitute** -- hit the wrong key. Cheaper when the wrong key is a
  physical neighbour on the layout being used, which is most typos.
* **delete** -- an extra keystroke that nothing in the intended string
  explains. Deliberately the dearest of the three: a typist who adds a letter
  is usually adding it to disambiguate, so a candidate that cannot account for
  it should fall sharply rather than shrug it off.
* **skip** -- characters of the intended string that were never typed. This is
  abbreviation, and it is charged per *gap*, not per character.

**Why gaps and not characters.** Charging a flat rate per skipped character
says every omitted letter is an independent accident, and that is simply not
how anyone abbreviates. Typing "helhay" for "hello how are you" drops eleven
characters in four runs -- `hel[lo ]h[ow ]a[re ]y[ou]` -- which at a flat 2.3
nats each came to 25.3 and lost to explaining the same keystrokes as three
unrelated *substitutions* at 12.0. The real reading of the abbreviation was
literally more expensive than nonsense, and "Here you go" (11.0) beat the
sentence the typist meant.

So a gap costs ``skip_open`` to start and ``skip_extend`` per character after.
Dropping the tail of a word is then one decision rather than five, which is
what it is; and because opening is dear, a candidate cannot cheaply litter
small gaps everywhere to fit arbitrary keystrokes. Characters *past* what was
typed stay free -- that is the prediction, not an error.

**The prefix-closure property.** The grid is the standard alignment DP,
``d[i][j]`` = cost of turning the first ``i`` keystrokes into the first ``j``
characters of the candidate, in two states: ``A`` for alignments whose last
step consumed a keystroke, ``B`` for those inside a gap. Because every
elementary cost is non-negative, the *minimum of a column* is non-decreasing
as the candidate grows. So once a partial candidate's column minimum exceeds a
budget, no extension can bring it back under -- which is what lets the search
prune a branch instead of decoding it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "ChannelCosts",
    "MatchResult",
    "Grid",
    "LAYOUTS",
    "neighbor_map",
    "initial_column",
    "push_candidate_char",
    "initial_row",
    "push_query_char",
    "grid_values",
    "partial_cost",
    "match",
]

INF = float("inf")

#: One column (or row) of the DP, as ``(A, B)``: costs of alignments ending
#: outside a gap and inside one. Two states are what an affine gap needs.
Grid = tuple[tuple[float, ...], tuple[float, ...]]

# Physical key grids. A typo is usually a finger landing one key off, so the
# substitution cost is discounted between neighbours -- which requires knowing
# the layout the user actually types on, not the letters' alphabetical order.
LAYOUTS: dict[str, tuple[str, ...]] = {
    "qwerty": ("qwertyuiop", "asdfghjkl", "zxcvbnm"),
    # Colemak-DH, the layout this was built on.
    "colemak-dh": ("qwfpbjluy", "arstgmneio", "zxcdvkh"),
    "none": (),
}


def neighbor_map(layout: str) -> dict[str, frozenset[str]]:
    """Adjacency (horizontal, vertical and diagonal) on a keyboard grid.

    Rows are treated as aligned columns, which is exactly true on the
    ortholinear board this was written for and close enough on a staggered one
    to be worth having.
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

    The defaults are readable as probabilities -- ``exp(-cost)`` -- so they can
    be argued with. They were chosen to separate real abbreviations from
    coincidences on a set of measured pairs; see ``tests/test_channel.py``.
    """

    substitute: float = 4.0  # ~1.8% -- an unrelated wrong letter
    substitute_near: float = 2.6  # ~7%  -- a neighbouring key
    #: A keystroke nothing explains, in the *middle* of a match. It used to be
    #: 6.0 to make an added letter count for something, but the unread-tail
    #: charge carries that now -- and at 6.0 it had become dearer than simply
    #: stopping early, so a typo in the middle of a word truncated the match
    #: instead of being corrected through.
    delete: float = 4.5
    skip_open: float = 2.0  # starting to leave characters out
    skip_extend: float = 0.35  # ...and continuing to, which is nearly free
    #: What it costs to leave the rest of the keystrokes unread *for now*.
    #: Trailing keystrokes a candidate does not reach are not mistakes -- they
    #: are the rest of what is being typed, for the next suggestion to take.
    #: Billed as spurious keystrokes at 6.0 each, "this is a test" cost 132.6
    #: nats against a 33-character shorthand and could never be offered at
    #: all, so nothing could be accepted until a single candidate covered the
    #: whole sentence at once.
    #:
    #: Affine for the same reason gaps are. A flat rate cannot be both things
    #: at once: dear enough that ignoring the "l" you just typed in "hel"
    #: means something, and cheap enough that ignoring the last nineteen
    #: characters of a long shorthand does not. Stopping early is one
    #: decision, not nineteen.
    #: ``tail_extend`` is really the credit for each keystroke a candidate
    #: *does* read, so it has to exceed what reading one costs -- roughly 1.5
    #: to 2.5 nats of gap in dense shorthand. Below that the ranking prefers
    #: to stop early and read almost nothing: against a 33-character
    #: shorthand, 1.2 put "Thi|s week" on top having read three keystrokes,
    #: while 3.0 gives "This is the start of the next section" having read
    #: twenty-four.
    tail_open: float = 4.0
    tail_extend: float = 3.0
    case: float = 0.4  # right letter, wrong case
    #: Total error tolerated before a branch is abandoned. It grows with the
    #: query because a longer abbreviation opens more gaps.
    budget_base: float = 8.0
    budget_per_char: float = 1.5
    neighbors: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def for_layout(cls, layout: str, **kwargs: float) -> "ChannelCosts":
        return cls(neighbors=neighbor_map(layout), **kwargs)

    def budget(self, query_len: int) -> float:
        return self.budget_base + self.budget_per_char * query_len

    def tail_charge(self, unread: int) -> float:
        """Cost of stopping this candidate with ``unread`` keystrokes to go."""
        if unread <= 0:
            return 0.0
        return self.tail_open + self.tail_extend * (unread - 1)

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
    #: How many keystrokes this candidate accounts for. Accepting it consumes
    #: exactly these, leaving the rest to be matched by what comes next.
    keystrokes: int = 0
    #: The part of ``cost`` that is actual error, with the charge for
    #: keystrokes left unread taken back out.
    errors: float = 0.0

    @property
    def error_rate(self) -> float:
        """Error in nats per keystroke read -- the length-independent measure.

        Raw error cannot describe match quality on its own: reading
        thirty-three keystrokes of dense shorthand accrues more of it than
        reading three, without being any worse a reading.
        """
        return self.errors / self.keystrokes if self.keystrokes else float("inf")

    @property
    def likelihood(self) -> float:
        """P(keystrokes | candidate), for reading the cost as a probability."""
        return math.exp(-self.cost) if self.cost != INF else 0.0


def grid_values(grid: Grid) -> tuple[float, ...]:
    """Best cost at each keystroke position, whichever state it ends in."""
    a, b = grid
    return tuple(map(min, a, b))


# -- the grid, extended along either axis -------------------------------
#
# Two entry points because the two callers grow the grid along different axes:
# the search grows the *candidate* one token at a time with the query fixed,
# while the UI grows the *query* one keystroke at a time with the candidate
# fixed. Both maintain the same numbers, which a test asserts.


def initial_column(query_len: int, costs: ChannelCosts) -> Grid:
    """Column j=0: an empty candidate explains nothing, so every keystroke is
    a deletion, and no gap has been opened."""
    return (
        tuple(i * costs.delete for i in range(query_len + 1)),
        (INF,) * (query_len + 1),
    )


def push_candidate_char(
    grid: Grid, query: str, ch: str, costs: ChannelCosts
) -> Grid:
    """Extend the candidate by one character (search direction)."""
    a_old, b_old = grid
    m = len(a_old) - 1
    b_new = tuple(
        min(a_old[i] + costs.skip_open, b_old[i] + costs.skip_extend)
        for i in range(m + 1)
    )
    a_new = [INF] * (m + 1)
    for i in range(1, m + 1):
        diagonal = min(a_old[i - 1], b_old[i - 1])
        above = min(a_new[i - 1], b_new[i - 1])
        a_new[i] = min(
            diagonal + costs.substitution(query[i - 1], ch),
            above + costs.delete,
        )
    return (tuple(a_new), b_new)


def initial_row(candidate_len: int, costs: ChannelCosts) -> Grid:
    """Row i=0: nothing typed yet, so the candidate is one long unopened gap."""
    a = (0.0,) + (INF,) * candidate_len
    b = (INF,) + tuple(
        costs.skip_open + (j - 1) * costs.skip_extend
        for j in range(1, candidate_len + 1)
    )
    return (a, b)


def push_query_char(grid: Grid, candidate: str, ch: str, costs: ChannelCosts) -> Grid:
    """Extend the query by one keystroke (interactive direction)."""
    a_old, b_old = grid
    n = len(a_old) - 1
    a_new = [INF] * (n + 1)
    b_new = [INF] * (n + 1)
    a_new[0] = min(a_old[0], b_old[0]) + costs.delete
    for j in range(1, n + 1):
        diagonal = min(a_old[j - 1], b_old[j - 1])
        a_new[j] = min(
            diagonal + costs.substitution(ch, candidate[j - 1]),
            min(a_old[j], b_old[j]) + costs.delete,
        )
        b_new[j] = min(
            a_new[j - 1] + costs.skip_open, b_new[j - 1] + costs.skip_extend
        )
    return (tuple(a_new), tuple(b_new))


def partial_cost(
    grid: Grid, query_len: int, costs: ChannelCosts
) -> tuple[float, int]:
    """Cheapest way to account for *some* leading part of the keystrokes.

    Used by the search to decide whether a finished candidate is worth
    offering. It reads one column, so the candidate's own characters are all
    charged; the exact figure, with the predicted tail free, is recomputed
    when the candidate is ranked.
    """
    values = grid_values(grid)
    best, keystrokes = INF, 0
    for covered, cost in enumerate(values):
        total = cost + costs.tail_charge(query_len - covered)
        if total < best:
            best, keystrokes = total, covered
    return best, keystrokes


def match(query: str, candidate: str, costs: ChannelCosts) -> MatchResult:
    """-log P(query | candidate), and how much of each side it accounts for.

    Both tails are cheap, and for the same reason. Candidate characters past
    what was typed are free -- that is the prediction. Keystrokes past what
    the candidate reaches cost ``costs.tail`` each, because they are not
    errors either: they are the rest of the sentence, waiting for the next
    suggestion to take them.

    So the alignment is chosen over *both* axes: how many keystrokes to
    account for, and how much of the candidate to spend doing it.
    """
    rows = [initial_row(len(candidate), costs)]
    for ch in query:
        rows.append(push_query_char(rows[-1], candidate, ch, costs))

    best = MatchResult(cost=INF, consumed=0, keystrokes=0)
    for keystrokes, grid in enumerate(rows):
        values = grid_values(grid)
        consumed, cost = 0, values[0]
        for j in range(1, len(values)):
            if values[j] < cost:
                cost, consumed = values[j], j
        total = cost + costs.tail_charge(len(query) - keystrokes)
        if total < best.cost:
            best = MatchResult(
                cost=total, consumed=consumed, keystrokes=keystrokes,
                errors=cost,
            )
    return best
