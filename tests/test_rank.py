"""Ranking: probabilities, the length preference, and the error budget."""

from __future__ import annotations

import pytest

from fuzzytype.channel import ChannelCosts
from fuzzytype.rank import DEFAULT_LENGTH_BONUS, length_credit, rerank
from fuzzytype.search import Candidate

COSTS = ChannelCosts()


def _candidate(text, logprob):
    return Candidate(
        text=text, raw=" " + text, logprob=logprob, cost=0.0,
        consumed=0, keystrokes=0, n_paths=1, tokens=(),
    )


POOL = [
    _candidate("clothes", -2.0),
    _candidate("clothing", -3.0),
    _candidate("bread", -2.5),
]


def test_probabilities_are_a_distribution():
    shown, coverage = rerank(POOL, "", COSTS)
    assert sum(s.probability for s in shown) == pytest.approx(1.0)
    assert coverage == pytest.approx(1.0)


def test_coverage_reports_what_the_visible_rows_carry():
    shown, coverage = rerank(POOL, "", COSTS, k=1)
    assert len(shown) == 1
    assert coverage == pytest.approx(shown[0].probability)
    assert coverage < 1.0


def test_keystrokes_beat_the_prior():
    """"clothes" is the likelier string, but "brd" is unambiguous."""
    shown, _ = rerank(POOL, "brd", COSTS)
    assert shown[0].text == "bread"


def test_matched_and_predicted_split_the_candidate():
    shown, _ = rerank(POOL, "clo", COSTS)
    top = shown[0]
    assert top.matched == "clo"
    assert top.predicted == "thes"
    assert top.matched + top.predicted == top.text


def test_over_budget_candidates_are_dropped():
    shown, _ = rerank(POOL, "zzzzzzzz", COSTS)
    assert shown == []


def test_length_bonus_shifts_the_preference_to_longer_text():
    pool = [_candidate("go", -1.0), _candidate("go to the shop", -4.0)]
    short, _ = rerank(pool, "", COSTS, length_bonus=0.0)
    long, _ = rerank(pool, "", COSTS, length_bonus=3.0)
    assert short[0].text == "go"
    assert long[0].text == "go to the shop"


def test_reranking_never_touches_the_model():
    """The per-keystroke path must be pure Python; this is what makes it fast."""
    shown, _ = rerank(POOL, "clo", COSTS)
    assert shown  # no LanguageModel was supplied at all


def test_a_disambiguating_letter_promotes_the_matching_candidate():
    """The reported bug: typing "he", then adding "l" to mean "hello".

    Every suggestion stayed "Here ...". Two causes, both fixed elsewhere: the
    Hello family was never decoded at all, and a 27-character candidate
    collected +10.8 nats of an uncapped length bonus -- more than the cost of
    ignoring the "l" outright. Adding a letter to disambiguate made the wrong
    answer *more* confident.

    The priors here are the ones the model actually assigns after the default
    preamble, so this is the measured case rather than an invented one.
    """
    pool = [
        _candidate("Hello everyone", -11.25),
        _candidate("Here is what I have written", -10.70),
    ]
    # "he" is genuinely ambiguous -- both match, so the prior may lead.
    shown, _ = rerank(pool, "he", COSTS, length_bonus=DEFAULT_LENGTH_BONUS)
    assert shown[0].text == "Here is what I have written"
    # ...but the "l" is evidence, and it has to move the ranking.
    shown, _ = rerank(pool, "hel", COSTS, length_bonus=DEFAULT_LENGTH_BONUS)
    assert shown[0].text == "Hello everyone"


def test_the_length_credit_saturates():
    """Unbounded, it eventually decides the ranking by itself.

    Doubling the length must be worth steadily less, so the gap between any
    two candidates stays small enough that a better match can overcome it.
    """
    steps = [length_credit("x" * n, DEFAULT_LENGTH_BONUS) for n in (5, 10, 15, 20, 25)]
    gains = [b - a for a, b in zip(steps, steps[1:])]
    assert all(later < earlier for earlier, later in zip(gains, gains[1:]))
    assert steps == sorted(steps), "longer must still be preferred"


def test_length_credit_is_off_when_the_coefficient_is_zero():
    assert length_credit("a long candidate", 0.0) == 0.0


def test_typing_less_is_cheaper_than_typing_wrong():
    """The tool is for fewer keystrokes first, typo tolerance second.

    Omitting characters is the intended way to use it, so it must be the
    cheapest thing a typist can do; a keystroke the candidate cannot account
    for is the most expensive.
    """
    assert COSTS.skip_open < COSTS.substitute < COSTS.delete
    assert COSTS.skip_extend < COSTS.skip_open


def test_the_two_signals_are_reported_separately():
    """So it is visible which of them is driving a suggestion."""
    # "bird" reads the keystrokes slightly better; "bread" is the likelier text
    pool = [_candidate("bird", -8.0), _candidate("bread", -2.0)]
    shown, _ = rerank(pool, "brd", COSTS)
    by_text = {s.text: s for s in shown}
    assert by_text["bread"].lm_probability > by_text["bird"].lm_probability
    assert by_text["bird"].match_probability > by_text["bread"].match_probability
    for column in ("probability", "lm_probability", "match_probability"):
        assert sum(getattr(s, column) for s in shown) == pytest.approx(1.0)


def test_a_forced_alignment_is_not_offered():
    """Past a certain error per keystroke the matcher is inventing a reading."""
    pool = [_candidate("bread", -2.0)]
    assert rerank(pool, "brd", COSTS)[0], "a real abbreviation stays"
    assert rerank(pool, "qzxvk", COSTS)[0] == [], "a forced one does not"
