"""Ranking: probabilities, the length preference, and the error budget."""

from __future__ import annotations

import pytest

from fuzzytype.channel import ChannelCosts
from fuzzytype.rank import rerank
from fuzzytype.search import Candidate

COSTS = ChannelCosts()


def _candidate(text, logprob):
    return Candidate(
        text=text, raw=" " + text, logprob=logprob, cost=0.0,
        consumed=0, n_paths=1, tokens=(),
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
    long, _ = rerank(pool, "", COSTS, length_bonus=0.8)
    assert short[0].text == "go"
    assert long[0].text == "go to the shop"


def test_reranking_never_touches_the_model():
    """The per-keystroke path must be pure Python; this is what makes it fast."""
    shown, _ = rerank(POOL, "clo", COSTS)
    assert shown  # no LanguageModel was supplied at all
