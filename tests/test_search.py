"""The posterior walk: merging, emission, pruning and seeding."""

from __future__ import annotations

import math

import pytest
from fake_lm import cat_lm, long_lm, prefix_lm

from fuzzytype.channel import ChannelCosts
from fuzzytype.search import (
    PredictConfig,
    _emission_key,
    _seed_token_paths,
    predict,
)

COSTS = ChannelCosts()
CONTEXT = (1,)


def _run(query="", **kwargs):
    lm = cat_lm(CONTEXT)
    config = PredictConfig(k=20, **{"max_rounds": 8, **kwargs})
    candidates, stats = predict(lm, CONTEXT, query, config, COSTS)
    return lm, {c.text: c for c in candidates}, stats


def test_every_spelling_of_a_string_is_summed_into_one_candidate():
    """"cat" can be written three ways; it should appear once, at 0.7.

    P = 0.5*0.7 (" cat"+" ") + 0.5*0.3 (" cat"+".") + 0.2*1.0 (" ca"+"t"+" ").
    Getting this wrong in either direction -- listing the spellings
    separately, or counting one twice -- is the single easiest way for the
    ranking to become quietly meaningless.
    """
    _, found, _ = _run()
    assert math.exp(found["cat"].logprob) == pytest.approx(0.7)
    assert found["cat"].n_paths == 3


def test_sibling_strings_keep_their_own_probability():
    _, found, _ = _run()
    assert math.exp(found["car"].logprob) == pytest.approx(0.3 * 0.6)
    assert math.exp(found["cart"].logprob) == pytest.approx(0.3 * 0.4)


def test_no_candidate_can_be_more_than_certain():
    """A log-prior above zero means probability was double counted."""
    _, found, _ = _run()
    assert found
    for candidate in found.values():
        assert candidate.logprob <= 0.0


def test_the_terminator_is_evidence_not_content():
    _, found, _ = _run()
    assert "cat" in found
    assert not any(text.endswith((" ", ".")) for text in found)


def test_candidates_are_ranked_by_the_posterior():
    _, found, _ = _run()
    scores = [c.score for c in found.values()]
    assert scores == sorted(scores, reverse=True) or len(scores) == 1


def test_keystrokes_reorder_the_candidates():
    """The prior alone prefers "cat"; typing "car" must override that."""
    _, without, _ = _run()
    _, with_query, _ = _run("car")
    assert max(without, key=lambda t: without[t].score) == "cat"
    assert max(with_query, key=lambda t: with_query[t].score) == "car"


def test_a_branch_that_cannot_explain_the_keystrokes_is_never_decoded():
    """Pruning must stop exploration, not merely filter the results.

    The observable difference is how much the search *asked the model*: a
    hopeless query has to expand strictly fewer states than no query at all,
    and has to run out of frontier rather than out of rounds.

    The query has to be reasonably long for this to bite at all. Any
    candidate can be "explained" by simply deleting every keystroke, which
    costs ``len(query) * delete`` no matter how wrong it is -- so a branch is
    only ever abandoned once that fallback also exceeds the budget.
    """
    config = PredictConfig(k=20, max_rounds=8)
    idle_lm = long_lm(CONTEXT)
    predict(idle_lm, CONTEXT, "", config, COSTS)

    dead_lm = long_lm(CONTEXT)
    candidates, stats = predict(dead_lm, CONTEXT, "zzzzzz", config, COSTS)

    assert stats.pruned_by_channel > 0
    assert not candidates
    assert stats.exhausted, "the frontier should die, not hit the round limit"
    assert len(dead_lm.seen) < len(idle_lm.seen)


def test_emission_key_drops_the_terminator_and_fires_once():
    assert _emission_key("", " cat ") == "cat"
    assert _emission_key("", " cat.") == "cat"
    # Walking further into a run of terminators must not report "cat" again,
    # or its probability would be added twice.
    assert _emission_key(" cat ", " cat ,") is None
    # ...but a genuinely longer phrase is a new candidate.
    assert _emission_key(" cat ", " cat and ") == "cat and"
    # Unfinished words are not offered at all.
    assert _emission_key("", " cat") is None


def test_seeding_offers_the_token_aligned_prefix_as_well():
    """The partial word's own token path is usually a dead end.

    " aprico" spells as [" apr", "ico"], from which the model will not write
    "t". The prefix [" apr"] is the path that reaches the real word, so both
    must be offered.
    """
    lm = cat_lm(CONTEXT)
    paths = _seed_token_paths(lm, [" cart"])
    assert tuple(lm.encode(" cart")) in paths  # the literal
    assert tuple(lm.encode(" car")) in paths  # the token-aligned prefix


def test_seeding_prices_the_branch_with_a_real_forward_pass():
    lm = cat_lm(CONTEXT)
    candidates, stats = predict(
        lm, CONTEXT, "cart", PredictConfig(k=20, max_rounds=8), COSTS,
        seeds=[" cart"],
    )
    assert stats.seeded >= 1
    by_text = {c.text: c for c in candidates}
    # Seeded or not, the probability reported is the model's own.
    assert math.exp(by_text["cart"].logprob) == pytest.approx(0.3 * 0.4)


def test_search_is_deterministic():
    first = {t: c.score for t, c in _run("ca")[1].items()}
    second = {t: c.score for t, c in _run("ca")[1].items()}
    assert first == second


def test_rounds_bound_the_wall_clock():
    _, _, stats = _run(max_rounds=1)
    assert stats.rounds == 1


def test_a_word_outside_the_top_k_is_reachable_through_the_vocabulary():
    """Walking the tree cannot find a word the model never proposes.

    A whole word is often a single token that is ranked far below the search's
    cut-off while still being a good guess -- "Hello" scores -14.2 against
    "Here" at -11.6, under three nats apart, yet nowhere near the top-64 the
    search expands. Typing "hel" then offered every "Here ..." and no "Hello",
    and adding the "l" made it *worse*, because the letter could only be
    charged as a slip.
    """
    config = PredictConfig(k=20, max_rounds=6, child_top_k=2)

    blind = prefix_lm(CONTEXT)
    found, _ = predict(blind, CONTEXT, "cart", config, COSTS)
    assert "cart" not in {c.text for c in found}, "top-k should hide it"

    seeing = prefix_lm(CONTEXT)
    found, stats = predict(
        seeing, CONTEXT, "cart", config, COSTS, seeds=[" cart"]
    )
    assert "cart" in {c.text for c in found}
    assert stats.seeded >= 1


def test_the_vocabulary_seed_carries_the_model_s_own_probability():
    seeing = prefix_lm(CONTEXT)
    found, _ = predict(
        seeing, CONTEXT, "cart",
        PredictConfig(k=20, max_rounds=6, child_top_k=2), COSTS, seeds=[" cart"],
    )
    cart = next(c for c in found if c.text == "cart")
    # 0.05 for " cart" then 1.0 for the terminator -- not an assumed prior.
    assert math.exp(cart.logprob) == pytest.approx(0.05)
