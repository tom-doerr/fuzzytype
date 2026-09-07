"""The posterior walk: merging, emission, pruning and seeding."""

from __future__ import annotations

import math

import pytest
from fake_lm import FakeLM, cat_lm, long_lm, prefix_lm

from fuzzytype.channel import ChannelCosts
from fuzzytype.search import PredictConfig, _emission_key, predict

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

    The budget here is deliberately strict. With the shipped costs a gap is
    cheap by design -- that is the whole point of the tool -- so the "explain
    nothing and skip the entire candidate" alignment stays affordable for a
    long time and the channel prunes little. The mechanism still has to work
    when the budget does bind, and this pins that.
    """
    strict = ChannelCosts(budget_base=1.0, budget_per_char=0.0)
    config = PredictConfig(k=20, max_rounds=8)

    idle_lm = long_lm(CONTEXT)
    predict(idle_lm, CONTEXT, "", config, strict)

    dead_lm = long_lm(CONTEXT)
    candidates, stats = predict(dead_lm, CONTEXT, "zzz", config, strict)

    assert stats.pruned_by_channel > 0
    assert not candidates
    assert stats.exhausted, "the frontier should die, not hit the round limit"
    assert len(dead_lm.seen) < len(idle_lm.seen)


def test_a_hopeless_query_yields_nothing_even_when_it_is_explored():
    """With cheap gaps the branch may survive; the budget still rejects it."""
    lm = long_lm(CONTEXT)
    candidates, _ = predict(
        lm, CONTEXT, "zzzzzz", PredictConfig(k=20, max_rounds=8), COSTS
    )
    assert not candidates


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


def test_search_is_deterministic():
    first = {t: c.score for t, c in _run("ca")[1].items()}
    second = {t: c.score for t, c in _run("ca")[1].items()}
    assert first == second


def test_rounds_bound_the_wall_clock():
    _, _, stats = _run(max_rounds=1)
    assert stats.rounds == 1


def _node_for(text, query, costs, logprob=-5.0):
    """Build the search's internal node for a candidate text, as _grow would."""
    from fuzzytype.channel import grid_values, initial_column, push_candidate_char
    from fuzzytype.search import _Node

    column = initial_column(len(query), costs)
    best, consumed = grid_values(column)[-1], 0
    m = len(query)
    for offset, ch in enumerate(text, start=1):
        column = push_candidate_char(column, query, ch, costs)
        full = min(column[0][m], column[1][m])
        if full < best:
            best, consumed = full, offset
    return _Node(
        tokens=(), logprob=logprob, data=b"", text=text,
        column=column, best_cost=best, best_consumed=consumed,
    )


def test_priority_scores_cost_and_remaining_work_together():
    """Guards the regression that cost "th wthr hs bn" all but four candidates.

    Once gaps became cheap, the cheapest partial alignment of almost any node
    was "open one gap and explain nothing at all". A signal built by taking
    that alignment first and *then* counting what it left unexplained
    therefore read as the whole query almost everywhere, went constant, and
    the search lost its sense of progress. Minimising the sum keeps a node
    that has explained more ahead of one that has not, at equal prior and
    equal length.
    """
    costs = ChannelCosts()
    query = "cat sat"
    for length in (4, 10, 30):
        covering = _node_for("cat "[:length].ljust(length, "z"), query, costs)
        unrelated = _node_for("z" * length, query, costs)
        assert covering.priority(2.0) > unrelated.priority(2.0), length


def test_priority_improves_as_more_of_the_query_is_explained():
    costs = ChannelCosts()
    query = "cat sat"
    scores = [
        _node_for(text, query, costs).priority(2.0)
        for text in ("c", "ca", "cat", "cat s", "cat sat")
    ]
    assert scores == sorted(scores), scores


def test_priority_still_reduces_to_the_bound_without_a_penalty():
    costs = ChannelCosts()
    node = _node_for("cat sat on", "cat sat", costs)
    assert node.priority(0.0) == pytest.approx(node.bound())


def test_batched_pricing_matches_pricing_one_at_a_time():
    """Padding a batch must not change a single number.

    Continuations of different lengths share a batch, padded to the widest
    and masked out of the sum. If the mask or the kept-slice offset were
    wrong the totals would silently include padding, which no other test
    would notice.
    """
    lm = cat_lm(CONTEXT)
    conts = [
        tuple(lm.encode(text)) for text in (" cat", " cat ", " car", " ca", " cart")
    ]
    together = lm.sequence_logprobs([(CONTEXT, c) for c in conts])
    alone = [lm.sequence_logprobs([(CONTEXT, c)])[0] for c in conts]
    assert together == pytest.approx(alone)


def test_a_token_is_worth_the_same_however_the_search_reaches_it():
    """Looking harder in one direction must not reprice anything.

    A second top-k is taken over the tokens that begin the way the keystrokes
    do, which changes which continuations get explored. Both are top-k over
    the same softmax across the whole vocabulary, so a token reached that way
    is worth exactly what it would have been worth in the first -- otherwise
    the posterior would quietly depend on how hard the search happened to
    look.
    """
    lm = cat_lm(CONTEXT)
    wide = lm.top_next([CONTEXT], top_k=10, top_p=1.0)[0]
    reference = dict(zip(wide.token_ids, wide.logprobs))

    narrow = lm.top_next([CONTEXT], top_k=1, top_p=1.0)[0]
    hidden = [t for t in reference if t not in narrow.token_ids]
    assert hidden, "the narrow call should have hidden something"

    recovered = lm.top_next(
        [CONTEXT], top_k=1, top_p=1.0, match_chars=["c"], match_top_k=10
    )[0]
    by_id = dict(zip(recovered.token_ids, recovered.logprobs))
    for token in hidden:
        assert by_id[token] == pytest.approx(reference[token])
    # ...and the mass reported counts what was actually handed back.
    assert recovered.kept_mass == pytest.approx(wide.kept_mass)


def test_an_empty_context_fails_with_something_readable():
    """A forward pass needs at least one token.

    Without this the failure surfaces as an IndexError from inside the tensor
    library, which says nothing about what the caller did wrong.
    """
    lm = cat_lm(CONTEXT)
    with pytest.raises(ValueError, match="nothing to continue from"):
        predict(lm, (), "ca", PredictConfig(k=5, max_rounds=2), COSTS)


def test_repetition_is_refused():
    """Neither signal rejects it, so it has to be rejected outright.

    A base model with little context in front of it loops, and a repeated
    fragment lines up against the keystrokes again at every repeat -- so
    "thisthisthisthis" reads "thisatest" about as well as "this is a test"
    does, and the model is happy with it too.
    """
    from fuzzytype.search import is_degenerate

    for text in (
        "thisthisthis",
        "Helvetica Helvetica Helvetica",
        "the cat the cat sat",
    ):
        assert is_degenerate(text), text
    for text in (
        "this is a test of the new text input system",
        "get back to you as soon as possible",
        "This is a test",
        "hello",
    ):
        assert not is_degenerate(text), text


def test_a_repeating_candidate_never_reaches_the_ranking():
    lm = FakeLM(
        ["<eos>", " ", ".", "ha"],
        {
            (): {"ha": 1.0},
            ("ha",): {"ha": 1.0},
            ("ha", "ha"): {"ha": 1.0},
            ("ha", "ha", "ha"): {" ": 1.0},
            ("ha", "ha", "ha", " "): {"<eos>": 1.0},
        },
        context=(1,),
    )
    found, _ = predict(lm, (1,), "hh", PredictConfig(k=10, max_rounds=6), COSTS)
    assert not [c for c in found if c.text == "hahaha"]


def test_a_word_outside_the_top_k_is_still_reachable():
    """Pruning cannot rescue a token that was never proposed.

    A plain top-k selects on the prior alone, and the word wanted is often
    nowhere near the top: "Hello" sits under three nats from "Here" and far
    outside the top sixty-four, so typing "hel" returned every "Here ..." and
    no "Hello" however long the search ran. Where a new word can begin, a
    second top-k is taken over only the tokens that begin with the next
    unexplained keystroke -- both exact top-k over the same distribution, so
    the search sees more of the right part of it without repricing anything.
    """
    hidden = PredictConfig(k=20, max_rounds=6, child_top_k=1, boundary_top_k=0)
    blind = prefix_lm(CONTEXT)
    found, _ = predict(blind, CONTEXT, "cart", hidden, COSTS)
    assert "cart" not in {c.text for c in found}, "top-k alone should hide it"

    seeing = prefix_lm(CONTEXT)
    found, _ = predict(
        seeing, CONTEXT, "cart",
        PredictConfig(k=20, max_rounds=6, child_top_k=1, boundary_top_k=8),
        COSTS,
    )
    assert "cart" in {c.text for c in found}
