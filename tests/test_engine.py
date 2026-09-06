"""Engine: committed text, seeds, and when a fresh decode is worth it."""

from __future__ import annotations

from fake_lm import cat_lm

from fuzzytype.channel import ChannelCosts
from fuzzytype.engine import Engine, EngineConfig
from fuzzytype.search import Candidate, PredictConfig

CONTEXT = (1,)


def _engine(**config):
    # No preamble: the fake vocabulary cannot spell one, and these tests are
    # about the engine's bookkeeping rather than the model's register.
    config.setdefault("preamble", "")
    return Engine(
        lm=cat_lm(CONTEXT),
        config=EngineConfig(**config),
        predict_config=PredictConfig(k=20, max_rounds=6),
        costs=ChannelCosts(),
    )


def test_accepting_a_suggestion_appends_it_and_invalidates_the_pool():
    engine = _engine()
    engine.text = "I saw"
    engine.pool = [
        Candidate(" a cat", " a cat", -1.0, 0.0, 0, 1, ())
    ]
    engine.commit(" a cat")
    assert engine.text == "I saw a cat"
    assert engine.pool == []


def test_the_first_word_does_not_start_with_a_space():
    """Candidates carry the space that joins them to the previous word."""
    engine = _engine()
    engine.commit(" cat")
    assert engine.text == "cat"
    engine.commit(" sat")
    assert engine.text == "cat sat"


def test_literal_commit_adds_the_missing_space_exactly_once():
    engine = _engine()
    engine.commit_literal("hello")
    assert engine.text == "hello"
    engine.commit_literal("there")
    assert engine.text == "hello there"


def test_literal_commit_does_not_double_a_space_already_present():
    engine = _engine()
    engine.text = "hello "
    engine.commit_literal("there")
    assert engine.text == "hello there"


def test_seeds_carry_a_leading_space_only_mid_sentence():
    engine = _engine()
    assert engine.seeds("cat") == ["cat", "Cat"]
    engine.text = "I saw a"
    assert engine.seeds("cat") == [" cat", " Cat"]


def test_seeds_offer_the_capitalised_spelling_for_names():
    """A typist does not reach for shift; "alic" must still reach "Alice"."""
    engine = _engine()
    engine.text = "write to"
    assert " Alic" in engine.seeds("alic")


def test_seeds_are_empty_with_nothing_typed():
    assert _engine().seeds("") == []


def test_a_decode_is_only_requested_when_the_pool_stops_explaining():
    engine = _engine()
    engine.text = " "  # the fake model's context token
    engine.refresh("")
    assert engine.pool, "the fake model should produce candidates"
    # An ordinary prefix of a pooled candidate needs no GPU work...
    assert not engine.needs_refresh("ca")
    # ...but something the pool cannot explain does.
    assert engine.needs_refresh("zzzz")


def test_an_empty_pool_always_needs_a_decode():
    assert _engine().needs_refresh("")


def test_context_is_truncated_to_bound_the_forward_pass():
    engine = _engine(max_context_tokens=4)
    engine.text = " " + " cat" * 50
    assert len(engine.context_ids()) == 4


def test_backspacing_committed_text_invalidates_the_pool():
    engine = _engine()
    engine.text = "hello"
    engine.pool = [Candidate("x", " x", -1.0, 0.0, 0, 1, ())]
    engine.backspace_text()
    assert engine.text == "hell"
    assert engine.pool == []
