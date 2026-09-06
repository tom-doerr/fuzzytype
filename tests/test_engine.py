"""Engine: committed text, seeds, and when a fresh decode is worth it."""

from __future__ import annotations

from fake_lm import cat_lm

from fuzzytype.channel import ChannelCosts
from fuzzytype.engine import Engine, EngineConfig
from fuzzytype.search import Candidate, PredictConfig
from fuzzytype.shorthand import build_prompt

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


def test_the_pool_accumulates_across_decodes():
    """Editing the query must not throw away what is already known.

    A candidate's prior is P(text | context) and does not depend on the
    keystrokes, so anything found under an earlier query is still exactly as
    probable. Replacing the pool on every edit meant re-deriving the same
    phrases from nothing each time.
    """
    engine = _engine()
    engine.text = " "
    engine.refresh("")
    first = {c.text for c in engine.pool}
    assert first
    engine.refresh("ca")
    assert first <= {c.text for c in engine.pool}


def test_the_pool_is_bounded():
    engine = _engine(max_pool=2)
    engine.text = " "
    engine.refresh("")
    engine.refresh("ca")
    assert len(engine.pool) <= 2


def test_the_next_decode_can_be_given_what_is_already_known():
    """So it extends known phrases instead of re-deriving them.

    Off by default -- measured as a quarter more GPU for a tenth more
    candidates -- but the mechanism has to keep working.
    """
    engine = _engine(resume_seeds=8)
    engine.text = " "
    engine.refresh("")
    known = {c.raw for c in engine.pool}
    assert known & set(engine.seeds("ca"))


def _prompt_engine(**config):
    config.setdefault("preamble", "")
    config.setdefault("mode", "prompt")
    return Engine(
        lm=cat_lm(CONTEXT),
        config=EngineConfig(**config),
        predict_config=PredictConfig(k=20, max_rounds=6),
        costs=ChannelCosts(),
    )


def test_prompt_mode_puts_the_keystrokes_in_the_prompt():
    """Not in a hand-built error model -- the LM does the expanding."""
    engine = _prompt_engine()
    engine.text = "some context"
    prompt = build_prompt(engine.text, "ca", engine.examples)
    assert "shorthand: ca" in prompt
    assert prompt.rstrip().endswith("full text:")


def test_prompt_mode_does_not_score_the_keystrokes_twice():
    """The prior already accounts for them; the channel would double count."""
    engine = _prompt_engine()
    engine.pool = [Candidate("cat", " cat", -1.0, 0.0, 0, 1, ())]
    scored, _ = engine.suggest("zzzz")
    assert scored, "an unrelated query must not be filtered out by the channel"

    assisted = _prompt_engine(channel_assist=True)
    assisted.pool = list(engine.pool)
    assert assisted.suggest("zzzz")[0] == []


def _with_fake_prompt(engine):
    """Isolate re-pricing from prompt construction.

    The toy vocabulary cannot spell an English prompt, and prompt building is
    asserted separately; what matters here is that the pool is re-priced,
    reordered and bounded.
    """
    engine.prefix_ids = lambda query: list(CONTEXT)
    return engine


def test_rescoring_reprices_without_searching_again():
    engine = _with_fake_prompt(_prompt_engine())
    engine.text = " "
    engine.pool = [
        Candidate("cat", " cat", -99.0, 0.0, 0, 1, ()),
        Candidate("car", " car", -99.0, 0.0, 0, 1, ()),
    ]
    assert engine.rescore("ca") == 2
    assert all(c.logprob > -99.0 for c in engine.pool)
    assert [c.logprob for c in engine.pool] == sorted(
        (c.logprob for c in engine.pool), reverse=True
    )


def test_rescoring_is_bounded_and_drops_the_tail():
    engine = _with_fake_prompt(_prompt_engine(max_rescore=1))
    engine.text = " "
    engine.pool = [
        Candidate("cat", " cat", -1.0, 0.0, 0, 1, ()),
        Candidate("car", " car", -2.0, 0.0, 0, 1, ()),
    ]
    assert engine.rescore("ca") == 1
    assert len(engine.pool) == 1


def test_channel_mode_does_not_rescore():
    """Its priors do not depend on the keystrokes, so there is nothing to do."""
    engine = _engine()
    engine.pool = [Candidate("cat", " cat", -1.0, 0.0, 0, 1, ())]
    assert engine.rescore("ca") == 0
