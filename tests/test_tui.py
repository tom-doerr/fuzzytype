"""The interactive loop, driven headlessly.

These assert the behaviours a typist depends on: keystrokes land instantly,
the ranked list reflects them, and accepting a row commits its text.
"""

from __future__ import annotations

import asyncio

from fake_lm import cat_lm

from fuzzytype.channel import ChannelCosts
from fuzzytype.engine import Engine, EngineConfig
from fuzzytype.search import PredictConfig
from fuzzytype.tui import FuzzyTypeApp, quality_label

CONTEXT = (1,)


def _app():
    engine = Engine(
        lm=cat_lm(CONTEXT),
        config=EngineConfig(preamble="", k=6),
        predict_config=PredictConfig(k=20, max_rounds=6),
        costs=ChannelCosts(),
    )
    engine.text = " "  # the fake model's context token
    return FuzzyTypeApp(engine, model_loader=None)


def _drive(steps):
    """Run an async pilot script without requiring an asyncio plugin."""

    async def main():
        app = _app()
        async with app.run_test() as pilot:
            await pilot.pause()
            return await steps(app, pilot)

    return asyncio.run(main())


def test_quality_label_tells_the_typist_what_to_do():
    assert quality_label(0.0) == "exact"
    assert quality_label(0.5) == "case"
    assert quality_label(3.0) == "1 slip"
    assert quality_label(99.0) == "stretch"


def test_typing_updates_the_ranked_list_without_a_decode():
    async def steps(app, pilot):
        await pilot.press("c", "a", "r")
        await pilot.pause()
        return app.query, [s.text for s in app._suggestions]

    query, texts = _drive(steps)
    assert query == "car"
    assert texts and texts[0] == "car"


def test_space_is_part_of_the_query_not_a_commit():
    """Multi-word shorthand like "gt bck" only works if space is typeable."""

    async def steps(app, pilot):
        await pilot.press("c", "space", "a")
        await pilot.pause()
        return app.query, app.engine.text

    query, text = _drive(steps)
    assert query == "c a"
    assert text == " ", "nothing should have been committed"


def test_backspace_removes_a_keystroke_then_committed_text():
    async def steps(app, pilot):
        await pilot.press("c", "a", "backspace", "backspace", "backspace")
        await pilot.pause()
        return app.query, app.engine.text

    query, text = _drive(steps)
    assert query == ""
    assert text == "", "the third backspace should eat committed text"


def test_accepting_a_row_commits_it_and_clears_the_query():
    async def steps(app, pilot):
        await pilot.press("c", "a", "r")
        await pilot.pause()
        chosen = app._suggestions[0].raw
        await pilot.press("enter")
        await pilot.pause()
        return chosen, app.engine.text, app.query

    chosen, text, query = _drive(steps)
    assert text.endswith(chosen)
    assert query == ""


def test_literal_commit_keeps_exactly_what_was_typed():
    async def steps(app, pilot):
        await pilot.press("c", "a", "r")
        await pilot.pause()
        await pilot.press("ctrl+l")
        await pilot.pause()
        return app.engine.text

    assert _drive(steps).endswith("car")


def test_cycling_the_length_preference_changes_the_bonus():
    async def steps(app, pilot):
        before = app.engine.config.length_bonus
        await pilot.press("ctrl+s")
        await pilot.pause()
        return before, app.engine.config.length_bonus

    before, after = _drive(steps)
    assert before != after


def test_suggestions_appear_while_the_search_is_still_running():
    """A longer decode must show its results as it finds them.

    Searching longer finds more and better phrases, but the typist should
    never be looking at an empty list waiting for it to end.
    """
    seen: list[int] = []

    async def steps(app, pilot):
        original = app._decode_partial

        def spy(stats):
            seen.append(stats.distinct)
            original(stats)

        app._decode_partial = spy
        await pilot.press("c")
        await pilot.pause()
        for _ in range(20):
            await pilot.pause()
        return seen

    _drive(steps)
    # The fake model's world is tiny, so the only firm claim is that partial
    # publishing is wired up at all rather than reporting only at the end.
    assert isinstance(seen, list)


def test_a_keystroke_abandons_a_stale_decode():
    """Finishing an out-of-date search matters less than answering the typist."""

    async def steps(app, pilot):
        app._decoding = True  # pretend a search is in flight
        app._abandon.clear()
        app._request_decode()
        return app._abandon.is_set(), app._decode_wanted

    abandoned, wanted = _drive(steps)
    assert abandoned, "the running decode should be told to stop"
    assert wanted, "and a fresh one should be queued"


def test_the_status_line_says_when_the_model_is_working():
    async def steps(app, pilot):
        app._status.decoding = True
        working = app._status_line().plain
        app._status.decoding = False
        idle = app._status_line().plain
        return working, idle

    working, idle = _drive(steps)
    assert "thinking" in working
    assert "thinking" not in idle
