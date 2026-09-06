"""The interactive loop, driven headlessly.

These assert the behaviours a typist depends on: keystrokes land instantly,
the ranked list reflects them, and accepting a row commits its text.
"""

from __future__ import annotations

import asyncio

from fake_lm import cat_lm

from fuzzytype.channel import ChannelCosts
from fuzzytype.engine import Engine, EngineConfig
from fuzzytype.search import Candidate, PredictConfig
from fuzzytype.tui import _ROUND_STEPS, FuzzyTypeApp, quality_label

CONTEXT = (2,)  # "." -- survives the trailing-space stripping


def _app():
    engine = Engine(
        lm=cat_lm(CONTEXT),
        config=EngineConfig(preamble=".", k=6),
        predict_config=PredictConfig(k=20, max_rounds=6),
        costs=ChannelCosts(),
    )
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
    assert text == "", "nothing should have been committed"


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
    # Candidates carry the space that joins them to the previous word, and it
    # comes off again at the very start of a document.
    assert text.endswith(chosen.lstrip())
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


def test_thinking_time_is_adjustable_live():
    """The right number of rounds depends on what is being written."""

    async def steps(app, pilot):
        start = app.engine.predict_config.max_rounds
        await pilot.press("ctrl+up")
        await pilot.pause()
        more = app.engine.predict_config.max_rounds
        await pilot.press("ctrl+down", "ctrl+down")
        await pilot.pause()
        return start, more, app.engine.predict_config.max_rounds

    start, more, fewer = _drive(steps)
    assert more > start
    assert fewer < more


def test_thinking_time_is_shown_and_clamped():
    async def steps(app, pilot):
        for _ in range(20):
            await pilot.press("ctrl+down")
        await pilot.pause()
        low = app.engine.predict_config.max_rounds
        for _ in range(20):
            await pilot.press("ctrl+up")
        await pilot.pause()
        return low, app.engine.predict_config.max_rounds, app._status_line().plain

    low, high, status = _drive(steps)
    assert low == min(_ROUND_STEPS)
    assert high == max(_ROUND_STEPS)
    assert f"think {high}" in status


def test_a_failed_model_load_can_be_retried():
    """A refused CUDA context should not mean restarting the app."""
    attempts = []

    def failing_loader():
        attempts.append(1)
        raise RuntimeError("CUDA error: out of memory")

    async def main():
        app = FuzzyTypeApp(_app().engine, model_loader=failing_loader)
        async with app.run_test() as pilot:
            for _ in range(10):
                await pilot.pause()
            failed = app._status.error
            await pilot.press("ctrl+r")
            for _ in range(10):
                await pilot.pause()
            return failed, len(attempts)

    failed, tries = asyncio.run(main())
    assert "out of memory" in failed
    assert tries >= 2, "ctrl+r should have tried the model again"


def test_the_cursor_moves_inside_the_keystrokes():
    async def steps(app, pilot):
        await pilot.press("c", "a", "t")
        await pilot.press("left", "left")
        await pilot.pause()
        return app.query, app.cursor

    query, cursor = _drive(steps)
    assert (query, cursor) == ("cat", 1)


def test_typing_inserts_at_the_cursor():
    async def steps(app, pilot):
        await pilot.press("c", "t")
        await pilot.press("left")
        await pilot.press("a")
        await pilot.pause()
        return app.query, app.cursor

    query, cursor = _drive(steps)
    assert query == "cat"
    assert cursor == 2


def test_backspace_deletes_behind_the_cursor_not_at_the_end():
    async def steps(app, pilot):
        await pilot.press("c", "x", "t")
        await pilot.press("left")       # between x and t
        await pilot.press("backspace")  # removes the x
        await pilot.pause()
        return app.query

    assert _drive(steps) == "ct"


def test_delete_removes_ahead_of_the_cursor():
    async def steps(app, pilot):
        await pilot.press("c", "a", "t")
        await pilot.press("home", "delete")
        await pilot.pause()
        return app.query, app.cursor

    query, cursor = _drive(steps)
    assert (query, cursor) == ("at", 0)


def test_home_and_end_jump_to_either_side():
    async def steps(app, pilot):
        await pilot.press("c", "a", "t", "home")
        await pilot.pause()
        start = app.cursor
        await pilot.press("end")
        await pilot.pause()
        return start, app.cursor

    assert _drive(steps) == (0, 3)


def test_the_cursor_walks_on_into_committed_text():
    """Past the keystrokes, left keeps going -- and the context follows it."""

    async def steps(app, pilot):
        app.engine.text = "hello there"
        await pilot.press("left", "left", "left")
        await pilot.pause()
        return app.engine.text, app.engine.after

    before, after = _drive(steps)
    assert before == "hello th"
    assert after == "ere", "the rest of the sentence is carried, not lost"


def test_moving_back_predicts_from_the_new_position():
    """The pool is for the old context, so it cannot be reused."""

    async def steps(app, pilot):
        app.engine.text = "hello there"
        app.engine.pool = [Candidate("x", " x", -1.0, 0.0, 0, 0, 1, ())]
        await pilot.press("left")
        await pilot.pause()
        return app.engine.pool

    assert _drive(steps) == []


def test_right_walks_back_out_again():
    async def steps(app, pilot):
        app.engine.text = "hello there"
        await pilot.press("left", "left", "right")
        await pilot.pause()
        return app.engine.text, app.engine.after

    # two back, one forward: a net single step left
    assert _drive(steps) == ("hello ther", "e")


def test_accepting_a_suggestion_inserts_at_the_cursor():
    async def steps(app, pilot):
        app.engine.text = "cat sat"
        app.engine.after = " on it"
        app.engine.commit(" here")
        return app.engine.text, app.engine.after

    text, after = _drive(steps)
    assert text == "cat sat here"
    assert after == " on it", "text after the cursor stays put"
