"""The noisy channel: is it a likelihood, and is it safe to prune with?"""

from __future__ import annotations

import random

import pytest

from fuzzytype.channel import (
    ChannelCosts,
    initial_column,
    initial_row,
    match,
    neighbor_map,
    push_candidate_char,
    push_query_char,
)

COSTS = ChannelCosts()


def test_exact_prefix_is_free_and_the_tail_is_the_prediction():
    result = match("the", "there", COSTS)
    assert result.cost == 0.0
    assert result.consumed == 3  # "the" typed, "re" predicted


def test_nothing_typed_costs_nothing():
    # Untyped characters are the prediction, not an error.
    assert match("", "anything at all", COSTS).cost == 0.0


def test_abbreviation_costs_one_skip_per_omitted_character():
    # "wte" -> "write" leaves out "r" and "i".
    assert match("wte", "write", COSTS).cost == pytest.approx(2 * COSTS.skip)


def test_a_spurious_keystroke_costs_a_deletion():
    assert match("thex", "the", COSTS).cost == pytest.approx(COSTS.delete)


def test_wrong_case_is_much_cheaper_than_a_wrong_letter():
    assert match("paris", "Paris", COSTS).cost == pytest.approx(COSTS.case)
    assert COSTS.case < COSTS.substitute


def test_neighbouring_key_typo_is_cheaper_than_an_arbitrary_one():
    near = ChannelCosts.for_layout("colemak-dh")
    assert "s" in neighbor_map("colemak-dh")["r"]
    assert near.substitution("r", "s") == near.substitute_near
    assert near.substitution("r", "m") == near.substitute
    assert near.substitute_near < near.substitute


def test_layout_must_be_known():
    with pytest.raises(ValueError, match="unknown layout"):
        neighbor_map("dvorak-ish")


def test_budget_grows_with_the_query():
    assert COSTS.budget(10) > COSTS.budget(2)


def _full_grid_by_columns(query, candidate, costs):
    column = initial_column(len(query), costs)
    grid = [column]
    for ch in candidate:
        column = push_candidate_char(column, query, ch, costs)
        grid.append(column)
    return grid


def _full_grid_by_rows(query, candidate, costs):
    row = initial_row(len(candidate), costs)
    rows = [row]
    for ch in query:
        row = push_query_char(row, candidate, ch, costs)
        rows.append(row)
    # transpose to the same shape the column walk produces
    return [tuple(r[j] for r in rows) for j in range(len(candidate) + 1)]


@pytest.mark.parametrize("seed", range(25))
def test_both_grid_directions_compute_the_same_numbers(seed):
    """The search grows the candidate; the UI grows the query. Same grid.

    They are separate code paths precisely because they extend along
    different axes, so nothing but a test keeps them honest.
    """
    rng = random.Random(seed)
    alphabet = "abcdeR "
    query = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6)))
    candidate = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8)))
    by_columns = _full_grid_by_columns(query, candidate, COSTS)
    by_rows = _full_grid_by_rows(query, candidate, COSTS)
    # Tolerance, not equality: the two walks add the same costs in a
    # different order, so they differ in the last bit and nothing else.
    assert len(by_columns) == len(by_rows)
    for left, right in zip(by_columns, by_rows):
        assert left == pytest.approx(right)


@pytest.mark.parametrize("seed", range(25))
def test_column_minimum_never_decreases(seed):
    """This is what makes pruning sound.

    A branch is abandoned when its column minimum passes the error budget. If
    the minimum could fall again later, that would discard reachable
    candidates -- so the monotonicity is the property to assert, not the
    pruning code.
    """
    rng = random.Random(seed + 100)
    query = "".join(rng.choice("abcde") for _ in range(rng.randint(1, 6)))
    candidate = "".join(rng.choice("abcde ") for _ in range(12))
    grid = _full_grid_by_columns(query, candidate, COSTS)
    minima = [min(col) for col in grid]
    assert minima == sorted(minima)


def test_cost_reads_as_a_probability():
    result = match("wte", "write", COSTS)
    assert 0.0 < result.likelihood < 1.0
    assert match("write", "write", COSTS).likelihood == 1.0
