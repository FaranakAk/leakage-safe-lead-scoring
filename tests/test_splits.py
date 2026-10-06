import numpy as np
import pandas as pd
import pytest

from src.splits import (
    chronological_split,
    expanding_window_folds,
    first_test_row,
    month_run_starts,
    validation_start,
)


def _months(runs: list[tuple[str, int]]) -> pd.Series:
    return pd.Series([m for m, n in runs for _ in range(n)], name="month")


def test_first_test_row_matches_phase1_definition():
    assert first_test_row(41188) == 32949  # int(0.6 n) + int(0.2 n)


def test_month_run_starts():
    months = _months([("may", 3), ("jun", 2), ("may", 4)])
    assert month_run_starts(months).tolist() == [3, 5]


def test_validation_start_is_nearest_month_boundary_before_test():
    # 100 rows; target 70; boundaries at 60 and 75 -> 75 is nearer.
    months = _months([("may", 60), ("jun", 15), ("jul", 25)])
    assert validation_start(months) == 75


def test_validation_start_ignores_boundaries_inside_test_segment():
    # first test row is 80; the boundary at 85 is excluded even though it is a month start.
    months = _months([("may", 50), ("jun", 35), ("jul", 15)])
    assert validation_start(months) == 50


def test_validation_start_tie_goes_to_earlier_boundary():
    months = _months([("may", 65), ("jun", 10), ("jul", 25)])  # boundaries 65 and 75, target 70
    assert validation_start(months) == 65


def test_validation_start_requires_a_boundary():
    with pytest.raises(ValueError):
        validation_start(_months([("may", 100)]))


def test_chronological_split_is_contiguous_ordered_and_complete():
    months = _months([("may", 60), ("jun", 15), ("jul", 25)])
    split = chronological_split(months)
    development, validation, test = split["development"], split["validation"], split["test"]
    assert np.array_equal(np.concatenate([development, validation, test]), np.arange(100))
    assert (development[-1], validation[0], validation[-1], test[0]) == (74, 75, 79, 80)


def test_split_does_not_depend_on_anything_but_month_order():
    months = _months([("may", 60), ("jun", 15), ("jul", 25)])
    first = chronological_split(months)
    second = chronological_split(months.copy())
    for name in first:
        assert np.array_equal(first[name], second[name])


def test_expanding_window_folds_train_only_on_earlier_rows():
    folds = expanding_window_folds(1000, n_folds=4)
    assert len(folds) == 4
    previous_fit_size = 0
    for fit_index, eval_index in folds:
        assert fit_index[0] == 0
        assert fit_index.max() < eval_index.min()
        assert eval_index[0] == fit_index[-1] + 1
        assert np.array_equal(eval_index, np.arange(eval_index[0], eval_index[-1] + 1))
        assert len(fit_index) > previous_fit_size
        previous_fit_size = len(fit_index)
    assert folds[-1][1][-1] == 999


def test_expanding_window_folds_reject_too_few_rows():
    with pytest.raises(ValueError):
        expanding_window_folds(3, n_folds=4)


# --- the real file ----------------------------------------------------------


def test_real_split_boundaries(raw_full):
    split = chronological_split(raw_full["month"])
    assert split["development"][-1] == 27971
    assert split["validation"][0] == 27972  # first row of April 2009
    assert split["test"][0] == 32949  # unchanged since Phase 1
    assert raw_full["month"].iloc[27971] != raw_full["month"].iloc[27972]
