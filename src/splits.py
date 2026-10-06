"""Order-preserving splits.

Rows are in date order and are never shuffled for the primary evaluation.

* Test: the final 20% of rows (fixed since Phase 1; never moved).
* Development / validation: the remaining rows are divided at the month
  boundary closest to 70% of all rows, chosen from chronology alone.
* Inner folds: expanding-window folds inside the development segment, used for
  tuning only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_DEVELOPMENT_FRACTION = 0.7
N_INNER_FOLDS = 4


def first_test_row(n_rows: int) -> int:
    """First test row. Kept identical to the Phase 1 definition so the test segment never moves."""
    return int(n_rows * 0.6) + int(n_rows * 0.2)


def month_run_starts(month: pd.Series) -> np.ndarray:
    """Positions where the month label changes (the start of each contiguous month run), excluding 0."""
    values = month.to_numpy()
    return np.flatnonzero(values[1:] != values[:-1]) + 1


def validation_start(month: pd.Series, target_fraction: float = TARGET_DEVELOPMENT_FRACTION) -> int:
    """Month boundary closest to ``target_fraction`` of rows, strictly before the test segment.

    Uses row positions only; no target or model information. Ties go to the earlier boundary.
    """
    n_rows = len(month)
    candidates = month_run_starts(month)
    candidates = candidates[candidates < first_test_row(n_rows)]
    if len(candidates) == 0:
        raise ValueError("no month boundary before the test segment")
    target = target_fraction * n_rows
    return int(candidates[np.argmin(np.abs(candidates - target))])


def chronological_split(month: pd.Series, target_fraction: float = TARGET_DEVELOPMENT_FRACTION) -> dict[str, np.ndarray]:
    """Positional indices for contiguous development / validation / test segments, in row order."""
    n_rows = len(month)
    val_start = validation_start(month, target_fraction)
    first_test = first_test_row(n_rows)
    return {
        "development": np.arange(0, val_start),
        "validation": np.arange(val_start, first_test),
        "test": np.arange(first_test, n_rows),
    }


def expanding_window_folds(n_rows: int, n_folds: int = N_INNER_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: each fold trains on all earlier rows and evaluates on the next block.

    The rows are split into ``n_folds + 1`` contiguous blocks; fold k trains on
    blocks 0..k and evaluates on block k + 1.
    """
    if n_folds < 1 or n_rows < n_folds + 1:
        raise ValueError("need at least one row per block")
    edges = np.linspace(0, n_rows, n_folds + 2).astype(int)
    return [(np.arange(0, edges[k + 1]), np.arange(edges[k + 1], edges[k + 2])) for k in range(n_folds)]
