"""Descriptive checks on the raw data: schema, ordering, class balance, sentinels.

These are audit-only. In particular the inferred calendar period is used to
verify row ordering and describe drift; it is not a model feature.
"""

from __future__ import annotations

import pandas as pd

from src.features import PDAYS_SENTINEL, RAW_COLUMNS, TARGET
from src.splits import chronological_split

EXPECTED_ROWS = 41188
START_YEAR = 2008
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
ECONOMIC_COLUMNS = ("emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")


def check_schema(raw: pd.DataFrame) -> dict:
    expected = [c.name for c in RAW_COLUMNS]
    return {
        "n_rows": int(len(raw)),
        "n_columns": int(raw.shape[1]),
        "columns_match_expected": list(raw.columns) == expected,
        "missing_columns": [c for c in expected if c not in raw.columns],
        "unexpected_columns": [c for c in raw.columns if c not in expected],
        "n_nan_cells": int(raw.isna().sum().sum()),
        "n_exact_duplicate_rows": int(raw.duplicated().sum()),
    }


def infer_period(raw: pd.DataFrame, start_year: int = START_YEAR) -> pd.Series:
    """Infer a 'YYYY-MM' label per row from row order.

    Assumes rows are in date order: the year advances whenever the month
    number decreases from one row to the next.
    """
    month_number = raw["month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    year = start_year + (month_number.diff() < 0).cumsum()
    return (year.astype(str) + "-" + month_number.astype(str).str.zfill(2)).rename("period")


def period_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per contiguous run of the same month, in row order."""
    block = (raw["month"] != raw["month"].shift()).cumsum()
    grouped = raw.assign(period=infer_period(raw), converted=raw[TARGET].eq("yes")).groupby(block)
    summary = grouped.agg(
        period=("period", "first"),
        first_row=("period", lambda s: int(s.index[0])),
        n_rows=("converted", "size"),
        conversion_rate=("converted", "mean"),
    )
    for column in ECONOMIC_COLUMNS:
        summary[f"n_unique_{column}"] = grouped[column].nunique()
    return summary.reset_index(drop=True)


def check_ordering(raw: pd.DataFrame) -> dict:
    """Is the month sequence consistent with the documented May 2008 - Nov 2010 date order?"""
    summary = period_summary(raw)
    periods = summary["period"].tolist()
    return {
        "n_month_runs": len(periods),
        "each_period_is_one_contiguous_run": len(set(periods)) == len(periods),
        "first_period": periods[0],
        "last_period": periods[-1],
        "consistent_with_documented_range": periods[0] == "2008-05" and periods[-1] == "2010-11",
    }


def class_balance(raw: pd.DataFrame) -> dict:
    converted = raw[TARGET].eq("yes")
    return {
        "n_yes": int(converted.sum()),
        "n_no": int((~converted).sum()),
        "conversion_rate": float(converted.mean()),
    }


def contiguous_segment_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Describe the contiguous development/validation/test segments (features and base rate only)."""
    period = infer_period(raw)
    rows = []
    for name, index in chronological_split(raw["month"]).items():
        segment = raw.iloc[index]
        pdays_recorded = segment["pdays"] != PDAYS_SENTINEL
        previous_gt_0 = segment["previous"] > 0
        rows.append(
            {
                "segment": name,
                "first_row": int(index[0]),
                "last_row": int(index[-1]),
                "n_rows": len(segment),
                "share_of_all_rows": len(segment) / len(raw),
                "first_period": period.iloc[index[0]],
                "last_period": period.iloc[index[-1]],
                "conversion_rate": float(segment[TARGET].eq("yes").mean()),
                "n_pdays_recorded": int(pdays_recorded.sum()),
                "share_pdays_recorded": float(pdays_recorded.mean()),
                "n_previous_gt_0": int(previous_gt_0.sum()),
                "share_previous_gt_0": float(previous_gt_0.mean()),
                "n_poutcome_success": int(segment["poutcome"].eq("success").sum()),
                "share_cellular": float(segment["contact"].eq("cellular").mean()),
                "months_present": " ".join(segment["month"].unique()),
            }
        )
    return pd.DataFrame(rows)


def unknown_shares(raw: pd.DataFrame) -> dict:
    categorical = raw.select_dtypes(include=["object", "string"]).drop(columns=[TARGET], errors="ignore")
    return {column: float(share) for column, share in categorical.eq("unknown").mean().items()}


def sentinel_checks(raw: pd.DataFrame) -> dict:
    no_pdays = raw["pdays"] == PDAYS_SENTINEL
    return {
        "campaign_min": int(raw["campaign"].min()),
        "campaign_max": int(raw["campaign"].max()),
        "share_pdays_sentinel": float(no_pdays.mean()),
        "pdays_max_excluding_sentinel": int(raw.loc[~no_pdays, "pdays"].max()),
        "n_previous_gt_0_with_pdays_sentinel": int((no_pdays & (raw["previous"] > 0)).sum()),
        "poutcome_nonexistent_iff_previous_0": bool(
            (raw["poutcome"].eq("nonexistent") == raw["previous"].eq(0)).all()
        ),
        "n_duration_0": int((raw["duration"] == 0).sum()),
        "n_duration_0_converted": int(((raw["duration"] == 0) & raw[TARGET].eq("yes")).sum()),
        "n_default_yes": int(raw["default"].eq("yes").sum()),
    }


def run_audit(raw: pd.DataFrame) -> dict:
    return {
        "schema": check_schema(raw),
        "ordering": check_ordering(raw),
        "class_balance": class_balance(raw),
        "unknown_shares": unknown_shares(raw),
        "sentinels": sentinel_checks(raw),
        "economic_unique_values": {c: int(raw[c].nunique()) for c in ECONOMIC_COLUMNS},
    }
