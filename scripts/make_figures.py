"""Build the Phase 2-3 figures from saved reports (no model fitting, no re-scoring).

Usage: python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import plots  # noqa: E402
from src.data import PROJECT_ROOT  # noqa: E402
from src.features import FEATURE_LABELS, availability_table  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    test = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((REPORTS / "random_split_diagnostic.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(REPORTS / "test_scores.csv")

    fig = plots.gains_curve(
        scores["y"],
        {"Logistic regression (primary)": scores["logreg_primary"], "XGBoost (comparison)": scores["xgboost_comparison"]},
        "Cumulative gains on the untouched test period",
        "Test segment: May 2009 - Nov 2010, 8,239 leads, 30.8% converted. Models fitted on May 2008 - Mar 2009. "
        "Dots mark the top 10%, 20% and 30%.",
    )
    fig.savefig(FIGURES / "test_gains_curve.png")

    fig = plots.feature_audit_graphic(
        availability_table(),
        FEATURE_LABELS,
        "Leakage audit: which fields the model may use before a call is dialled",
        "Decision point: a lead is scored immediately before the outbound call is dialled. Availability was judged on "
        "that basis. Fields kept out were excluded for usefulness or undocumented timing, not leakage. "
        "Source: src/features.py (single source of truth) and reports/feature_availability.csv.",
    )
    fig.savefig(FIGURES / "feature_audit.png")

    calibration = pd.read_csv(REPORTS / "calibration_test.csv")
    fig = plots.reliability_plot(
        {
            "Before calibration": calibration[calibration["stage"] == "before calibration"],
            "After Platt calibration": calibration[calibration["stage"] != "before calibration"],
        },
        test["base_rate"],
        "Probability calibration drifted on the later test period",
        "Deciles of predicted probability. Platt mapping fitted on Apr-May 2009 (12.8% converted) and applied "
        "unchanged; the test period converted at 30.8%, so probabilities remain too low (temporal calibration drift).",
    )
    fig.savefig(FIGURES / "test_calibration.png")

    def lifts(result):
        return [r["lift"] for r in result["models"]["logreg_primary"]["top_k"]]

    fig = plots.lift_comparison(
        {"Chronological test (headline)": lifts(test), "Random split (optimistic diagnostic)": lifts(diagnostic)},
        title="Same recipe, two evaluation designs (logistic regression)",
        caption="Random split: stratified 70/10/20 over all periods, so training and evaluation share months. "
        "Diagnostic only; not used for any model choice.",
    )
    fig.savefig(FIGURES / "lift_chronological_vs_random.png")

    spec = json.loads((REPORTS / "frozen_spec.json").read_text(encoding="utf-8"))
    period_summary = pd.read_csv(REPORTS / "period_summary.csv")
    fig = plots.drift_by_period(
        period_summary,
        {name: seg["first_row"] for name, seg in spec["segments"].items()},
        "Conversion rate by month: the base rate drifts sharply over time",
        "Months in file order (May 2008 - Nov 2010). Shaded band = validation (April and part of May 2009); May 2009 "
        "is split between validation and test. Development 5.2%, validation 12.8%, test 30.8% converted. "
        "Some months are tiny (e.g. Oct 2008: 67 rows; Dec 2008: 10).",
    )
    fig.savefig(FIGURES / "conversion_rate_by_month.png")
    print("Figures written to", FIGURES)


if __name__ == "__main__":
    main()
