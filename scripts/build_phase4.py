"""Phase 4: freeze the priority-band rule, build the inference artifact, characterise bands, explain.

Usage (in this order):
    python scripts/build_phase4.py freeze-bands   # artifact + cutoffs from the validation reference; validation bands
    python scripts/build_phase4.py test-bands     # SECONDARY: apply the frozen rule to saved test scores (once)
    python scripts/build_phase4.py explain        # SHAP (development background) on validation leads + figures
    python scripts/build_phase4.py band-figure    # redraw the band figure from saved band tables
    python scripts/build_phase4.py threshold-analysis  # confusion matrices at the band operating points

The model is the frozen Phase 3 logistic regression; nothing is fitted or recalibrated here.
`test-bands` reads reports/test_scores.csv and never re-scores the test segment.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev  # noqa: E402
from src import explainability as xp  # noqa: E402
from src import inference as inf  # noqa: E402
from src import plots  # noqa: E402
from src.data import PROJECT_ROOT, RAW_CSV, load_raw, sha256_of  # noqa: E402
from src.features import make_target  # noqa: E402
from src.splits import chronological_split  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
FROZEN_SPEC = REPORTS / "frozen_spec.json"
ARTIFACT = PROJECT_ROOT / "artifacts" / "models" / f"{inf.ARTIFACT_VERSION}.joblib"
ARTIFACT_METADATA = ARTIFACT.with_suffix(".json")
PRIORITY_RULE = REPORTS / "priority_rule.json"
BANDS_VALIDATION = REPORTS / "bands_validation.csv"
BANDS_TEST = REPORTS / "bands_test_secondary.csv"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def _segments(raw):
    split = chronological_split(raw["month"])
    return {name: raw.iloc[index] for name, index in split.items()}


def freeze_bands() -> None:
    if BANDS_TEST.exists():
        raise SystemExit("Test band results exist; the band rule and artifact are frozen and cannot be rebuilt.")
    spec = json.loads(FROZEN_SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "frozen" or spec["primary_family"] != "logreg" or not spec["calibration"]["used"]:
        raise SystemExit("Unexpected frozen spec")
    raw = load_raw()
    segments = _segments(raw)
    calibrated = joblib.load(PROJECT_ROOT / spec["model_files"]["primary_calibrated"])

    metadata = {
        "created": date.today().isoformat(),
        "model": "Logistic regression (C = 1.0) on the 12 primary pre-contact features; frozen in Phase 3",
        "ranking_score": "log-odds (decision function) of the uncalibrated logistic regression",
        "reference_distribution": "ranking scores of the validation period (Apr - May 2009), "
        f"{len(segments['validation'])} leads",
        "periods": {name: {k: spec["segments"][name][k] for k in ("first_period", "last_period", "n_rows")} for name in spec["segments"]},
        "trained_on": "development period (May 2008 - Mar 2009)",
        "calibrated_on": "validation period (Apr - May 2009)",
        "calibration": {k: spec["calibration"][k] for k in ("method", "slope", "intercept")},
        "test_calibration_drift": "test period mean calibrated probability 14.7% vs actual 30.8% (reports/metrics.json)",
        "shap_background": f"{xp.BACKGROUND_SIZE} development-period rows, seed {xp.BACKGROUND_SEED}",
        "frozen_spec_sha256": sha256_of(FROZEN_SPEC),
        "data_sha256": sha256_of(RAW_CSV),
        "library_versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "shap": shap.__version__},
        "limitations": [
            "Ranking is a historical association, not guaranteed conversion.",
            "Probability calibration drifts with the overall conversion rate; recalibrate on recent outcomes before use.",
            "Trained on one Portuguese bank's 2008-2009 campaigns.",
        ],
    }
    scorer = inf.build_scorer(
        calibrated,
        reference_raw=segments["validation"],
        background_raw=xp.background_sample(segments["development"]),
        metadata=metadata,
    )
    inf.save_scorer(scorer, ARTIFACT)
    _write_json(ARTIFACT_METADATA, {**scorer.metadata, "feature_names": scorer.feature_names})

    reference = scorer.reference_scores
    _write_json(
        PRIORITY_RULE,
        {
            "status": "frozen before any band-level test results were computed",
            "rule": "High = top 20% of ranking scores; Medium = next 30%; Low = remaining 50%",
            "shares": inf.BAND_RULE,
            "ranking_score": metadata["ranking_score"],
            "reference": metadata["reference_distribution"],
            "cutoffs": scorer.cutoffs,
            "tie_rule": "score >= cutoff goes to the higher band",
            "batch_mode": "the same 20/30/50 rule applied to the batch's own score distribution",
            "reference_percentiles": {f"p{q}": float(np.quantile(reference, q / 100)) for q in range(0, 101, 5)},
            "artifact": str(ARTIFACT.relative_to(PROJECT_ROOT)),
            "artifact_sha256": sha256_of(ARTIFACT),
        },
    )

    y_val = make_target(segments["validation"]).to_numpy()
    scored = scorer.score(segments["validation"])
    ev.band_table(y_val, scored["priority"]).assign(segment="validation (reference)").to_csv(BANDS_VALIDATION, index=False)
    print(json.dumps(scorer.cutoffs, indent=2))
    print(pd.read_csv(BANDS_VALIDATION).round(3).to_string(index=False))


def test_bands() -> None:
    """SECONDARY operational characterisation: frozen cutoffs applied to the saved Phase 3 test scores."""
    if BANDS_TEST.exists():
        raise SystemExit(f"{BANDS_TEST} exists; test band results are produced once.")
    rule = json.loads(PRIORITY_RULE.read_text(encoding="utf-8"))
    if sha256_of(ARTIFACT) != rule["artifact_sha256"]:
        raise SystemExit("Artifact changed after the band rule was frozen")
    scores = pd.read_csv(REPORTS / "test_scores.csv")
    p = scores["logreg_primary"].clip(1e-12, 1 - 1e-12)
    ranking_score = np.log(p / (1 - p))  # the saved uncalibrated probability's log-odds = decision function
    reference_bands = inf.assign_band(ranking_score, rule["cutoffs"])
    batch_bands = inf.assign_band(ranking_score, inf.band_cutoffs(np.sort(ranking_score)))
    pd.concat(
        [
            ev.band_table(scores["y"], reference_bands).assign(segment="test (secondary): reference cutoffs"),
            ev.band_table(scores["y"], batch_bands).assign(segment="test (secondary): batch-relative 20/30/50"),
        ]
    ).to_csv(BANDS_TEST, index=False)
    print(pd.read_csv(BANDS_TEST).round(3).to_string(index=False))


def explain() -> None:
    scorer = inf.load_scorer(ARTIFACT)
    raw = load_raw()
    validation = _segments(raw)["validation"]
    explainer = scorer.explainer()

    contributions = xp.field_contributions(scorer.pipeline, explainer, validation)
    reconstructed = contributions.sum(axis=1) + xp.base_value(explainer)
    max_error = float(np.max(np.abs(reconstructed - scorer.ranking_score(validation))))
    importance = xp.global_importance(contributions)
    importance.to_csv(REPORTS / "shap_global_importance.csv", index=False)

    # Representative local examples, chosen by rule (not by outcome): for High and for Low, the validation
    # lead whose ranking score is closest to that band's median score.
    scored = scorer.score(validation)
    examples = {}
    for band in ("High", "Low"):
        members = scored[scored["priority"] == band]["ranking_score"]
        index = (members - members.median()).abs().idxmin()
        local = scorer.explain(validation.loc[[index]])
        local.to_csv(REPORTS / f"shap_local_example_{band.lower()}.csv", index=False)
        examples[band] = (index, local, scored.loc[index])

    _write_json(
        REPORTS / "shap_summary.json",
        {
            "explains": "log-odds ranking score of the frozen logistic regression (not the calibrated probability)",
            "background": scorer.metadata["shap_background"],
            "explained_rows": f"validation period, {len(validation)} leads",
            "base_value": xp.base_value(explainer),
            "additivity_max_abs_error": max_error,
            "interpretation": "Contributions are model associations: how much each field pushed this model's score "
            "up or down relative to an average development-period lead. They are not causal effects.",
            "local_examples": {
                band: {
                    "row": int(index),
                    "selection_rule": f"validation lead whose ranking score is closest to the median of the {band} band",
                    "priority": row["priority"],
                    "ranking_score": float(row["ranking_score"]),
                    "historical_percentile": float(row["historical_percentile"]),
                }
                for band, (index, _, row) in examples.items()
            },
        },
    )

    fig = plots.importance_bars(
        importance["field"].tolist(),
        importance["mean_abs_contribution"].to_numpy(),
        "What the ranking model relies on most",
        "Mean absolute SHAP contribution to the model's ranking score (log-odds), validation leads (Apr - May 2009), "
        "development-period background. One-hot categories summed back to their business field. Associations, "
        "not causes.",
    )
    fig.savefig(FIGURES / "shap_global_importance.png")

    for band, (_, local, row) in examples.items():
        fig = plots.local_explanation_bars(
            local["description"].tolist(),
            local["contribution"].to_numpy(),
            f"Why one lead ranks {band} (historical percentile {row['historical_percentile']:.0f})",
            "SHAP contributions to this lead's ranking score relative to an average development-period lead. Bars to "
            "the right pushed the score higher; to the left, lower. Model associations, not causes. Example chosen by "
            f"rule: the validation lead closest to the {band}-band median score.",
        )
        fig.savefig(FIGURES / f"shap_local_example_{band.lower()}.png")
    print(importance.round(3).to_string(index=False))
    for band, (_, local, _) in examples.items():
        print(band)
        print(local[["description", "contribution"]].round(3).to_string(index=False))
    print("additivity max error", max_error)


def band_figure() -> None:
    validation = pd.read_csv(BANDS_VALIDATION)
    test = pd.read_csv(BANDS_TEST)
    test_reference = test[test["segment"].str.contains("reference")]
    fig = plots.band_bars(
        {"Validation (reference)": validation, "Test, reference cutoffs (secondary)": test_reference},
        "Conversion rate by priority band",
        "Bands: High = top 20% of ranking scores, Medium = next 30%, Low = remaining 50%, cutoffs fixed on validation "
        "(Apr - May 2009). Test (May 2009 - Nov 2010) applies the same cutoffs to saved scores; band sizes shift "
        "because the score distribution drifted. Labels show conversion rate and share of leads.",
    )
    fig.savefig(FIGURES / "priority_bands.png")


def threshold_analysis() -> None:
    """Confusion matrices at the two band operating points, computed from the saved band tables only."""
    validation = pd.read_csv(BANDS_VALIDATION)
    test = pd.read_csv(BANDS_TEST)
    frames = [ev.operating_point_confusion(validation).assign(segment="validation (reference)")]
    for segment, group in test.groupby("segment", sort=False):
        frames.append(ev.operating_point_confusion(group).assign(segment=segment))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(REPORTS / "threshold_analysis.csv", index=False)
    print(out.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["freeze-bands", "test-bands", "explain", "band-figure", "threshold-analysis"])
    args = parser.parse_args()
    if args.stage == "freeze-bands":
        freeze_bands()
    elif args.stage == "test-bands":
        test_bands()
        band_figure()
    elif args.stage == "explain":
        explain()
    elif args.stage == "threshold-analysis":
        threshold_analysis()
    else:
        band_figure()


if __name__ == "__main__":
    main()
