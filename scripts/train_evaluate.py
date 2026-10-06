"""Phases 2-3: develop, freeze, then score the untouched test segment once.

Usage:
    python scripts/train_evaluate.py develop   # inner CV on development, selection + calibration on validation -> draft spec
    python scripts/train_evaluate.py freeze    # explicit, reviewed step: draft spec becomes the frozen spec
    python scripts/train_evaluate.py test      # one-time scoring of the final 20%
    python scripts/train_evaluate.py random-split-diagnostic   # only after test; optimism diagnostic

`develop` never uses the test rows. `test` requires a frozen spec, refuses to
run twice, and never refits on test results. Once test results exist, `develop` and
`freeze` refuse to run, so the frozen models and spec cannot be overwritten.

Design history: the first design (60/20/20, CV-argmax tuning, `month` in the
primary set) is archived in reports/history/v1_60_20_20/. It was revised before
any test scoring; the reasons are recorded in the spec's `redesign_history`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev  # noqa: E402
from src import modeling as md  # noqa: E402
from src.audit import infer_period  # noqa: E402
from src.data import PROJECT_ROOT, RAW_CSV, load_raw, sha256_of  # noqa: E402
from src.features import (  # noqa: E402
    EXTENDED,
    FEATURE_SETS,
    MONTH_SENSITIVITY,
    PDAYS_SENSITIVITY,
    PRIMARY,
    feature_names,
    make_target,
)
from src.splits import TARGET_DEVELOPMENT_FRACTION, chronological_split, expanding_window_folds  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
MODELS = PROJECT_ROOT / "artifacts" / "models"
DRAFT_SPEC = REPORTS / "draft_spec.json"  # written by develop
FROZEN_SPEC = REPORTS / "frozen_spec.json"  # written only by the explicit freeze stage
TEST_METRICS = REPORTS / "metrics.json"

SENSITIVITY_SETS = (MONTH_SENSITIVITY, PDAYS_SENSITIVITY, EXTENDED)
SUBGROUP_COLUMNS = ("contact", "poutcome", "day_of_week", "period")

BUSINESS_METRICS = [
    "precision in top 10/20/30% of scored leads",
    "share of conversions captured in top 10/20/30%",
    "lift over the segment's own base rate",
    "contacts per conversion",
]
RANKING_AND_PROBABILITY_METRICS = [
    "PR-AUC (primary ranking metric), with bootstrap 95% interval",
    "ROC-AUC, with bootstrap 95% interval",
    "Brier score, uncalibrated and calibrated",
    "reliability table / calibration plot",
]


def _segments(raw: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    y = make_target(raw).to_numpy()
    return {name: (raw.iloc[index], y[index]) for name, index in chronological_split(raw["month"]).items()}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _evaluate(y, p) -> dict:
    return {**ev.probability_metrics(y, p), "top_k": ev.top_k_table(y, p).to_dict(orient="records")}


def _per_period_pr_auc(periods: np.ndarray, y, p) -> dict:
    return {
        str(level): float(average_precision_score(y[periods == level], p[periods == level]))
        for level in sorted(set(periods))
        if 0 < y[periods == level].sum() < (periods == level).sum()
    }


# --- develop ------------------------------------------------------------------


def develop() -> None:
    raw = load_raw()
    split = chronological_split(raw["month"])
    period = infer_period(raw)
    segments = _segments(raw)
    dev_raw, y_dev = segments["development"]
    val_raw, y_val = segments["validation"]
    val_periods = period.iloc[split["validation"]].to_numpy()
    folds = expanding_window_folds(len(dev_raw))

    # 1. Inner, order-preserving CV on the development segment only.
    cv = {
        md.NAIVE: md.cross_validate(dev_raw, y_dev, md.NAIVE, [{}], folds),
        md.LOGREG: md.cross_validate(dev_raw, y_dev, md.LOGREG, md.logreg_candidates(), folds),
        md.XGBOOST: md.cross_validate(dev_raw, y_dev, md.XGBOOST, md.xgb_candidates(), folds),
    }
    all_cv = pd.concat(cv.values(), ignore_index=True)
    all_cv.to_csv(REPORTS / "cv_folds.csv", index=False)
    md.summarise_cv(all_cv).to_csv(REPORTS / "cv_summary.csv", index=False)
    tuning = {
        md.LOGREG: md.select_candidate(cv[md.LOGREG], md.LOGREG_DEFAULT),
        md.XGBOOST: md.select_candidate(cv[md.XGBOOST], md.XGB_DEFAULT),
    }
    params = {family: result["chosen_params"] for family, result in tuning.items()}

    # 2. Fit on the full development segment; evaluate on out-of-time validation (uncalibrated).
    MODELS.mkdir(parents=True, exist_ok=True)
    validation, model_files, pipelines = {}, {}, {}
    for feature_set in (PRIMARY, *SENSITIVITY_SETS):
        families = (md.NAIVE, md.LOGREG, md.XGBOOST) if feature_set == PRIMARY else (md.LOGREG, md.XGBOOST)
        validation[feature_set] = {}
        for family in families:
            pipeline = md.fit_pipeline(dev_raw, y_dev, family, params.get(family), feature_set)
            scores = md.positive_scores(pipeline, val_raw)
            validation[feature_set][family] = {
                **_evaluate(y_val, scores),
                "pr_auc_by_period": _per_period_pr_auc(val_periods, y_val, scores),
            }
            key = f"{feature_set}__{family}"
            pipelines[key] = pipeline
            model_files[key] = f"artifacts/models/dev_{key}.joblib"
            joblib.dump(pipeline, PROJECT_ROOT / model_files[key])

    # 3. Primary model by the predeclared rule (primary feature set only; sensitivity sets never compete).
    primary_family, primary_checks = md.choose_primary(validation[PRIMARY])

    # 4. Platt calibration of the chosen primary model, fitted on validation.
    #    Rejected, never flipped, if the slope is not positive.
    calibration = {"method": "Platt: logistic regression on the model's log-odds", "fitted_on": "validation"}
    if primary_family is not None:
        calibrated = md.PlattCalibratedModel(pipelines[f"{PRIMARY}__{primary_family}"]).fit(val_raw, y_val)
        slope, intercept = calibrated.slope_intercept
        calibration.update({"slope": slope, "intercept": intercept, "used": slope > 0})
        uncalibrated = md.positive_scores(calibrated.pipeline, val_raw)
        if slope > 0:
            model_files["primary_calibrated"] = "artifacts/models/dev_primary_calibrated.joblib"
            joblib.dump(calibrated, PROJECT_ROOT / model_files["primary_calibrated"])
            pd.concat(
                [
                    ev.calibration_table(y_val, uncalibrated).assign(stage="uncalibrated (out-of-time)"),
                    ev.calibration_table(y_val, md.positive_scores(calibrated, val_raw)).assign(
                        stage="calibrated (IN-SAMPLE: same rows as the calibrator fit)"
                    ),
                ]
            ).to_csv(REPORTS / "calibration_validation.csv", index=False)
        else:
            calibration["reason"] = "non-positive slope: ranking reversed on validation; model rejected"
        groups = val_raw.assign(period=val_periods)
        pd.concat(
            [
                ev.subgroup_table(groups[c].rename(c), y_val, uncalibrated).rename(columns={c: "level"}).assign(column=c)
                for c in SUBGROUP_COLUMNS
            ]
        ).to_csv(REPORTS / "error_analysis_validation.csv", index=False)
    else:
        calibration.update({"used": False, "reason": "no eligible primary model"})

    drift = {
        "unseen_category_levels_in_validation": ev.unseen_levels(
            dev_raw, val_raw, feature_names(MONTH_SENSITIVITY, "categorical")
        )
    }

    spec = {
        "status": "draft",
        "data_sha256": sha256_of(RAW_CSV),
        "segments": {
            name: {
                "first_row": int(index[0]),
                "last_row": int(index[-1]),
                "n_rows": int(len(index)),
                "first_period": period.iloc[index[0]],
                "last_period": period.iloc[index[-1]],
            }
            for name, index in split.items()
        },
        "split_rule": (
            "Test = final 20% of rows, unchanged since Phase 1. Validation starts at the month boundary closest "
            f"to {TARGET_DEVELOPMENT_FRACTION:.0%} of all rows (chronology only, no target or model information)."
        ),
        "inner_folds": [{"fit_rows": [0, int(f[0][-1])], "eval_rows": [int(f[1][0]), int(f[1][-1])]} for f in folds],
        "feature_sets": {fs: feature_names(fs) for fs in FEATURE_SETS},
        "primary_feature_set": PRIMARY,
        "preprocessing": {
            "feature_construction": "src.features.build_features: stateless, whitelists inputs; duration and y never pass",
            "numeric": "median imputation + standardisation, fitted on training rows only",
            "categorical": "one-hot; 'unknown' kept as a level; unseen levels ignored at inference",
            "pdays_sensitivity_only": "days_since_prior_contact zero-imputed + standardised; indicator passed through",
        },
        "class_weighting": "not used (Phase 2 experiment: lowered PR-AUC and distorted probabilities)",
        "resampling": "none (no SMOTE)",
        "tuning": {
            "logreg_candidates": md.logreg_candidates(),
            "xgb_search_space": md.XGB_SEARCH_SPACE,
            "n_xgb_candidates_sampled": md.N_XGB_CANDIDATES,
            "defaults": {md.LOGREG: md.LOGREG_DEFAULT, md.XGBOOST: md.XGB_DEFAULT},
            "rule": (
                "Use the best mean inner-CV PR-AUC candidate only if its paired per-fold gain over the default is "
                f">= {md.MIN_TUNING_GAIN} and >= 2 standard errors; otherwise the predeclared default."
            ),
            "results": tuning,
            "naive_baseline_by_fold": cv[md.NAIVE][["fold", "n", "base_rate", "pr_auc"]].to_dict(orient="records"),
        },
        "params": params,
        "primary_rule": md.PRIMARY_RULE,
        "primary_checks": primary_checks,
        "primary_family": primary_family,
        "primary_model": None if primary_family is None else f"{PRIMARY}__{primary_family}",
        "calibration": calibration,
        "model_files": model_files,
        "metrics_to_report": {"ranking_and_probability": RANKING_AND_PROBABILITY_METRICS, "business": BUSINESS_METRICS},
        "priority_bands": "NOT DEFINED. High/Medium/Low bands belong to Phase 4 and are not part of this spec.",
        "sensitivity_sets_role": "Reported for context only; never used to choose features, models or thresholds.",
        "redesign_history": [
            "v1 (archived in reports/history/v1_60_20_20/): 60/20/20 split, month in the primary set, CV-argmax "
            "tuning, class-weighting experiment. Inner CV on 2008-only data could not distinguish candidates; the "
            "selected XGBoost (depth 4, min_child_weight 1) ranked validation leads in reverse (ROC-AUC 0.438).",
            "v1 observation, NOT blind selection: on the v1 validation period a shallow XGBoost (depth 2, "
            "min_child_weight 20) scored ROC-AUC 0.558. This motivated the conservative v2 search space and default; "
            "it was not used to pick a configuration.",
            "v1 diagnostic: logistic regression's validation advantage came largely from month levels absent from "
            "training. v2 drops month from the primary set on generalisation grounds.",
            "v2 (this spec): month-aligned ~70/10/20 split with the test segment unchanged, month as a sensitivity "
            "set, no class weighting, predeclared defaults when tuning is uninformative, predeclared primary rule.",
        ],
    }
    _write_json(DRAFT_SPEC, spec)
    _write_json(REPORTS / "validation_metrics.json", {"drift": drift, "models": validation})
    print(json.dumps(_jsonable({k: spec[k] for k in ("segments", "params", "primary_model", "calibration")}), indent=2))


# --- test (run once, after freezing) ----------------------------------------------

# Models scored on test. Sensitivity-set models stay validation-only.
TEST_MODELS = {
    "naive": f"{PRIMARY}__{md.NAIVE}",
    "logreg_primary": f"{PRIMARY}__{md.LOGREG}",
    "xgboost_comparison": f"{PRIMARY}__{md.XGBOOST}",
}
BOOTSTRAP_METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}
LIMITATIONS = [
    "May 2009 is split between validation (rows 30,430-32,948) and test (from row 32,949). Segments are row-wise "
    "chronological and non-overlapping but not fully month-disjoint.",
    "Models are fitted on May 2008 - March 2009 only; the test segment (May 2009 - November 2010) has a much higher "
    "conversion rate and far more prior-campaign history.",
]


def _score_frozen(raw_eval: pd.DataFrame, y_eval: np.ndarray, models: dict) -> tuple[dict, dict]:
    """Frozen metrics for naive, primary (before/after calibration) and comparison models on one segment."""
    scores = {name: md.positive_scores(model, raw_eval) for name, model in models.items()}
    out = {"base_rate": float(y_eval.mean()), "n": int(len(y_eval)), "models": {}}
    for name, s in scores.items():
        if name != "primary_calibrated":
            out["models"][name] = _evaluate(y_eval, s)
    if "primary_calibrated" in scores:
        before, after = scores["logreg_primary"], scores["primary_calibrated"]
        out["primary_calibration"] = {
            "brier_before": float(ev.probability_metrics(y_eval, before)["brier"]),
            "brier_after": float(ev.probability_metrics(y_eval, after)["brier"]),
            "brier_at_segment_base_rate": float(y_eval.mean() * (1 - y_eval.mean())),
            "mean_predicted_before": float(before.mean()),
            "mean_predicted_after": float(after.mean()),
            "actual_rate": float(y_eval.mean()),
            "ranking_unchanged_by_calibration": bool(
                np.isclose(average_precision_score(y_eval, after), average_precision_score(y_eval, before))
            ),
        }
    out["bootstrap"] = ev.paired_bootstrap(
        y_eval,
        {k: scores[k] for k in ("logreg_primary", "xgboost_comparison")},
        BOOTSTRAP_METRICS,
        differences=(("logreg_primary", "xgboost_comparison"),),
    )
    return out, scores


def test() -> None:
    if TEST_METRICS.exists():
        raise SystemExit(
            f"{TEST_METRICS} exists: the test segment was scored once with the frozen spec and cannot be re-scored "
            "as part of model development."
        )
    if not FROZEN_SPEC.exists():
        raise SystemExit("No frozen spec: review the draft, then run the freeze stage first.")
    spec = json.loads(FROZEN_SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "frozen" or spec["data_sha256"] != sha256_of(RAW_CSV):
        raise SystemExit("Spec is not frozen, or the raw data changed since it was frozen")

    raw = load_raw()
    segments = _segments(raw)
    dev_raw, _ = segments["development"]
    test_raw, y_test = segments["test"]

    models = {name: joblib.load(PROJECT_ROOT / spec["model_files"][key]) for name, key in TEST_MODELS.items()}
    if spec["calibration"]["used"]:
        models["primary_calibrated"] = joblib.load(PROJECT_ROOT / spec["model_files"]["primary_calibrated"])
    result, scores = _score_frozen(test_raw, y_test, models)

    if "primary_calibrated" in scores:
        pd.concat(
            [
                ev.calibration_table(y_test, scores["logreg_primary"]).assign(stage="before calibration"),
                ev.calibration_table(y_test, scores["primary_calibrated"]).assign(
                    stage="after Platt calibration (fitted on validation)"
                ),
            ]
        ).to_csv(REPORTS / "calibration_test.csv", index=False)
    pd.DataFrame({"y": y_test, **scores}, index=test_raw.index).to_csv(REPORTS / "test_scores.csv", index_label="row")

    _write_json(
        TEST_METRICS,
        {
            "note": "ONE-TIME scoring of the untouched final 20% with the frozen spec. Not to be regenerated.",
            "frozen_spec_sha256": sha256_of(FROZEN_SPEC),
            "primary_model": spec["primary_model"],
            "comparison_model": f"{PRIMARY}__{md.XGBOOST}",
            "test_segment": spec["segments"]["test"],
            **result,
            "drift": {
                "unseen_category_levels_in_test": ev.unseen_levels(
                    dev_raw, test_raw, feature_names(MONTH_SENSITIVITY, "categorical")
                )
            },
            "sensitivity_sets": "validation-only; not scored on test",
            "limitations": LIMITATIONS,
        },
    )
    print(f"Test segment scored once; results in {TEST_METRICS}")


def random_split_diagnostic() -> None:
    """OPTIMISTIC DIAGNOSTIC ONLY: frozen recipe on a stratified random 70/10/20 split. Never used for any decision."""
    if not TEST_METRICS.exists():
        raise SystemExit("Run only after the test segment has been scored with the frozen spec.")
    spec = json.loads(FROZEN_SPEC.read_text(encoding="utf-8"))
    raw = load_raw()
    y = make_target(raw).to_numpy()
    fit_index, rest = train_test_split(np.arange(len(raw)), train_size=0.7, stratify=y, random_state=md.SEED)
    cal_index, eval_index = train_test_split(rest, test_size=2 / 3, stratify=y[rest], random_state=md.SEED)

    models = {"naive": md.fit_pipeline(raw.iloc[fit_index], y[fit_index], md.NAIVE)}
    for name, family in (("logreg_primary", md.LOGREG), ("xgboost_comparison", md.XGBOOST)):
        models[name] = md.fit_pipeline(raw.iloc[fit_index], y[fit_index], family, spec["params"][family], PRIMARY)
    models["primary_calibrated"] = md.PlattCalibratedModel(models["logreg_primary"]).fit(
        raw.iloc[cal_index], y[cal_index]
    )
    result, _ = _score_frozen(raw.iloc[eval_index], y[eval_index], models)
    _write_json(
        REPORTS / "random_split_diagnostic.json",
        {
            "note": "OPTIMISTIC DIAGNOSTIC ONLY. Frozen recipe (features, hyperparameters, Platt calibration) on a "
            "stratified random 70/10/20 split of all rows, so training, calibration and evaluation rows share "
            "periods. Shows how optimistic random evaluation is; not a headline result and not used to revise or "
            "select the production model.",
            "split_sizes": {"fit": len(fit_index), "calibrate": len(cal_index), "evaluate": len(eval_index)},
            "seed": md.SEED,
            **result,
        },
    )
    print("Random-split diagnostic written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["develop", "freeze", "test", "random-split-diagnostic"])
    args = parser.parse_args()
    REPORTS.mkdir(exist_ok=True)
    if args.stage in ("develop", "freeze") and TEST_METRICS.exists():
        raise SystemExit("The test segment has been scored; development and freezing are closed.")
    if args.stage == "develop":
        develop()
    elif args.stage == "freeze":
        spec = json.loads(DRAFT_SPEC.read_text(encoding="utf-8"))
        spec["status"] = "frozen"
        _write_json(FROZEN_SPEC, spec)
        print(f"Frozen: {FROZEN_SPEC}")
    elif args.stage == "test":
        test()
    else:
        random_split_diagnostic()


if __name__ == "__main__":
    main()
