import numpy as np
import pandas as pd
import pytest

from src import features as ft
from src import modeling as md


@pytest.mark.parametrize("family", [md.NAIVE, md.LOGREG, md.XGBOOST])
@pytest.mark.parametrize("feature_set", ft.FEATURE_SETS)
def test_estimator_sees_only_whitelisted_features(raw_train, family, feature_set):
    model = md.fit_pipeline(raw_train, ft.make_target(raw_train), family, feature_set=feature_set)
    preprocess = model.named_steps["preprocess"]
    seen = set(preprocess.feature_names_in_)
    assert seen == set(ft.feature_names(feature_set))
    assert not seen & (ft.FORBIDDEN_COLUMNS | ft.DERIVE_ONLY_COLUMNS)
    output_names = set(preprocess.get_feature_names_out())
    assert not any(name.startswith(("duration", "y_", "campaign", "pdays")) for name in output_names)


@pytest.mark.parametrize("family", [md.LOGREG, md.XGBOOST])
def test_predictions_do_not_depend_on_duration_or_target(raw_train, family):
    model = md.fit_pipeline(raw_train, ft.make_target(raw_train), family)
    baseline = md.positive_scores(model, raw_train)
    perturbed = raw_train.assign(duration=raw_train["duration"] * 7 + 5, y="yes")
    np.testing.assert_array_equal(md.positive_scores(model, perturbed), baseline)
    np.testing.assert_array_equal(md.positive_scores(model, raw_train.drop(columns=["duration", "y"])), baseline)


def test_preprocessing_is_fitted_on_training_rows_only(raw_train):
    fit_rows = raw_train.iloc[:30]
    model = md.fit_pipeline(fit_rows, ft.make_target(fit_rows), md.LOGREG)
    numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
    built = ft.build_features(fit_rows)
    np.testing.assert_allclose(numeric.named_steps["scale"].mean_, built[["age", "prior_contacts_current_campaign", "previous"]].mean())
    encoder = model.named_steps["preprocess"].named_transformers_["categorical"]
    job_levels = list(encoder.categories_[ft.feature_names(ft.PRIMARY, "categorical").index("job")])
    assert job_levels == sorted(fit_rows["job"].unique())


def test_unseen_category_at_inference_is_ignored_not_an_error(raw_train):
    model = md.fit_pipeline(raw_train, ft.make_target(raw_train), md.LOGREG)
    unseen = raw_train.head(2).assign(month="sep", job="never-seen")
    assert md.positive_scores(model, unseen).shape == (2,)


def test_days_since_prior_contact_is_zero_imputed_with_indicator(raw_sample):
    model = md.make_pipeline(md.make_estimator(md.NAIVE), ft.PDAYS_SENSITIVITY)
    model.fit(raw_sample, ft.make_target(raw_sample))
    zero_imputed = model.named_steps["preprocess"].named_transformers_["zero_imputed"]
    assert zero_imputed.named_steps["impute"].statistics_.tolist() == [0.0]


def test_naive_baseline_predicts_training_rate(raw_train):
    model = md.fit_pipeline(raw_train, ft.make_target(raw_train), md.NAIVE)
    scores = md.positive_scores(model, raw_train.head(5))
    np.testing.assert_allclose(scores, ft.make_target(raw_train).mean())


def test_no_class_weighting_is_applied():
    assert md.make_estimator(md.LOGREG).get_params()["class_weight"] is None
    assert md.make_estimator(md.XGBOOST).get_params().get("scale_pos_weight") in (None, 1, 1.0)


def test_xgb_candidates_are_reproducible_inside_search_space_and_include_default():
    first, second = md.xgb_candidates(), md.xgb_candidates()
    assert first == second
    assert md.N_XGB_CANDIDATES <= len(first) <= md.N_XGB_CANDIDATES + 1
    assert dict(sorted(md.XGB_DEFAULT.items())) in first
    for candidate in first:
        for key, value in candidate.items():
            assert value in md.XGB_SEARCH_SPACE[key]
    assert max(md.XGB_SEARCH_SPACE["max_depth"]) <= 3  # conservative complexity


def test_logreg_candidates_include_default():
    assert md.LOGREG_DEFAULT in md.logreg_candidates()


def _cv_rows(per_candidate: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for candidate_id, (params, scores) in enumerate(per_candidate.items()):
        for fold, score in enumerate(scores):
            rows.append(
                {"family": "logreg", "feature_set": "primary", "candidate_id": candidate_id, "params": params,
                 "fold": fold, "pr_auc": score, "roc_auc": 0.5, "brier": 0.1, "mean_predicted": 0.1, "base_rate": 0.1}
            )
    return pd.DataFrame(rows)


def test_select_candidate_keeps_default_when_gain_is_noise():
    cv = _cv_rows({repr({"C": 1.0}): [0.060, 0.070, 0.050, 0.065], repr({"C": 0.1}): [0.068, 0.062, 0.058, 0.063]})
    result = md.select_candidate(cv, {"C": 1.0})
    assert result["best_params"] == {"C": 0.1}
    assert not result["informative"]
    assert result["chosen_params"] == {"C": 1.0}


def test_select_candidate_switches_when_gain_is_consistent_and_large():
    cv = _cv_rows({repr({"C": 1.0}): [0.060, 0.070, 0.050, 0.065], repr({"C": 0.1}): [0.080, 0.091, 0.069, 0.086]})
    result = md.select_candidate(cv, {"C": 1.0})
    assert result["informative"]
    assert result["chosen_params"] == {"C": 0.1}


def _validation(lr_pr, xgb_pr, lr_lift, xgb_lift, lr_months, xgb_months, xgb_roc=0.6):
    def entry(pr, roc, lift, months):
        return {"pr_auc": pr, "roc_auc": roc, "top_k": [{"top_fraction": 0.2, "lift": lift}], "pr_auc_by_period": months}

    return {
        md.NAIVE: entry(0.1, 0.5, 1.0, {}),
        md.LOGREG: entry(lr_pr, 0.6, lr_lift, lr_months),
        md.XGBOOST: entry(xgb_pr, xgb_roc, xgb_lift, xgb_months),
    }


def test_choose_primary_prefers_logreg_unless_xgboost_wins_every_check():
    months_lr, months_xgb_better, months_xgb_mixed = {"a": 0.2, "b": 0.1}, {"a": 0.25, "b": 0.12}, {"a": 0.3, "b": 0.09}
    assert md.choose_primary(_validation(0.20, 0.25, 1.6, 1.7, months_lr, months_xgb_better))[0] == md.XGBOOST
    assert md.choose_primary(_validation(0.20, 0.205, 1.6, 1.7, months_lr, months_xgb_better))[0] == md.LOGREG  # gain < 0.01
    assert md.choose_primary(_validation(0.20, 0.25, 1.6, 1.5, months_lr, months_xgb_better))[0] == md.LOGREG  # lift
    assert md.choose_primary(_validation(0.20, 0.25, 1.6, 1.7, months_lr, months_xgb_mixed))[0] == md.LOGREG  # stability


def test_choose_primary_rejects_reversed_ranking():
    validation = _validation(0.20, 0.30, 1.6, 2.0, {"a": 0.2}, {"a": 0.3}, xgb_roc=0.45)
    primary, checks = md.choose_primary(validation)
    assert primary == md.LOGREG
    assert not checks["eligible"][md.XGBOOST]


def test_cross_validate_reports_every_candidate_and_fold(raw_train):
    folds = [(np.arange(0, 40), np.arange(40, 60))]
    cv = md.cross_validate(raw_train, ft.make_target(raw_train), md.LOGREG, [{"C": 0.1}, {"C": 1.0}], folds)
    assert len(cv) == 2
    assert {"pr_auc", "roc_auc", "brier"} <= set(cv.columns)
    assert len(md.summarise_cv(cv)) == 2


def test_platt_calibration_preserves_ranking_and_shifts_scale(raw_train):
    y = ft.make_target(raw_train)
    pipeline = md.fit_pipeline(raw_train, y, md.LOGREG)
    calibrated = md.PlattCalibratedModel(pipeline).fit(raw_train, y)
    before = md.positive_scores(pipeline, raw_train)
    after = md.positive_scores(calibrated, raw_train)
    slope, _ = calibrated.slope_intercept
    if slope > 0:
        assert np.array_equal(np.argsort(before, kind="stable"), np.argsort(after, kind="stable"))
    assert after.mean() == pytest.approx(y.mean(), abs=0.02)
