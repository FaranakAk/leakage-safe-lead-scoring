import numpy as np
import pytest

from src import explainability as xp
from src import features as ft
from src import modeling as md


@pytest.fixture
def fitted(raw_train):
    pipeline = md.fit_pipeline(raw_train, ft.make_target(raw_train), md.LOGREG)
    explainer = xp.make_explainer(pipeline, raw_train.head(30))
    return pipeline, explainer


def test_every_model_feature_has_a_plain_language_label():
    assert set(ft.feature_names(ft.EXTENDED)) | set(ft.feature_names(ft.PDAYS_SENSITIVITY)) | {"month"} <= set(ft.FEATURE_LABELS)


def test_transformed_columns_map_back_to_business_fields(fitted):
    pipeline, _ = fitted
    mapping = xp.transformed_to_field(pipeline)
    assert set(mapping.values()) == set(ft.feature_names(ft.PRIMARY))
    assert mapping["job_admin."] == "job"


def test_field_contributions_are_additive_to_ranking_score(fitted, raw_train):
    pipeline, explainer = fitted
    contributions = xp.field_contributions(pipeline, explainer, raw_train)
    assert list(contributions.columns) == ft.feature_names(ft.PRIMARY)
    np.testing.assert_allclose(
        contributions.sum(axis=1) + xp.base_value(explainer), pipeline.decision_function(raw_train), atol=1e-8
    )


def test_linear_shap_matches_closed_form(fitted, raw_train):
    """For a linear model with independent features, SHAP = coef * (x - background mean)."""
    pipeline, explainer = fitted
    background = xp._transform(pipeline, raw_train.head(30))
    x = xp._transform(pipeline, raw_train.head(5))
    expected = pipeline.named_steps["model"].coef_[0] * (x - background.mean(axis=0))
    np.testing.assert_allclose(np.asarray(explainer.shap_values(x)), expected, atol=1e-8)


def test_explanations_do_not_alter_predictions(fitted, raw_train):
    pipeline, explainer = fitted
    before = pipeline.decision_function(raw_train)
    xp.field_contributions(pipeline, explainer, raw_train)
    xp.local_explanation(pipeline, explainer, raw_train.head(1))
    np.testing.assert_array_equal(pipeline.decision_function(raw_train), before)


def test_local_explanation_language_is_associational(fitted, raw_train):
    pipeline, explainer = fitted
    local = xp.local_explanation(pipeline, explainer, raw_train.head(1))
    assert len(local) == len(ft.feature_names(ft.PRIMARY))
    assert local["contribution"].abs().is_monotonic_decreasing
    for sentence in local["sentence"]:
        assert "pushed the model score" in sentence
        assert "cause" not in sentence.lower()
    with pytest.raises(ValueError):
        xp.local_explanation(pipeline, explainer, raw_train.head(2))


def test_global_importance_uses_labels(fitted, raw_train):
    pipeline, explainer = fitted
    table = xp.global_importance(xp.field_contributions(pipeline, explainer, raw_train))
    assert table["mean_abs_contribution"].is_monotonic_decreasing
    assert set(table["field"]) <= set(ft.FEATURE_LABELS.values())
