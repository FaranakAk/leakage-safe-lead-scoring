import io
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src import app_logic as app
from src import features as ft
from src import inference as inf
from src import modeling as md
from src.data import PROJECT_ROOT

FULL_ARTIFACT = PROJECT_ROOT / "artifacts" / "models" / f"{inf.ARTIFACT_VERSION}.joblib"
APP = PROJECT_ROOT / "app.py"
BASE_LEAD = {
    "age": 35, "job": "management", "marital_status": "married", "education": "university.degree",
    "credit_default": "no", "housing_loan": "yes", "personal_loan": "no", "contact_channel": "cellular",
    "call_weekday": "thu", "earlier_calls_this_campaign": 0, "earlier_campaign_contacts": 0,
    "previous_campaign_outcome": "nonexistent",
}


@pytest.fixture(scope="module")
def scorer():
    return app.load_scorer()


@pytest.fixture(scope="module")
def example():
    return pd.read_csv(app.EXAMPLE_CSV)


@pytest.fixture
def no_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fit called at inference time")

    for cls in (Pipeline, ColumnTransformer, BaseEstimator):
        for name in ("fit", "fit_transform"):
            monkeypatch.setattr(cls, name, forbidden, raising=False)
    monkeypatch.setattr(md.PlattCalibratedModel, "fit", forbidden)


# --- artifact and startup -------------------------------------------------------------------------


def test_deployment_artifact_loads_with_metadata(scorer):
    assert scorer.metadata["version"] == inf.DEPLOY_VERSION
    assert scorer.metadata["derived_from"] == inf.ARTIFACT_VERSION
    assert scorer.feature_names == ft.feature_names(ft.PRIMARY)
    assert "calibration_warning" in scorer.metadata
    assert not hasattr(scorer, "shap_background")  # no raw development rows shipped
    assert app.DEPLOY_ARTIFACT.stat().st_size < 200_000


def test_app_inference_starts_without_dataset_shap_or_xgboost(tmp_path):
    """Load + score in a fresh interpreter where xgboost, shap and the dataset loader are unavailable."""
    code = textwrap.dedent(
        f"""
        import sys
        sys.modules["xgboost"] = None          # any import of xgboost would now fail
        sys.modules["shap"] = None             # any import of shap would now fail
        sys.path.insert(0, r"{PROJECT_ROOT}")
        from src import app_logic as app
        scorer = app.load_scorer()
        result = app.score_lead(scorer, {BASE_LEAD!r})
        table = app.score_batch_table(scorer, app.split_valid(scorer, app.normalise_upload(
            __import__("pandas").read_csv(app.EXAMPLE_CSV)))[0])
        assert len(table) == 200
        assert "src.data" not in sys.modules, "the dataset module was imported"
        print("ok", result["historical_reference_priority"])
        """
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("ok")


def test_scoring_and_explaining_never_fit(scorer, example, no_fitting):
    coef = scorer.pipeline.named_steps["model"].coef_.copy()
    app.score_lead(scorer, BASE_LEAD)
    valid, _ = app.split_valid(scorer, app.normalise_upload(example))
    app.score_batch_table(scorer, valid)
    np.testing.assert_array_equal(scorer.pipeline.named_steps["model"].coef_, coef)


# --- inputs -----------------------------------------------------------------------------------------


def test_inputs_never_include_forbidden_or_hidden_fields(scorer):
    hidden = {"duration", "y", "pdays", "month", "campaign", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
              "euribor3m", "nr.employed"}
    assert not hidden & set(app.INPUT_COLUMNS)
    assert set(app.to_raw(pd.DataFrame([BASE_LEAD])).columns) == set(ft.required_raw_columns(ft.PRIMARY))
    assert set(scorer.input_schema["never_inputs"]) >= {"duration", "y", "pdays", "month"}


def test_categories_come_from_the_frozen_schema(scorer):
    encoder = scorer.pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    frozen = dict(zip(scorer.pipeline.named_steps["preprocess"].transformers_[1][2], encoder.categories_))
    for field, raw in app.CATEGORICAL_FIELDS.items():
        assert app.categories(scorer, field) == list(frozen[raw])


def test_every_allowed_category_and_numeric_boundary_scores(scorer):
    for field in app.CATEGORICAL_FIELDS:
        if field == "previous_campaign_outcome":
            continue
        for code in app.categories(scorer, field):
            assert app.score_lead(scorer, {**BASE_LEAD, field: code})["historical_reference_priority"] in inf.BANDS
    for outcome in app.outcome_options(scorer, 1):
        app.score_lead(scorer, {**BASE_LEAD, "earlier_campaign_contacts": 1, "previous_campaign_outcome": outcome})
    for field in app.NUMERIC_FIELDS:
        for bound in scorer.input_schema["numeric_limits"][field]:
            lead = {**BASE_LEAD, field: bound}
            if field == "earlier_campaign_contacts" and bound > 0:
                lead["previous_campaign_outcome"] = "failure"
            app.score_lead(scorer, lead)


def test_outcome_options_follow_earlier_contacts(scorer):
    assert app.outcome_options(scorer, 0) == ["nonexistent"]
    assert set(app.outcome_options(scorer, 3)) == {"failure", "success"}


@pytest.mark.parametrize(
    "change, message",
    [
        ({"earlier_campaign_contacts": 0, "previous_campaign_outcome": "failure"}, "must be 'nonexistent'"),
        ({"earlier_campaign_contacts": 2, "previous_campaign_outcome": "nonexistent"}, "cannot be 'nonexistent'"),
        ({"job": "astronaut"}, "not a known value"),
        ({"age": 7}, "between"),
        ({"earlier_calls_this_campaign": -1}, "between"),
        ({"earlier_calls_this_campaign": 1.5}, "whole number"),
        ({"age": "old"}, "not a number"),
    ],
)
def test_impossible_inputs_are_rejected(scorer, change, message):
    lead = {**BASE_LEAD, **change}
    assert any(message in p for p in app.validate_lead(scorer, lead))
    with pytest.raises(ValueError):
        app.score_lead(scorer, lead)


def test_missing_field_is_rejected(scorer):
    lead = dict(BASE_LEAD)
    del lead["contact_channel"]
    assert app.validate_lead(scorer, lead)


def test_out_of_training_range_is_flagged_not_rejected(scorer):
    lead = {**BASE_LEAD, "earlier_campaign_contacts": 5, "previous_campaign_outcome": "success"}
    assert app.score_lead(scorer, lead)["outside_training_range"]
    assert not app.score_lead(scorer, BASE_LEAD)["outside_training_range"]


# --- outputs ----------------------------------------------------------------------------------------


def test_single_lead_matches_inference_module(scorer):
    lead = {**BASE_LEAD, "earlier_campaign_contacts": 1, "previous_campaign_outcome": "failure"}
    result = app.score_lead(scorer, lead)
    direct = scorer.score(app.to_raw(pd.DataFrame([lead]))).iloc[0]
    assert result["historical_reference_priority"] == direct["priority"]
    assert result["historical_percentile"] == direct["historical_percentile"]
    assert result["historical_calibrated_estimate"] == direct["calibrated_probability_secondary"]
    contributions = result["explanation"]["contribution"]
    assert contributions.sum() + scorer.explanation_base_value == pytest.approx(result["ranking_score"])


def test_top_factors_are_plain_language(scorer):
    result = app.score_lead(scorer, {**BASE_LEAD, "earlier_campaign_contacts": 2, "previous_campaign_outcome": "failure"})
    higher, lower = app.top_factors(result["explanation"])
    assert higher and lower and len(higher) <= 3 and len(lower) <= 3
    assert any("Contact channel: Mobile phone" == item for item in higher)
    assert all("_" not in item for item in higher + lower)


def test_batch_output_columns_and_exact_20_30_50(scorer, example):
    valid, invalid = app.split_valid(scorer, app.normalise_upload(example))
    assert invalid.empty
    table = app.score_batch_table(scorer, valid)
    assert list(table.columns) == list(app.SCORED_COLUMNS)
    assert {"batch_priority", "batch_percentile", "historical_reference_priority", "historical_percentile",
            "ranking_score", "historical_calibrated_estimate"} <= set(table.columns)
    # The example file has no tied scores at the band boundaries, so the split is exact.
    assert table["batch_priority"].value_counts().to_dict() == {"Low": 100, "Medium": 60, "High": 40}
    assert table["batch_rank"].tolist() == sorted(table["batch_rank"])


@pytest.mark.parametrize("n", [10, 13, 37, 50, 99, 150])
def test_batch_bands_are_within_one_lead_of_targets(scorer, example, n):
    valid, _ = app.split_valid(scorer, app.normalise_upload(example.head(n)))
    counts = app.score_batch_table(scorer, valid)["batch_priority"].value_counts()
    assert abs(counts.get("High", 0) - 0.2 * n) <= 1
    assert abs(counts.get("Medium", 0) - 0.3 * n) <= 1
    if n % 10 == 0:
        assert (counts.get("High", 0), counts.get("Medium", 0)) == (n // 5, 3 * n // 10)


def test_tied_batch_scores_share_the_higher_band(scorer):
    leads = pd.DataFrame([BASE_LEAD] * 5)
    valid, _ = app.split_valid(scorer, leads)
    assert set(app.score_batch_table(scorer, valid)["batch_priority"]) == {"High"}


def test_upload_drops_target_and_forbidden_columns(scorer, example):
    upload = example.head(20).assign(duration=300, y="yes", month="may", euribor3m=4.8, pdays=999, note="x")
    out = app.score_batch_table(scorer, app.split_valid(scorer, app.normalise_upload(upload))[0])
    assert not {"duration", "y", "month", "euribor3m", "pdays", "note"} & set(out.columns)
    baseline = app.score_batch_table(scorer, app.split_valid(scorer, app.normalise_upload(example.head(20)))[0])
    pd.testing.assert_frame_equal(out, baseline)


def test_upload_accepts_raw_uci_format(scorer, raw_full):
    sample = raw_full.iloc[27972:28022]
    buffer = io.StringIO()
    sample.to_csv(buffer, sep=";", index=False)
    parsed = pd.read_csv(io.StringIO(buffer.getvalue()), sep=None, engine="python")
    leads = app.normalise_upload(parsed)
    valid, invalid = app.split_valid(scorer, leads)
    assert invalid.empty and len(valid) == 50
    table = app.score_batch_table(scorer, valid)
    assert "duration" not in table.columns and "y" not in table.columns
    direct = scorer.score(sample)
    assert sorted(table["historical_percentile"]) == sorted(direct["historical_percentile"].round(1))


def test_upload_with_wrong_columns_gives_clear_error():
    with pytest.raises(ValueError, match="Missing columns"):
        app.normalise_upload(pd.DataFrame({"foo": [1]}))


def test_invalid_rows_are_reported_not_scored(scorer, example):
    upload = example.head(5).copy()
    upload.loc[0, "previous_campaign_outcome"] = "success"
    upload.loc[0, "earlier_campaign_contacts"] = 0
    upload.loc[1, "job"] = "pilot"
    valid, invalid = app.split_valid(scorer, app.normalise_upload(upload))
    assert len(valid) == 3 and len(invalid) == 2
    assert invalid["problem"].str.len().gt(0).all()


# --- deployment artifact == Phase 4 artifact --------------------------------------------------------


def test_deployment_artifact_reproduces_phase4_artifact(scorer, raw_full):
    if not FULL_ARTIFACT.exists():
        pytest.skip("Full Phase 4 artifact is not present (it is not committed)")
    from src import explainability as xp

    full = inf.load_scorer(FULL_ARTIFACT)
    examples = pd.concat([raw_full.iloc[27972:28472], raw_full.iloc[30000:30100]])
    pd.testing.assert_frame_equal(scorer.score(examples), full.score(examples))
    pd.testing.assert_frame_equal(scorer.score_batch(examples), full.score_batch(examples))
    np.testing.assert_allclose(
        scorer.contributions(examples), xp.field_contributions(full.pipeline, full.explainer(), examples), atol=1e-10
    )
    one = examples.iloc[[3]]
    a, b = scorer.explain(one), full.explain(one)
    assert a["feature"].tolist() == b["feature"].tolist()
    np.testing.assert_allclose(a["contribution"], b["contribution"], atol=1e-10)


# --- Streamlit smoke test ---------------------------------------------------------------------------


def test_streamlit_app_smoke(no_fitting):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Lead Priority Scorer"
    assert [m.label for m in at.metric] == ["Historical percentile"]
    outcome = [s for s in at.selectbox if s.label == "Outcome of the previous campaign"][0]
    assert outcome.disabled and outcome.value == "nonexistent"

    at.number_input(key="earlier_campaign_contacts").set_value(2).run()
    outcome = [s for s in at.selectbox if s.label == "Outcome of the previous campaign"][0]
    assert not outcome.disabled and outcome.value in {"failure", "success"}
    assert not at.exception

    at.toggle(key="use_example").set_value(True).run()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Leads scored"] == "200"
    assert (metrics["Batch priority: High"], metrics["Batch priority: Medium"], metrics["Batch priority: Low"]) == ("40", "60", "100")
    assert not at.exception
    page_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "chance of converting" not in page_text.lower()
