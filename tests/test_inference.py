import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src import features as ft
from src import inference as inf
from src import modeling as md
from src.data import PROJECT_ROOT

ARTIFACT = PROJECT_ROOT / "artifacts" / "models" / f"{inf.ARTIFACT_VERSION}.joblib"


@pytest.fixture
def scorer(raw_train) -> inf.LeadScorer:
    """Small scorer built from synthetic data with the same code path as the real artifact."""
    y = ft.make_target(raw_train)
    pipeline = md.fit_pipeline(raw_train, y, md.LOGREG)
    calibrated = md.PlattCalibratedModel(pipeline).fit(raw_train, y)
    return inf.build_scorer(calibrated, reference_raw=raw_train, background_raw=raw_train.head(20), metadata={})


@pytest.fixture(scope="module")
def real_scorer():
    if not ARTIFACT.exists():
        pytest.skip("Artifact not built; run scripts/build_phase4.py freeze-bands")
    return inf.load_scorer(ARTIFACT)


# --- band rule ----------------------------------------------------------------


def test_band_rule_is_frozen_20_30_50():
    assert inf.BAND_RULE == {"High": 0.20, "Medium": 0.30, "Low": 0.50}


def test_band_cutoffs_split_reference_20_30_50():
    reference = np.arange(1000, dtype=float)
    cutoffs = inf.band_cutoffs(reference)
    bands = inf.assign_band(reference, cutoffs)
    assert (bands == "High").mean() == pytest.approx(0.20, abs=0.002)
    assert (bands == "Medium").mean() == pytest.approx(0.30, abs=0.002)
    assert (bands == "Low").mean() == pytest.approx(0.50, abs=0.002)


def test_band_boundaries_and_ties_go_to_higher_band():
    cutoffs = {"high_min": 1.0, "medium_min": 0.0}
    scores = np.array([1.0, 0.9999, 0.0, -0.0001, 5.0, -5.0])
    assert inf.assign_band(scores, cutoffs).tolist() == ["High", "Medium", "Medium", "Low", "High", "Low"]


def test_percentile_of_reference():
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    assert inf.percentile_of([0.5, 1.0, 2.5, 4.0, 9.0], reference).tolist() == [0.0, 25.0, 50.0, 100.0, 100.0]


# --- scorer behaviour -----------------------------------------------------------


def test_scoring_never_refits(scorer, raw_train, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fit called during inference")

    for cls in (Pipeline, ColumnTransformer, BaseEstimator):
        for name in ("fit", "fit_transform"):
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, forbidden, raising=False)
    monkeypatch.setattr(md.PlattCalibratedModel, "fit", forbidden)
    coef_before = scorer.pipeline.named_steps["model"].coef_.copy()
    scorer.score(raw_train)
    scorer.score_batch(raw_train)
    scorer.explain(raw_train.head(1))
    np.testing.assert_array_equal(scorer.pipeline.named_steps["model"].coef_, coef_before)


def test_identical_input_gives_identical_outputs(scorer, raw_train):
    first = scorer.score(raw_train.head(5))
    second = scorer.score(raw_train.head(5).copy())
    pd.testing.assert_frame_equal(first, second)
    duplicated = pd.concat([raw_train.head(1)] * 3, ignore_index=True)
    out = scorer.score_batch(duplicated)
    assert out["ranking_score"].nunique() == 1
    assert out["priority"].nunique() == 1 and out["batch_priority"].nunique() == 1
    assert out["historical_percentile"].nunique() == 1


def test_calibrated_probability_is_monotonic_in_ranking_score(scorer, raw_train):
    out = scorer.score(raw_train).sort_values("ranking_score")
    assert out["calibrated_probability_secondary"].is_monotonic_increasing


def test_outputs_ignore_forbidden_columns(scorer, raw_train):
    base = scorer.score(raw_train.head(10))
    perturbed = scorer.score(raw_train.head(10).assign(duration=99999, y="yes"))
    pd.testing.assert_frame_equal(base, perturbed)
    assert not (set(scorer.feature_names) & (ft.FORBIDDEN_COLUMNS | ft.DERIVE_ONLY_COLUMNS))


def test_batch_mode_ranks_within_batch(scorer, raw_train):
    out = scorer.score_batch(raw_train)
    shares = out["batch_priority"].value_counts(normalize=True)
    assert shares.get("High", 0) <= 0.2 + 0.2  # ties on this tiny synthetic set can enlarge bands
    assert set(out["batch_priority"]) <= set(inf.BANDS)
    assert out["batch_percentile"].between(0, 100).all()


def test_build_scorer_rejects_leaky_feature_lists():
    with pytest.raises(ValueError):
        ft.assert_no_leakage(["age", "duration"])


# --- the real artifact ----------------------------------------------------------


def test_real_artifact_loads_with_metadata(real_scorer):
    meta = real_scorer.metadata
    assert meta["version"] == inf.ARTIFACT_VERSION
    assert meta["band_rule"] == inf.BAND_RULE
    assert meta["primary_outputs"][0] == "priority"
    assert "drift" in meta["calibration_warning"]
    assert meta["periods"]["development"]["first_period"] == "2008-05"
    assert real_scorer.feature_names == ft.feature_names(ft.PRIMARY)
    assert "duration" not in real_scorer.shap_background.columns and "y" not in real_scorer.shap_background.columns
    assert real_scorer.cutoffs["high_min"] > real_scorer.cutoffs["medium_min"]


def test_real_artifact_reference_split_is_20_30_50(real_scorer):
    bands = inf.assign_band(real_scorer.reference_scores, real_scorer.cutoffs)
    assert (bands == "High").mean() == pytest.approx(0.2, abs=0.002)
    assert (bands == "Medium").mean() == pytest.approx(0.3, abs=0.002)


def test_real_artifact_matches_frozen_phase3_model(real_scorer, raw_full):
    import joblib

    frozen = joblib.load(PROJECT_ROOT / "artifacts" / "models" / "dev_primary_calibrated.joblib")
    rows = raw_full.iloc[32949:33049]
    np.testing.assert_allclose(real_scorer.ranking_score(rows), frozen.pipeline.decision_function(rows))
    np.testing.assert_allclose(
        real_scorer.score(rows)["calibrated_probability_secondary"], frozen.predict_proba(rows)[:, 1]
    )
