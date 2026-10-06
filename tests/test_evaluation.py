import numpy as np
import pandas as pd
import pytest

from src import evaluation as ev


def test_top_k_table_matches_hand_computation():
    # 10 leads, 3 conversions; the two highest scores are conversions.
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    p = np.linspace(1.0, 0.1, 10)
    table = ev.top_k_table(y, p, fractions=(0.2, 0.5)).set_index("top_fraction")
    assert table.loc[0.2, "n_contacted"] == 2
    assert table.loc[0.2, "precision"] == 1.0
    assert table.loc[0.2, "capture_rate"] == pytest.approx(2 / 3)
    assert table.loc[0.2, "lift"] == pytest.approx(1.0 / 0.3)
    assert table.loc[0.2, "contacts_per_conversion"] == 1.0
    assert table.loc[0.5, "precision"] == pytest.approx(3 / 5)
    assert table.loc[0.5, "capture_rate"] == 1.0


def test_ties_at_the_cutoff_use_expected_value():
    y = np.array([1, 0, 0, 0])
    p = np.full(4, 0.5)  # constant scores: a random pick
    assert ev.expected_hits_in_top(y, p, 2) == pytest.approx(0.5)
    table = ev.top_k_table(y, p, fractions=(0.5,))
    assert table.loc[0, "lift"] == pytest.approx(1.0)


def test_expected_hits_rejects_bad_k():
    with pytest.raises(ValueError):
        ev.expected_hits_in_top([0, 1], [0.1, 0.2], 0)


def test_probability_metrics_for_constant_predictor():
    y = np.array([0, 0, 0, 1])
    metrics = ev.probability_metrics(y, np.full(4, 0.25))
    assert metrics["base_rate"] == 0.25
    assert metrics["pr_auc"] == pytest.approx(0.25)
    assert metrics["roc_auc"] == 0.5
    assert metrics["brier"] == pytest.approx(metrics["brier_at_segment_base_rate"])


def test_calibration_table_bins_are_equal_count_and_cover_all_rows():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=1000)
    y = rng.binomial(1, p)
    table = ev.calibration_table(y, p, n_bins=10)
    assert table["n"].tolist() == [100] * 10
    assert table["mean_predicted"].is_monotonic_increasing


def test_operating_point_confusion_from_band_counts():
    bands = ev.band_table([1, 1, 0, 1, 0, 0, 0, 0, 1, 0], ["High"] * 2 + ["Medium"] * 3 + ["Low"] * 5)
    table = ev.operating_point_confusion(bands).set_index("operating_point")
    high = table.loc["contact High only"]
    assert (high["true_positive"], high["false_positive"], high["false_negative"], high["true_negative"]) == (2, 0, 2, 6)
    both = table.loc["contact High + Medium"]
    assert (both["true_positive"], both["false_positive"], both["false_negative"], both["true_negative"]) == (3, 2, 1, 4)
    assert both["precision"] == pytest.approx(3 / 5)
    assert both["recall"] == pytest.approx(3 / 4)
    assert both["false_positive_rate"] == pytest.approx(2 / 6)
    for _, row in table.iterrows():
        assert row[["true_positive", "false_positive", "false_negative", "true_negative"]].sum() == 10


def test_unseen_levels_and_subgroups():
    train = pd.DataFrame({"month": ["may", "jun"]})
    other = pd.DataFrame({"month": ["may", "sep", "sep"]})
    assert ev.unseen_levels(train, other, ["month"]) == {"month": {"sep": 2}}
    table = ev.subgroup_table(pd.Series(["a", "a", "b"], name="g"), [1, 0, 1], [0.5, 0.5, 0.2])
    assert table.set_index("g").loc["a", "observed_rate"] == 0.5
    assert table["share_of_conversions"].sum() == pytest.approx(1.0)


def test_paired_bootstrap_uses_shared_resamples_and_is_reproducible():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.3, 400)
    a = y + rng.normal(0, 0.5, 400)
    scores = {"a": a, "same_as_a": a.copy(), "noise": rng.normal(size=400)}
    metrics = {"mean_score": lambda yy, pp: float(np.mean(pp))}
    first = ev.paired_bootstrap(y, scores, metrics, differences=(("a", "same_as_a"),), n_boot=100, seed=7)
    second = ev.paired_bootstrap(y, scores, metrics, differences=(("a", "same_as_a"),), n_boot=100, seed=7)
    assert first == second
    # identical scores on shared resamples -> identical intervals and a zero-width difference interval
    assert first["a"] == first["same_as_a"]
    assert first["differences"]["a - same_as_a"]["mean_score"] == [0.0, 0.0]
    assert first["settings"]["repetitions"] == 100


def test_bootstrap_settings_are_frozen():
    assert (ev.BOOTSTRAP_REPETITIONS, ev.BOOTSTRAP_SEED, ev.BOOTSTRAP_LEVEL) == (2000, 20261006, 0.95)
    assert "independence" in ev.BOOTSTRAP_CAVEAT and "client" in ev.BOOTSTRAP_CAVEAT
