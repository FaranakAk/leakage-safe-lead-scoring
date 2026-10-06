import numpy as np
import pandas as pd
import pytest

from src import features as ft

PDAYS_DERIVED = {"prior_contact_days_known", "days_since_prior_contact"}
ECONOMIC = {"emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"}


# --- leakage exclusions -----------------------------------------------------


@pytest.mark.parametrize("feature_set", ft.FEATURE_SETS)
def test_forbidden_columns_never_in_any_feature_set(feature_set):
    names = set(ft.feature_names(feature_set))
    sources = set(ft.required_raw_columns(feature_set))
    assert "duration" not in names | sources
    assert ft.TARGET not in names | sources


def test_forbidden_columns_are_duration_and_target():
    assert ft.FORBIDDEN_COLUMNS == {"duration", ft.TARGET}


@pytest.mark.parametrize("feature_set", ft.FEATURE_SETS)
def test_raw_campaign_and_pdays_are_not_direct_inputs(feature_set):
    names = set(ft.feature_names(feature_set))
    assert "campaign" not in names
    assert "pdays" not in names


def test_primary_set_excludes_economic_indicators():
    assert not ECONOMIC & set(ft.feature_names(ft.PRIMARY))
    assert not ECONOMIC & set(ft.required_raw_columns(ft.PRIMARY))
    assert ECONOMIC <= set(ft.feature_names(ft.EXTENDED))


def test_primary_set_excludes_pdays_derived_features():
    assert not PDAYS_DERIVED & set(ft.feature_names(ft.PRIMARY))
    assert "pdays" not in ft.required_raw_columns(ft.PRIMARY)
    assert PDAYS_DERIVED <= set(ft.feature_names(ft.PDAYS_SENSITIVITY))


def test_month_is_sensitivity_only_and_day_of_week_is_primary():
    assert "month" not in ft.feature_names(ft.PRIMARY)
    assert "month" not in ft.required_raw_columns(ft.PRIMARY)
    assert "day_of_week" in ft.feature_names(ft.PRIMARY)
    assert set(ft.feature_names(ft.MONTH_SENSITIVITY)) == set(ft.feature_names(ft.PRIMARY)) | {"month"}


def test_primary_feature_list_is_exactly_as_frozen():
    assert ft.feature_names(ft.PRIMARY) == [
        "age", "job", "marital", "education", "default", "housing", "loan", "contact",
        "day_of_week", "prior_contacts_current_campaign", "previous", "poutcome",
    ]


def test_sensitivity_sets_are_primary_plus_one_group():
    primary = ft.feature_names(ft.PRIMARY)
    assert set(ft.feature_names(ft.PDAYS_SENSITIVITY)) == set(primary) | PDAYS_DERIVED
    assert set(ft.feature_names(ft.EXTENDED)) == set(primary) | ECONOMIC


@pytest.mark.parametrize("feature_set", ft.FEATURE_SETS)
def test_build_features_drops_forbidden_columns_present_in_input(raw_sample, feature_set):
    built = ft.build_features(raw_sample, feature_set)
    assert list(built.columns) == ft.feature_names(feature_set)
    assert not (ft.FORBIDDEN_COLUMNS | ft.DERIVE_ONLY_COLUMNS) & set(built.columns)


def test_build_features_does_not_depend_on_forbidden_columns(raw_sample):
    without = raw_sample.drop(columns=["duration", ft.TARGET])
    changed = raw_sample.assign(duration=raw_sample["duration"] + 1000, y="yes")
    expected = ft.build_features(raw_sample)
    pd.testing.assert_frame_equal(ft.build_features(without), expected)
    pd.testing.assert_frame_equal(ft.build_features(changed), expected)


@pytest.mark.parametrize("column", ["duration", "y", "campaign", "pdays"])
def test_assert_no_leakage_rejects(column):
    with pytest.raises(ValueError, match=column):
        ft.assert_no_leakage(["age", column])


def test_assert_no_leakage_accepts_model_features():
    for feature_set in ft.FEATURE_SETS:
        ft.assert_no_leakage(ft.feature_names(feature_set))


# --- registry consistency (single source of truth) --------------------------


def test_every_raw_column_is_audited_once():
    names = [c.name for c in ft.RAW_COLUMNS]
    assert len(names) == len(set(names)) == 21


def test_model_features_only_use_audited_non_forbidden_sources():
    by_name = {c.name: c for c in ft.RAW_COLUMNS}
    for feature in ft.MODEL_FEATURES:
        for source in feature.sources:
            assert by_name[source].availability != ft.UNAVAILABLE


def test_primary_features_use_only_available_or_derived_sources():
    by_name = {c.name: c for c in ft.RAW_COLUMNS}
    for source in ft.required_raw_columns(ft.PRIMARY):
        column = by_name[source]
        assert column.availability == ft.AVAILABLE or column.decision == "derive"


def test_feature_kinds_partition_the_feature_set():
    for feature_set in ft.FEATURE_SETS:
        by_kind = sum((ft.feature_names(feature_set, kind) for kind in ("numeric", "categorical", "binary")), [])
        assert sorted(by_kind) == sorted(ft.feature_names(feature_set))


def test_availability_table_matches_registry():
    table = ft.availability_table().set_index("raw_column")
    assert len(table) == 21
    for column in ("duration", ft.TARGET):
        assert not table.loc[column, "in_primary_model"]
        assert table.loc[column, "model_features"] == ""
    assert table.loc["campaign", "model_features"] == "prior_contacts_current_campaign"
    assert table.loc["campaign", "in_primary_model"]
    for column in ECONOMIC | {"pdays", "month"}:
        assert not table.loc[column, "in_primary_model"]
        assert table.loc[column, "sensitivity_only"]


# --- transformations --------------------------------------------------------


def test_prior_contacts_is_campaign_minus_one(raw_sample):
    built = ft.build_features(raw_sample)
    assert built["prior_contacts_current_campaign"].tolist() == [0, 3, 1]


def test_campaign_below_one_is_rejected(raw_sample):
    with pytest.raises(ValueError, match="campaign"):
        ft.build_features(raw_sample.assign(campaign=[0, 1, 2]))


def test_pdays_sentinel_becomes_indicator_and_missing(raw_sample):
    built = ft.build_features(raw_sample, ft.PDAYS_SENSITIVITY)
    assert built["prior_contact_days_known"].tolist() == [0, 0, 1]
    days = built["days_since_prior_contact"]
    assert np.isnan(days.iloc[0]) and np.isnan(days.iloc[1])
    assert days.iloc[2] == 0  # a genuine zero stays zero, distinct from the sentinel
    assert ft.PDAYS_SENTINEL not in days.dropna().tolist()


def test_build_features_preserves_index_and_does_not_mutate_input(raw_sample):
    raw = raw_sample.set_axis([10, 20, 30])
    before = raw.copy()
    built = ft.build_features(raw)
    assert list(built.index) == [10, 20, 30]
    pd.testing.assert_frame_equal(raw, before)


def test_missing_required_column_raises(raw_sample):
    with pytest.raises(ValueError, match="age"):
        ft.build_features(raw_sample.drop(columns=["age"]))


def test_primary_features_do_not_require_economic_columns(raw_sample):
    built = ft.build_features(raw_sample.drop(columns=sorted(ECONOMIC)))
    assert list(built.columns) == ft.feature_names(ft.PRIMARY)
    with pytest.raises(ValueError, match="euribor3m"):
        ft.build_features(raw_sample.drop(columns=sorted(ECONOMIC)), ft.EXTENDED)


def test_unknown_feature_set_raises():
    with pytest.raises(ValueError):
        ft.feature_names("everything")


def test_make_target(raw_sample):
    assert ft.make_target(raw_sample).tolist() == [0, 0, 1]
    with pytest.raises(ValueError):
        ft.make_target(raw_sample.assign(y=["no", "maybe", "yes"]))
