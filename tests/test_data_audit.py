import pandas as pd

from src import audit
from src import features as ft


def test_infer_period_advances_year_when_month_wraps():
    raw = pd.DataFrame({"month": ["may", "may", "dec", "mar", "nov", "mar"]})
    assert audit.infer_period(raw).tolist() == [
        "2008-05",
        "2008-05",
        "2008-12",
        "2009-03",
        "2009-11",
        "2010-03",
    ]


def test_period_is_not_a_model_feature():
    for feature_set in ft.FEATURE_SETS:
        assert "period" not in ft.feature_names(feature_set)


def test_contiguous_segments_cover_all_rows_without_overlap(raw_sample):
    raw = pd.concat([raw_sample] * 4, ignore_index=True)
    segments = audit.contiguous_segment_summary(raw)
    assert segments["n_rows"].sum() == len(raw)
    assert segments["first_row"].tolist()[0] == 0
    assert (segments["first_row"].iloc[1:].values == segments["last_row"].iloc[:-1].values + 1).all()


# --- checks against the real file (skipped if it has not been downloaded) ----


def test_real_file_matches_documented_schema(raw_full):
    schema = audit.check_schema(raw_full)
    assert schema["n_rows"] == audit.EXPECTED_ROWS
    assert schema["columns_match_expected"]
    assert schema["n_nan_cells"] == 0


def test_real_file_is_in_documented_date_order(raw_full):
    ordering = audit.check_ordering(raw_full)
    assert ordering["each_period_is_one_contiguous_run"]
    assert ordering["consistent_with_documented_range"]


def test_real_file_campaign_includes_current_contact(raw_full):
    assert raw_full["campaign"].min() == 1
    assert ft.build_features(raw_full)["prior_contacts_current_campaign"].min() == 0


def test_real_file_builds_leak_free_features(raw_full):
    for feature_set in ft.FEATURE_SETS:
        built = ft.build_features(raw_full, feature_set)
        assert list(built.columns) == ft.feature_names(feature_set)
        assert len(built) == len(raw_full)
    primary = ft.build_features(raw_full, ft.PDAYS_SENSITIVITY)
    known = primary["prior_contact_days_known"] == 1
    assert primary.loc[~known, "days_since_prior_contact"].isna().all()
    assert primary.loc[known, "days_since_prior_contact"].notna().all()
    assert primary.loc[known, "days_since_prior_contact"].max() < ft.PDAYS_SENTINEL
