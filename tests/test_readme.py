"""Guard against stale numbers and broken links in the client-facing README."""

import json
import re

import pytest

from src.data import PROJECT_ROOT

README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
REPORTS = PROJECT_ROOT / "reports"


@pytest.fixture(scope="module")
def metrics():
    return json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))


def _top(model_metrics, fraction):
    return next(r for r in model_metrics["top_k"] if r["top_fraction"] == fraction)


def test_headline_numbers_match_saved_test_results(metrics):
    lr = metrics["models"]["logreg_primary"]
    boot = metrics["bootstrap"]["logreg_primary"]
    top20 = _top(lr, 0.2)
    expected = [
        f"{metrics['n']:,} leads",
        f"{metrics['base_rate']:.2%} of these leads converted",
        f"PR-AUC **{lr['pr_auc']:.3f}** (95% interval {boot['pr_auc'][0]:.3f}–{boot['pr_auc'][1]:.3f}",
        f"ROC-AUC **{lr['roc_auc']:.3f}** ({boot['roc_auc'][0]:.3f}–{boot['roc_auc'][1]:.3f})",
        f"**{top20['precision']:.1%}**",
        f"**{top20['capture_rate']:.1%}**",
        f"**{top20['lift']:.2f}×**",
        f"**{top20['contacts_per_conversion']:.2f}** | {1 / metrics['base_rate']:.2f}",
    ]
    for text in expected:
        assert text in README, text


def test_model_comparison_table_matches_saved_results(metrics):
    for name, label in (("logreg_primary", "Logistic regression"), ("xgboost_comparison", "XGBoost")):
        m, boot = metrics["models"][name], metrics["bootstrap"][name]
        row = f"{m['pr_auc']:.3f}** [{boot['pr_auc'][0]:.3f}, {boot['pr_auc'][1]:.3f}]" if name == "logreg_primary" else (
            f"{m['pr_auc']:.3f} [{boot['pr_auc'][0]:.3f}, {boot['pr_auc'][1]:.3f}]"
        )
        assert row in README, (label, row)
    diff = metrics["bootstrap"]["differences"]["logreg_primary - xgboost_comparison"]
    assert f"+{diff['pr_auc'][0]:.3f} to +{diff['pr_auc'][1]:.3f} PR-AUC" in README


def test_business_table_matches_saved_results(metrics):
    for r in metrics["models"]["logreg_primary"]["top_k"]:
        row = (
            f"| Top {int(r['top_fraction'] * 100)}% | {r['n_contacted']:,} | {round(r['expected_conversions']):,} | "
            f"{r['precision']:.1%} | {r['capture_rate']:.1%} | {r['lift']:.2f} | {r['contacts_per_conversion']:.2f} |"
        )
        assert row in README, row


def test_calibration_numbers_match_saved_results(metrics):
    c = metrics["primary_calibration"]
    assert f"predicted {c['mean_predicted_after']:.1%}" in README
    assert f"Brier score was {c['brier_before']:.3f} before calibration and {c['brier_after']:.3f} after" in README
    assert f"{c['brier_at_segment_base_rate']:.3f} from always predicting" in README


def test_random_split_lift_matches_diagnostic(metrics):
    diag = json.loads((REPORTS / "random_split_diagnostic.json").read_text(encoding="utf-8"))
    random_lift = _top(diag["models"]["logreg_primary"], 0.1)["lift"]
    chrono_lift = _top(metrics["models"]["logreg_primary"], 0.1)["lift"]
    assert f"({random_lift:.2f}× vs {chrono_lift:.2f}×)" in README
    assert diag["models"]["xgboost_comparison"]["pr_auc"] > diag["models"]["logreg_primary"]["pr_auc"]


def test_band_table_matches_saved_results():
    import pandas as pd

    validation = pd.read_csv(REPORTS / "bands_validation.csv").set_index("band")
    test = pd.read_csv(REPORTS / "bands_test_secondary.csv")
    batch = test[test["segment"].str.contains("batch")].set_index("band")
    for band in ("High", "Medium", "Low"):
        v, t = validation.loc[band], batch.loc[band]
        row = (
            f"| {band} | {v['conversion_rate']:.1%} ({v['lift']:.2f}×) | {t['conversion_rate']:.1%} ({t['lift']:.2f}×) |"
        )
        assert row in README, row


def test_relative_links_and_images_exist():
    targets = re.findall(r"\]\(((?!https?://|#)[^)\s]+)\)", README)
    assert targets
    for target in targets:
        assert (PROJECT_ROOT / target).exists(), target


def test_no_overclaiming_language():
    lowered = README.lower()
    for phrase in ("chance of converting", "guaranteed", "production-ready", "will convert"):
        assert phrase not in lowered, phrase
