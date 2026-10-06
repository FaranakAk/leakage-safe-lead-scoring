"""Phase 5: derive the minimal deployment artifact from the frozen Phase 4 artifact (no refitting).

Usage: python scripts/build_deployment_artifact.py

Writes (committed, so the app needs neither training nor the dataset):
    artifacts/models/lead-scorer-1.0.0-deploy.joblib
    artifacts/models/lead-scorer-1.0.0-deploy.json
    app_assets/example_leads.csv   (200 validation-period leads, outcome removed; UCI data, CC BY 4.0)

Then checks that the deployment artifact reproduces the Phase 4 artifact exactly on validation leads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import app_logic as app  # noqa: E402
from src import explainability as xp  # noqa: E402
from src import inference as inf  # noqa: E402
from src.data import PROJECT_ROOT, load_raw, sha256_of  # noqa: E402
from src.splits import chronological_split  # noqa: E402

FULL_ARTIFACT = PROJECT_ROOT / "artifacts" / "models" / f"{inf.ARTIFACT_VERSION}.joblib"
DEPLOY_METADATA = app.DEPLOY_ARTIFACT.with_suffix(".json")

# Logical limits for app inputs (impossible values are rejected); the development-period range is
# stored separately so out-of-range but plausible values can be flagged rather than rejected.
NUMERIC_LIMITS = {"age": [17, 100], "earlier_calls_this_campaign": [0, 60], "earlier_campaign_contacts": [0, 10]}


def input_schema(scorer: inf.LeadScorer, development: pd.DataFrame) -> dict:
    encoder = scorer.pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    columns = scorer.pipeline.named_steps["preprocess"].transformers_[1][2]
    return {
        "categories": {column: [str(c) for c in cats] for column, cats in zip(columns, encoder.categories_)},
        "numeric_limits": NUMERIC_LIMITS,
        "training_range": {
            "age": [int(development["age"].min()), int(development["age"].max())],
            "earlier_calls_this_campaign": [int(development["campaign"].min() - 1), int(development["campaign"].max() - 1)],
            "earlier_campaign_contacts": [int(development["previous"].min()), int(development["previous"].max())],
        },
        "rules": [
            "earlier_campaign_contacts == 0  <=>  previous_campaign_outcome == 'nonexistent' (holds for every row "
            "of the audited data)",
            "earlier_calls_this_campaign = campaign - 1 (the source field counts the call being scored)",
        ],
        "never_inputs": ["duration", "y", "pdays", "month", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                         "euribor3m", "nr.employed"],
    }


def to_app_fields(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw[list(app._RAW_TO_FIELD)].rename(columns=app._RAW_TO_FIELD)
    out["earlier_calls_this_campaign"] = raw["campaign"] - 1
    return out[list(app.INPUT_COLUMNS)].reset_index(drop=True)


def main() -> None:
    full = inf.load_scorer(FULL_ARTIFACT)
    raw = load_raw()
    split = chronological_split(raw["month"])
    development, validation = raw.iloc[split["development"]], raw.iloc[split["validation"]]

    deploy = inf.build_deployment_scorer(
        full,
        input_schema=input_schema(full, development),
        metadata={
            "derived_from_sha256": sha256_of(FULL_ARTIFACT),
            "explanation": "closed-form linear SHAP: coef * (x - mean of transformed development background); "
            "equals shap.LinearExplainer with an independent masker on the Phase 4 background",
            "app_outputs": {
                "historical_reference_priority": "band from the frozen validation-period cutoffs",
                "historical_percentile": "share of validation-period (Apr - May 2009) reference leads at or below",
                "batch_priority / batch_percentile": "same 20/30/50 rule applied within the uploaded batch",
                "historical_calibrated_estimate": "secondary; Platt calibration fitted Apr - May 2009; drifted later",
            },
        },
    )
    inf.save_scorer(deploy, app.DEPLOY_ARTIFACT)
    meta = {k: v for k, v in deploy.metadata.items()}
    DEPLOY_METADATA.write_text(
        json.dumps({**meta, "feature_names": deploy.feature_names, "input_schema": deploy.input_schema,
                    "explanation_base_value": deploy.explanation_base_value}, indent=2, default=float),
        encoding="utf-8",
    )

    app.EXAMPLE_CSV.parent.mkdir(exist_ok=True)
    to_app_fields(validation.sample(n=200, random_state=11).sort_index()).to_csv(app.EXAMPLE_CSV, index=False)

    # Equivalence with the Phase 4 artifact on validation leads.
    reloaded = inf.load_deployment_scorer(app.DEPLOY_ARTIFACT)
    a, b = full.score(validation), reloaded.score(validation)
    assert a.equals(b), "scores differ"
    shap_contrib = xp.field_contributions(full.pipeline, full.explainer(), validation)
    np.testing.assert_allclose(reloaded.contributions(validation), shap_contrib, atol=1e-10)
    print(f"Deployment artifact: {app.DEPLOY_ARTIFACT} ({app.DEPLOY_ARTIFACT.stat().st_size / 1024:.1f} KB)")
    print(f"Full artifact:       {FULL_ARTIFACT} ({FULL_ARTIFACT.stat().st_size / 1024:.1f} KB)")
    print("Identical scores, percentiles, priorities and calibrated estimates on", len(validation), "validation leads;")
    print("contributions match shap.LinearExplainer to 1e-10.")


if __name__ == "__main__":
    main()
