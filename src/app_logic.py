"""App-facing logic: input schema, validation, conversion to raw rows, scoring tables, explanations.

Kept free of Streamlit so it can be unit-tested. It loads only the committed
deployment artifact; it never reads the UCI dataset and never fits anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import CATEGORY_LABELS, FEATURE_LABELS
from src.inference import DeploymentScorer, load_deployment_scorer

DEPLOY_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "models" / "lead-scorer-1.0.0-deploy.joblib"
EXAMPLE_CSV = Path(__file__).resolve().parents[1] / "app_assets" / "example_leads.csv"

# App / CSV field -> raw column used by the frozen feature builder.
CATEGORICAL_FIELDS = {
    "job": "job",
    "marital_status": "marital",
    "education": "education",
    "credit_default": "default",
    "housing_loan": "housing",
    "personal_loan": "loan",
    "contact_channel": "contact",
    "call_weekday": "day_of_week",
    "previous_campaign_outcome": "poutcome",
}
NUMERIC_FIELDS = ("age", "earlier_calls_this_campaign", "earlier_campaign_contacts")
INPUT_COLUMNS = (
    "age", "job", "marital_status", "education", "credit_default", "housing_loan", "personal_loan",
    "contact_channel", "call_weekday", "earlier_calls_this_campaign", "earlier_campaign_contacts",
    "previous_campaign_outcome",
)
FIELD_LABELS = {
    "age": "Age",
    "job": "Job type",
    "marital_status": "Marital status",
    "education": "Education",
    "credit_default": "Credit in default",
    "housing_loan": "Housing loan",
    "personal_loan": "Personal loan",
    "contact_channel": "Contact channel",
    "call_weekday": "Planned call weekday",
    "earlier_calls_this_campaign": "Earlier calls in this campaign",
    "earlier_campaign_contacts": "Contacts in earlier campaigns",
    "previous_campaign_outcome": "Outcome of the previous campaign",
}
PRIORITY_COLUMNS = (
    "batch_rank", "batch_priority", "batch_percentile", "historical_reference_priority", "historical_percentile",
)
DETAIL_COLUMNS = ("ranking_score", "historical_calibrated_estimate", "outside_training_range")
# Scored file layout: row id, priorities first, then the inputs, then technical/secondary columns.
SCORED_COLUMNS = ("input_row", *PRIORITY_COLUMNS, *INPUT_COLUMNS, *DETAIL_COLUMNS)
NO_EARLIER_CAMPAIGN = "nonexistent"

# Raw UCI columns accepted in uploaded files, mapped to app fields.
_RAW_TO_FIELD = {raw: field for field, raw in CATEGORICAL_FIELDS.items()} | {"age": "age", "previous": "earlier_campaign_contacts"}


def load_scorer(path: Path = DEPLOY_ARTIFACT) -> DeploymentScorer:
    return load_deployment_scorer(path)


def categories(scorer: DeploymentScorer, field: str) -> list[str]:
    """Valid raw codes for an app field, taken from the frozen training schema."""
    return list(scorer.input_schema["categories"][CATEGORICAL_FIELDS[field]])


def outcome_options(scorer: DeploymentScorer, earlier_campaign_contacts: int) -> list[str]:
    """Previous-campaign outcomes consistent with the number of earlier-campaign contacts."""
    if earlier_campaign_contacts == 0:
        return [NO_EARLIER_CAMPAIGN]
    return [c for c in categories(scorer, "previous_campaign_outcome") if c != NO_EARLIER_CAMPAIGN]


def display_label(raw_column: str, code) -> str:
    return CATEGORY_LABELS.get(raw_column, {}).get(code, str(code))


def validate_lead(scorer: DeploymentScorer, lead: dict) -> list[str]:
    """Return a list of problems; an empty list means the lead can be scored."""
    problems = []
    missing = [c for c in INPUT_COLUMNS if c not in lead or pd.isna(lead[c])]
    if missing:
        return [f"missing {', '.join(missing)}"]
    limits = scorer.input_schema["numeric_limits"]
    for field in NUMERIC_FIELDS:
        value = lead[field]
        try:
            number = float(value)
        except (TypeError, ValueError):
            problems.append(f"{field} is not a number")
            continue
        if number != int(number):
            problems.append(f"{field} must be a whole number")
        low, high = limits[field]
        if not low <= number <= high:
            problems.append(f"{field} must be between {low} and {high}")
    for field, raw in CATEGORICAL_FIELDS.items():
        if lead[field] not in scorer.input_schema["categories"][raw]:
            problems.append(f"{field} '{lead[field]}' is not a known value")
    if not problems:
        contacts = int(float(lead["earlier_campaign_contacts"]))
        outcome = lead["previous_campaign_outcome"]
        if contacts == 0 and outcome != NO_EARLIER_CAMPAIGN:
            problems.append("previous_campaign_outcome must be 'nonexistent' when there were no earlier-campaign contacts")
        if contacts > 0 and outcome == NO_EARLIER_CAMPAIGN:
            problems.append("previous_campaign_outcome cannot be 'nonexistent' when there were earlier-campaign contacts")
    return problems


def outside_training_range(scorer: DeploymentScorer, leads: pd.DataFrame) -> pd.Series:
    """True where a numeric input lies outside the range seen in the development period."""
    flags = pd.Series(False, index=leads.index)
    for field, (low, high) in scorer.input_schema["training_range"].items():
        values = leads[field].astype(float)
        flags |= (values < low) | (values > high)
    return flags


def to_raw(leads: pd.DataFrame) -> pd.DataFrame:
    """App fields -> raw columns expected by the frozen feature builder.

    ``campaign`` in the source data counts the contact being scored, so it is
    earlier calls + 1; users never see that convention.
    """
    raw = pd.DataFrame(index=leads.index)
    raw["age"] = leads["age"].astype(int)
    for field, column in CATEGORICAL_FIELDS.items():
        raw[column] = leads[field].astype(str)
    raw["campaign"] = leads["earlier_calls_this_campaign"].astype(int) + 1
    raw["previous"] = leads["earlier_campaign_contacts"].astype(int)
    return raw


def normalise_upload(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept either the app template columns or raw UCI-style columns; return only the app fields.

    Any other column (including ``duration``, ``y`` or economic indicators) is dropped and never
    propagated to the scored output.
    """
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    if set(INPUT_COLUMNS) <= set(frame.columns):
        out = frame[list(INPUT_COLUMNS)].copy()
    elif {"campaign", "previous", "poutcome", "marital"} <= set(frame.columns):
        out = frame[[c for c in _RAW_TO_FIELD if c in frame.columns]].rename(columns=_RAW_TO_FIELD)
        out["earlier_calls_this_campaign"] = pd.to_numeric(frame["campaign"], errors="coerce") - 1
        out = out.reindex(columns=list(INPUT_COLUMNS))
    else:
        missing = sorted(set(INPUT_COLUMNS) - set(frame.columns))
        raise ValueError(f"Missing columns: {', '.join(missing)}. Download the template for the expected format.")
    for field in CATEGORICAL_FIELDS:
        out[field] = out[field].astype("string").str.strip()
    return out


def split_valid(scorer: DeploymentScorer, leads: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate scoreable rows from rows with problems (returned with a reason)."""
    problems = leads.apply(lambda row: "; ".join(validate_lead(scorer, row.to_dict())), axis=1)
    valid = leads[problems == ""].copy()
    invalid = leads[problems != ""].assign(problem=problems[problems != ""])
    for field in NUMERIC_FIELDS:
        valid[field] = valid[field].astype(float).astype(int)
    return valid, invalid


def score_batch_table(scorer: DeploymentScorer, valid: pd.DataFrame) -> pd.DataFrame:
    """Scored output: app input fields plus clearly separated batch and historical columns."""
    if valid.empty:
        return pd.DataFrame(columns=list(SCORED_COLUMNS))
    scored = scorer.score_batch(to_raw(valid))
    out = valid[list(INPUT_COLUMNS)].copy()
    out.insert(0, "input_row", valid.index + 1)
    out["batch_rank"] = scored["ranking_score"].rank(ascending=False, method="min").astype(int)
    out["batch_priority"] = scored["batch_priority"]
    out["batch_percentile"] = scored["batch_percentile"].round(1)
    out["historical_reference_priority"] = scored["priority"]
    out["historical_percentile"] = scored["historical_percentile"].round(1)
    out["ranking_score"] = scored["ranking_score"].round(4)
    out["historical_calibrated_estimate"] = scored["calibrated_probability_secondary"].round(4)
    out["outside_training_range"] = outside_training_range(scorer, valid)
    return out[list(SCORED_COLUMNS)].sort_values("batch_rank", kind="stable").reset_index(drop=True)


def score_lead(scorer: DeploymentScorer, lead: dict) -> dict:
    """Score one validated lead: historical priority, percentile, explanation, secondary estimate."""
    problems = validate_lead(scorer, lead)
    if problems:
        raise ValueError("; ".join(problems))
    leads = pd.DataFrame([lead])
    raw = to_raw(leads)
    scored = scorer.score(raw).iloc[0]
    explanation = scorer.explain(raw)
    return {
        "historical_reference_priority": scored["priority"],
        "historical_percentile": float(scored["historical_percentile"]),
        "ranking_score": float(scored["ranking_score"]),
        "historical_calibrated_estimate": float(scored["calibrated_probability_secondary"]),
        "outside_training_range": bool(outside_training_range(scorer, leads).iloc[0]),
        "explanation": friendly_explanation(explanation),
    }


def friendly_explanation(explanation: pd.DataFrame) -> pd.DataFrame:
    """Contributions with plain-language field/value descriptions, largest effect first."""
    rows = []
    for _, row in explanation.iterrows():
        feature, value = row["feature"], row["value"]
        if feature == "prior_contacts_current_campaign":
            shown = str(int(value))
        elif isinstance(value, (int, np.integer, float, np.floating)):
            shown = f"{value:g}"
        else:
            shown = display_label(feature, value)
        rows.append({"field": FEATURE_LABELS[feature], "value": shown, "contribution": float(row["contribution"])})
    return pd.DataFrame(rows)


def top_factors(explanation: pd.DataFrame, n: int = 3, min_abs: float = 0.005) -> tuple[list[str], list[str]]:
    """Plain-language bullets for the strongest factors pushing the score higher and lower."""
    higher = explanation[explanation["contribution"] > min_abs].nlargest(n, "contribution")
    lower = explanation[explanation["contribution"] < -min_abs].nsmallest(n, "contribution")

    def bullet(row):
        return f"{row['field']}: {row['value']}"

    return [bullet(r) for _, r in higher.iterrows()], [bullet(r) for _, r in lower.iterrows()]


def template_frame() -> pd.DataFrame:
    """Empty upload template with the expected columns and one illustrative row."""
    return pd.DataFrame(
        [{
            "age": 35, "job": "management", "marital_status": "married", "education": "university.degree",
            "credit_default": "no", "housing_loan": "yes", "personal_loan": "no", "contact_channel": "cellular",
            "call_weekday": "thu", "earlier_calls_this_campaign": 0, "earlier_campaign_contacts": 1,
            "previous_campaign_outcome": "failure",
        }],
        columns=list(INPUT_COLUMNS),
    )
