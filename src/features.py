"""Single source of truth for feature availability and model feature definitions.

Decision point: a lead is scored immediately before the recorded outbound call
is dialled. A raw column may feed the deployable model only if its value is
known at that moment. Availability is judged on that basis alone, never on
whether the column improves validation performance.

Everything else in the project (pipelines, tests, the audit report, the app)
must read feature lists from this module rather than re-declaring them.

All transformations here are stateless and row-wise: nothing is fitted, so
they cannot leak information across train/validation/test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "y"
PDAYS_SENTINEL = 999

AVAILABLE = "available"
QUESTIONABLE = "questionable"
UNAVAILABLE = "unavailable"

# Feature sets. Only PRIMARY is deployable; the others exist for clearly
# labelled sensitivity analyses and must never drive model selection.
PRIMARY = "primary"
MONTH_SENSITIVITY = "month_sensitivity"  # primary + call month
PDAYS_SENSITIVITY = "pdays_sensitivity"  # primary + pdays-derived features
EXTENDED = "extended"  # primary + economic context
FEATURE_SETS = (PRIMARY, MONTH_SENSITIVITY, PDAYS_SENSITIVITY, EXTENDED)

# Feature groups, and which groups make up each feature set.
_CORE, _MONTH, _PDAYS, _ECONOMIC = "core", "month", "pdays", "economic"
_GROUPS_IN_SET = {
    PRIMARY: (_CORE,),
    MONTH_SENSITIVITY: (_CORE, _MONTH),
    PDAYS_SENSITIVITY: (_CORE, _PDAYS),
    EXTENDED: (_CORE, _ECONOMIC),
}


@dataclass(frozen=True)
class RawColumn:
    """Availability audit entry for one column of the raw UCI file."""

    name: str
    availability: str
    decision: str
    rationale: str


@dataclass(frozen=True)
class ModelFeature:
    """One model input, the raw columns it is built from, and where it may be used."""

    name: str
    kind: str  # "numeric" | "categorical" | "binary"
    sources: tuple[str, ...]
    group: str  # only the core group is deployable


_ECON_RATIONALE = (
    "The file attaches a same-period value to each call, and the dataset does not document "
    "whether that value was available as of the call time. Excluded from the primary "
    "deployable model; evaluated only in the extended sensitivity set. These columns also "
    "take few distinct values and act as a proxy for the calendar period."
)

RAW_COLUMNS: tuple[RawColumn, ...] = (
    RawColumn("age", AVAILABLE, "use", "Client attribute held by the bank before the call."),
    RawColumn("job", AVAILABLE, "use", "Client attribute held before the call; 'unknown' kept as a category."),
    RawColumn("marital", AVAILABLE, "use", "Client attribute held before the call; 'unknown' kept as a category."),
    RawColumn("education", AVAILABLE, "use", "Client attribute held before the call; 'unknown' kept as a category."),
    RawColumn("default", AVAILABLE, "use", "Bank record held before the call; 'unknown' kept as a category."),
    RawColumn("housing", AVAILABLE, "use", "Bank record held before the call; 'unknown' kept as a category."),
    RawColumn("loan", AVAILABLE, "use", "Bank record held before the call; 'unknown' kept as a category."),
    RawColumn("contact", AVAILABLE, "use", "Channel of the planned call, chosen before dialling."),
    RawColumn(
        "month",
        AVAILABLE,
        "sensitivity only",
        "Month of the call being dialled, known at dial time, so not a leakage risk. Dropped from the "
        "primary model on generalisation grounds: with 26 months of data its learned effect is a "
        "calendar-period effect, and later months (e.g. March, April, September, December) are absent "
        "or rare in earlier training data. Evaluated only in the month sensitivity set.",
    ),
    RawColumn("day_of_week", AVAILABLE, "use", "Weekday of the call being dialled, known at dial time."),
    RawColumn(
        "duration",
        UNAVAILABLE,
        "exclude",
        "Length of the call being scored; known only after it ends, when the outcome is also known. "
        "Leakage-demonstration use only.",
    ),
    RawColumn(
        "campaign",
        QUESTIONABLE,
        "derive",
        "Counts contacts in this campaign including the one being scored (observed minimum is 1). "
        "Not used raw; replaced by prior_contacts_current_campaign = campaign - 1, the number of "
        "contacts already made, which is known before dialling.",
    ),
    RawColumn(
        "pdays",
        AVAILABLE,
        "derive (sensitivity only)",
        "History from an earlier campaign. 999 is a sentinel for 'not previously contacted', so it is "
        "split into an indicator plus a days value that is missing when the sentinel is present. "
        "The indicator is not 'ever contacted before': some rows have previous > 0 with pdays = 999; "
        "previous and poutcome carry that information. Available, but only 6 of 24,712 training rows "
        "have a recorded value, too few to learn its effect, so the derived features are kept out of "
        "the primary model and evaluated only in a sensitivity set. This is a usefulness decision, "
        "not an availability one.",
    ),
    RawColumn("previous", AVAILABLE, "use", "Contacts in earlier campaigns; fixed before this campaign."),
    RawColumn("poutcome", AVAILABLE, "use", "Outcome of the earlier campaign; fixed before this campaign."),
    RawColumn(
        "emp.var.rate",
        QUESTIONABLE,
        "sensitivity only",
        _ECON_RATIONALE,
    ),
    RawColumn(
        "cons.price.idx",
        QUESTIONABLE,
        "sensitivity only",
        _ECON_RATIONALE,
    ),
    RawColumn(
        "cons.conf.idx",
        QUESTIONABLE,
        "sensitivity only",
        _ECON_RATIONALE,
    ),
    RawColumn(
        "euribor3m",
        QUESTIONABLE,
        "sensitivity only",
        _ECON_RATIONALE,
    ),
    RawColumn(
        "nr.employed",
        QUESTIONABLE,
        "sensitivity only",
        _ECON_RATIONALE,
    ),
    RawColumn(TARGET, UNAVAILABLE, "target", "The outcome being predicted."),
)

MODEL_FEATURES: tuple[ModelFeature, ...] = (
    ModelFeature("age", "numeric", ("age",), _CORE),
    ModelFeature("job", "categorical", ("job",), _CORE),
    ModelFeature("marital", "categorical", ("marital",), _CORE),
    ModelFeature("education", "categorical", ("education",), _CORE),
    ModelFeature("default", "categorical", ("default",), _CORE),
    ModelFeature("housing", "categorical", ("housing",), _CORE),
    ModelFeature("loan", "categorical", ("loan",), _CORE),
    ModelFeature("contact", "categorical", ("contact",), _CORE),
    ModelFeature("month", "categorical", ("month",), _MONTH),
    ModelFeature("day_of_week", "categorical", ("day_of_week",), _CORE),
    ModelFeature("prior_contacts_current_campaign", "numeric", ("campaign",), _CORE),
    ModelFeature("prior_contact_days_known", "binary", ("pdays",), _PDAYS),
    ModelFeature("days_since_prior_contact", "numeric", ("pdays",), _PDAYS),
    ModelFeature("previous", "numeric", ("previous",), _CORE),
    ModelFeature("poutcome", "categorical", ("poutcome",), _CORE),
    ModelFeature("emp.var.rate", "numeric", ("emp.var.rate",), _ECONOMIC),
    ModelFeature("cons.price.idx", "numeric", ("cons.price.idx",), _ECONOMIC),
    ModelFeature("cons.conf.idx", "numeric", ("cons.conf.idx",), _ECONOMIC),
    ModelFeature("euribor3m", "numeric", ("euribor3m",), _ECONOMIC),
    ModelFeature("nr.employed", "numeric", ("nr.employed",), _ECONOMIC),
)

# Plain-language names for client-facing explanations (every model feature must have one).
FEATURE_LABELS: dict[str, str] = {
    "age": "Age",
    "job": "Job type",
    "marital": "Marital status",
    "education": "Education",
    "default": "Credit in default",
    "housing": "Housing loan",
    "loan": "Personal loan",
    "contact": "Contact channel",
    "month": "Call month",
    "day_of_week": "Call weekday",
    "prior_contacts_current_campaign": "Earlier calls in this campaign",
    "prior_contact_days_known": "Days since earlier-campaign contact recorded",
    "days_since_prior_contact": "Days since earlier-campaign contact",
    "previous": "Contacts in earlier campaigns",
    "poutcome": "Earlier campaign outcome",
    "emp.var.rate": "Employment variation rate",
    "cons.price.idx": "Consumer price index",
    "cons.conf.idx": "Consumer confidence index",
    "euribor3m": "3-month Euribor rate",
    "nr.employed": "Number employed (national)",
}

# Plain-language names for category values (display only; model inputs keep the raw codes).
CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "job": {
        "admin.": "Administrative", "blue-collar": "Blue-collar", "entrepreneur": "Entrepreneur",
        "housemaid": "Housemaid", "management": "Management", "retired": "Retired",
        "self-employed": "Self-employed", "services": "Services", "student": "Student",
        "technician": "Technician", "unemployed": "Unemployed", "unknown": "Unknown",
    },
    "marital": {"divorced": "Divorced or widowed", "married": "Married", "single": "Single", "unknown": "Unknown"},
    "education": {
        "basic.4y": "Basic, 4 years", "basic.6y": "Basic, 6 years", "basic.9y": "Basic, 9 years",
        "high.school": "High school", "illiterate": "No formal education", "professional.course": "Professional course",
        "university.degree": "University degree", "unknown": "Unknown",
    },
    "default": {"no": "No", "yes": "Yes", "unknown": "Unknown"},
    "housing": {"no": "No", "yes": "Yes", "unknown": "Unknown"},
    "loan": {"no": "No", "yes": "Yes", "unknown": "Unknown"},
    "contact": {"cellular": "Mobile phone", "telephone": "Landline"},
    "day_of_week": {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday", "fri": "Friday"},
    "poutcome": {"nonexistent": "No earlier campaign", "failure": "Did not subscribe", "success": "Subscribed"},
}

# Raw columns that must never reach any model, and raw columns that may only
# enter through a derived feature.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(c.name for c in RAW_COLUMNS if c.availability == UNAVAILABLE)
DERIVE_ONLY_COLUMNS: frozenset[str] = frozenset(c.name for c in RAW_COLUMNS if c.decision.startswith("derive"))


def _features_in(feature_set: str) -> tuple[ModelFeature, ...]:
    if feature_set not in _GROUPS_IN_SET:
        raise ValueError(f"Unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}")
    return tuple(f for f in MODEL_FEATURES if f.group in _GROUPS_IN_SET[feature_set])


def feature_names(feature_set: str = PRIMARY, kind: str | None = None) -> list[str]:
    """Model input names for a feature set, optionally filtered by kind."""
    return [f.name for f in _features_in(feature_set) if kind is None or f.kind == kind]


def required_raw_columns(feature_set: str = PRIMARY) -> list[str]:
    """Raw columns a caller must supply to build a feature set."""
    seen: dict[str, None] = {}
    for feature in _features_in(feature_set):
        for source in feature.sources:
            seen[source] = None
    return list(seen)


def assert_no_leakage(columns) -> None:
    """Raise if any forbidden or derive-only raw column appears among model inputs."""
    leaked = sorted((FORBIDDEN_COLUMNS | DERIVE_ONLY_COLUMNS) & set(columns))
    if leaked:
        raise ValueError(f"Columns not allowed as model inputs: {leaked}")


def build_features(raw: pd.DataFrame, feature_set: str = PRIMARY) -> pd.DataFrame:
    """Turn raw rows into model inputs for the given feature set.

    Columns outside the feature set (including ``duration`` and ``y``, if the
    caller's file happens to contain them) are ignored and never passed through.
    """
    required = required_raw_columns(feature_set)
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    campaign = raw["campaign"]
    if (campaign < 1).any():
        raise ValueError("campaign must be >= 1: it counts contacts including the one being scored")
    derived = {"prior_contacts_current_campaign": campaign - 1}

    if "pdays" in required:
        pdays = raw["pdays"]
        has_pdays = pdays != PDAYS_SENTINEL
        derived["prior_contact_days_known"] = has_pdays.astype("int8")
        # Missing, not 999, when there was no recorded prior contact; the
        # modelling pipeline imputes it and relies on the indicator above.
        derived["days_since_prior_contact"] = pdays.where(has_pdays, np.nan).astype("float64")

    names = feature_names(feature_set)
    features = pd.DataFrame(
        {name: derived[name] if name in derived else raw[name] for name in names},
        index=raw.index,
    )
    assert_no_leakage(features.columns)
    return features


def make_target(raw: pd.DataFrame) -> pd.Series:
    """Binary target: 1 if the client subscribed, else 0."""
    values = raw[TARGET]
    unexpected = set(values.unique()) - {"yes", "no"}
    if unexpected:
        raise ValueError(f"Unexpected target values: {sorted(unexpected)}")
    return (values == "yes").astype("int8").rename(TARGET)


def availability_table() -> pd.DataFrame:
    """Feature-availability audit, one row per raw column."""
    rows = []
    for column in RAW_COLUMNS:
        used_by = [f for f in MODEL_FEATURES if column.name in f.sources]
        rows.append(
            {
                "raw_column": column.name,
                "availability_at_scoring": column.availability,
                "decision": column.decision,
                "model_features": ", ".join(f.name for f in used_by),
                "in_primary_model": any(f.group == _CORE for f in used_by),
                "sensitivity_only": bool(used_by) and all(f.group != _CORE for f in used_by),
                "rationale": column.rationale,
            }
        )
    return pd.DataFrame(rows)
