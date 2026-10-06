"""SHAP explanations for the frozen logistic-regression ranking model.

* Explains the model's **ranking score** (log-odds from the uncalibrated logistic
  regression), not the calibrated probability.
* The SHAP background (reference) data comes from the development period only.
* One-hot columns are summed back to their original business field, which is
  valid because SHAP values are additive.
* Contributions are model associations: "this field pushed the score higher",
  never "this field causes conversion".

Two equivalent routes produce the same contributions:

* ``make_explainer`` + ``field_contributions``: the SHAP library's LinearExplainer
  (used to build and validate the Phase 4 outputs);
* ``linear_field_contributions``: the closed form coef * (x - background mean),
  which is what SHAP computes for a linear model with an independent masker. It
  needs only the background mean and no ``shap`` import, so the deployed app
  uses it.

Nothing here fits or modifies the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import FEATURE_LABELS

BACKGROUND_SIZE = 1000
BACKGROUND_SEED = 7


def transformed_to_field(pipeline: Pipeline) -> dict[str, str]:
    """Map each transformed (e.g. one-hot) column name to the model feature it came from."""
    preprocess = pipeline.named_steps["preprocess"]
    mapping = {}
    for name, transformer, columns in preprocess.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if hasattr(transformer, "categories_"):
            for column, categories in zip(columns, transformer.categories_):
                for category in categories:
                    mapping[f"{column}_{category}"] = column
        else:
            for column in columns:
                mapping[column] = column
    expected = list(preprocess.get_feature_names_out())
    if sorted(mapping) != sorted(expected):
        raise ValueError("Transformed column names do not match the preprocessing output")
    return mapping


def background_sample(raw_development: pd.DataFrame, size: int = BACKGROUND_SIZE, seed: int = BACKGROUND_SEED) -> pd.DataFrame:
    """Fixed random sample of development rows used as the SHAP reference distribution."""
    return raw_development.sample(n=min(size, len(raw_development)), random_state=seed)


def make_explainer(pipeline: Pipeline, background_raw: pd.DataFrame):
    """Linear SHAP explainer on the transformed features (independent-features masker)."""
    import shap  # imported lazily so the deployed app does not need it

    transformed = _transform(pipeline, background_raw)
    return shap.LinearExplainer(pipeline.named_steps["model"], shap.maskers.Independent(transformed, max_samples=len(transformed)))


def _transform(pipeline: Pipeline, raw: pd.DataFrame) -> np.ndarray:
    features = pipeline.named_steps["features"].transform(raw)
    return np.asarray(pipeline.named_steps["preprocess"].transform(features), dtype=float)


def background_mean(pipeline: Pipeline, background_raw: pd.DataFrame) -> np.ndarray:
    """Mean of the transformed background rows: all a linear explanation needs from the background."""
    return _transform(pipeline, background_raw).mean(axis=0)


def linear_base_value(pipeline: Pipeline, mean_transformed: np.ndarray) -> float:
    """Ranking score of an 'average background lead': intercept + coef . background mean."""
    model = pipeline.named_steps["model"]
    return float(model.intercept_[0] + model.coef_[0] @ mean_transformed)


def linear_field_contributions(pipeline: Pipeline, mean_transformed: np.ndarray, raw: pd.DataFrame) -> pd.DataFrame:
    """Closed-form linear SHAP contributions, coef * (x - background mean), summed per business field."""
    values = pipeline.named_steps["model"].coef_[0] * (_transform(pipeline, raw) - mean_transformed)
    return _aggregate(pipeline, values, raw.index)


def field_contributions(pipeline: Pipeline, explainer, raw: pd.DataFrame) -> pd.DataFrame:
    """SHAP contributions to the ranking score (log-odds), summed per business field.

    Returns one row per lead and one column per model feature. Each row's values
    plus ``base_value(explainer)`` equal the lead's ranking score.
    """
    return _aggregate(pipeline, np.asarray(explainer.shap_values(_transform(pipeline, raw))), raw.index)


def _aggregate(pipeline: Pipeline, values: np.ndarray, index) -> pd.DataFrame:
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    mapping = transformed_to_field(pipeline)
    by_column = pd.DataFrame(values, columns=names, index=index)
    fields = by_column.T.groupby(by_column.columns.map(mapping), sort=False).sum().T
    return fields[list(pipeline.named_steps["preprocess"].feature_names_in_)]  # model feature order


def base_value(explainer) -> float:
    return float(np.ravel(explainer.expected_value)[0])


def global_importance(contributions: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute contribution per field, largest first, with plain-language labels."""
    table = contributions.abs().mean().rename("mean_abs_contribution").to_frame()
    table["mean_contribution"] = contributions.mean()
    table["field"] = [FEATURE_LABELS[name] for name in table.index]
    return table.sort_values("mean_abs_contribution", ascending=False).rename_axis("feature").reset_index()


def describe_value(feature: str, value) -> str:
    label = FEATURE_LABELS[feature]
    if feature == "prior_contacts_current_campaign":
        return f"{label}: {int(value)}"
    if isinstance(value, (int, np.integer, float, np.floating)):
        return f"{label}: {value:g}"
    return f"{label}: {value}"


def local_explanation(pipeline: Pipeline, explainer, raw_row: pd.DataFrame) -> pd.DataFrame:
    """Per-field contributions for one lead, with the lead's value and a plain-language sentence."""
    if len(raw_row) != 1:
        raise ValueError("local_explanation expects exactly one row")
    return explanation_table(pipeline, field_contributions(pipeline, explainer, raw_row).iloc[0], raw_row)


def explanation_table(pipeline: Pipeline, contributions: pd.Series, raw_row: pd.DataFrame) -> pd.DataFrame:
    """Turn one lead's per-field contributions into a sorted, plain-language table."""
    features = pipeline.named_steps["features"].transform(raw_row).iloc[0]
    rows = []
    for feature, contribution in contributions.items():
        direction = "higher" if contribution > 0 else "lower"
        rows.append(
            {
                "feature": feature,
                "value": features[feature],
                "description": describe_value(feature, features[feature]),
                "contribution": float(contribution),
                "sentence": f"{describe_value(feature, features[feature])} pushed the model score {direction} "
                f"({contribution:+.2f} on the log-odds scale) relative to an average development-period lead.",
            }
        )
    return pd.DataFrame(rows).sort_values("contribution", key=np.abs, ascending=False, ignore_index=True)
