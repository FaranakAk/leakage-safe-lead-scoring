"""Versioned inference artifact: priority band, ranking score / percentile, optional probability.

Primary outputs, in order:
1. priority band (High / Medium / Low) from a fixed rank rule;
2. ranking score (log-odds of the frozen logistic regression) and its historical percentile.

The calibrated probability is a secondary output with a drift warning: the Platt
mapping was fitted on April-May 2009 and under-predicted badly on the later test
period. Nothing here fits, refits or recalibrates anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from src import explainability as xp
from src.features import FORBIDDEN_COLUMNS, assert_no_leakage

ARTIFACT_VERSION = "lead-scorer-1.0.0"

# Frozen operational rule (Phase 4): rank-based, not probability-based.
BAND_RULE = {"High": 0.20, "Medium": 0.30, "Low": 0.50}  # shares of leads, highest scores first
BANDS = ("High", "Medium", "Low")

CALIBRATION_WARNING = (
    "Secondary output. The probability calibration was fitted on April-May 2009 (12.8% conversion) and, on the "
    "later test period (30.8% conversion), predicted 14.7% on average. Calibration drifts when the overall "
    "conversion rate changes; it needs recent outcome data before these probabilities can be relied on. Use the "
    "priority band and ranking percentile for decisions."
)


def band_cutoffs(reference_scores: np.ndarray, rule: dict = BAND_RULE) -> dict[str, float]:
    """Score cutoffs so that, on the reference distribution, the top rule['High'] share is High, etc."""
    scores = np.asarray(reference_scores, dtype=float)
    high_quantile = 1 - rule["High"]
    medium_quantile = 1 - rule["High"] - rule["Medium"]
    return {"high_min": float(np.quantile(scores, high_quantile)), "medium_min": float(np.quantile(scores, medium_quantile))}


def assign_band(scores, cutoffs: dict[str, float]) -> np.ndarray:
    """High if score >= high_min; Medium if score >= medium_min; otherwise Low. Ties go to the higher band."""
    scores = np.asarray(scores, dtype=float)
    return np.where(scores >= cutoffs["high_min"], "High", np.where(scores >= cutoffs["medium_min"], "Medium", "Low"))


def percentile_of(scores, sorted_reference: np.ndarray) -> np.ndarray:
    """Share of reference scores at or below each score, as 0-100."""
    positions = np.searchsorted(sorted_reference, np.asarray(scores, dtype=float), side="right")
    return 100.0 * positions / len(sorted_reference)


class _Scoring:
    """Scoring shared by the full (Phase 4) and the minimal deployment artifacts."""

    def ranking_score(self, raw: pd.DataFrame) -> np.ndarray:
        return self.pipeline.decision_function(raw)

    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Score leads against the historical reference distribution."""
        scores = self.ranking_score(raw)
        return pd.DataFrame(
            {
                "priority": assign_band(scores, self.cutoffs),
                "ranking_score": scores,
                "historical_percentile": percentile_of(scores, self.reference_scores),
                "calibrated_probability_secondary": self.calibrator.predict_proba(raw)[:, 1],
            },
            index=raw.index,
        )

    def score_batch(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Reference-based outputs plus priority and percentile relative to this batch.

        Batch bands use the same 20/30/50 rule on the batch's own scores (cutoffs at the
        batch's 80th and 50th percentiles, linear interpolation; ties go to the higher band).
        With distinct scores, each band is within one lead of its target share.
        """
        result = self.score(raw)
        batch_scores = np.sort(result["ranking_score"].to_numpy())
        result["batch_percentile"] = percentile_of(result["ranking_score"], batch_scores)
        result["batch_priority"] = assign_band(result["ranking_score"], band_cutoffs(batch_scores))
        return result


@dataclass
class LeadScorer(_Scoring):
    """Frozen pipeline + Platt calibrator + band rule + reference distribution + metadata."""

    pipeline: object  # preprocessing + logistic regression (uncalibrated ranking model)
    calibrator: object  # PlattCalibratedModel wrapping the same pipeline
    reference_scores: np.ndarray  # sorted ranking scores of the reference (validation) period
    cutoffs: dict
    feature_names: list
    shap_background: pd.DataFrame  # development-period raw rows (reference for explanations)
    metadata: dict = field(default_factory=dict)

    def explainer(self):
        """SHAP explainer built from the stored development-period background. Does not touch the model."""
        if not hasattr(self, "_explainer"):
            self._explainer = xp.make_explainer(self.pipeline, self.shap_background)
        return self._explainer

    def explain(self, raw_row: pd.DataFrame) -> pd.DataFrame:
        return xp.local_explanation(self.pipeline, self.explainer(), raw_row)


def build_scorer(calibrated_model, reference_raw: pd.DataFrame, background_raw: pd.DataFrame, metadata: dict) -> LeadScorer:
    """Assemble the artifact from already-fitted objects. No fitting happens here."""
    pipeline = calibrated_model.pipeline
    feature_names = list(pipeline.named_steps["preprocess"].feature_names_in_)
    assert_no_leakage(feature_names)
    reference = np.sort(pipeline.decision_function(reference_raw))
    cutoffs = band_cutoffs(reference)
    return LeadScorer(
        pipeline=pipeline,
        calibrator=calibrated_model,
        reference_scores=reference,
        cutoffs=cutoffs,
        feature_names=feature_names,
        shap_background=background_raw.drop(columns=[c for c in FORBIDDEN_COLUMNS if c in background_raw]),
        metadata={
            "version": ARTIFACT_VERSION,
            "band_rule": BAND_RULE,
            "cutoffs": cutoffs,
            "primary_outputs": ["priority", "ranking_score / historical_percentile"],
            "secondary_outputs": ["calibrated_probability_secondary"],
            "calibration_warning": CALIBRATION_WARNING,
            **metadata,
        },
    )


DEPLOY_VERSION = f"{ARTIFACT_VERSION}-deploy"


@dataclass
class DeploymentScorer(_Scoring):
    """Minimal artifact for the app: same frozen objects, no raw background rows, no shap/xgboost needed.

    Explanations use the closed-form linear contributions coef * (x - background mean), with the
    background mean taken from the Phase 4 development-period SHAP background.
    """

    pipeline: object
    calibrator: object
    reference_scores: np.ndarray
    cutoffs: dict
    feature_names: list
    explanation_mean: np.ndarray  # mean of the transformed development-period background
    explanation_base_value: float
    input_schema: dict  # valid categories, numeric ranges and consistency rules for app inputs
    metadata: dict = field(default_factory=dict)

    def contributions(self, raw: pd.DataFrame) -> pd.DataFrame:
        return xp.linear_field_contributions(self.pipeline, self.explanation_mean, raw)

    def explain(self, raw_row: pd.DataFrame) -> pd.DataFrame:
        if len(raw_row) != 1:
            raise ValueError("explain expects exactly one row")
        return xp.explanation_table(self.pipeline, self.contributions(raw_row).iloc[0], raw_row)


def build_deployment_scorer(scorer: LeadScorer, input_schema: dict, metadata: dict) -> DeploymentScorer:
    """Derive the minimal artifact from the frozen Phase 4 artifact. Nothing is fitted."""
    mean = xp.background_mean(scorer.pipeline, scorer.shap_background)
    return DeploymentScorer(
        pipeline=scorer.pipeline,
        calibrator=scorer.calibrator,
        reference_scores=scorer.reference_scores,
        cutoffs=dict(scorer.cutoffs),
        feature_names=list(scorer.feature_names),
        explanation_mean=mean,
        explanation_base_value=xp.linear_base_value(scorer.pipeline, mean),
        input_schema=input_schema,
        metadata={**scorer.metadata, "version": DEPLOY_VERSION, "derived_from": ARTIFACT_VERSION, **metadata},
    )


def load_deployment_scorer(path) -> DeploymentScorer:
    scorer = joblib.load(path)
    if not isinstance(scorer, DeploymentScorer):
        raise TypeError(f"{path} does not contain a DeploymentScorer")
    if scorer.metadata.get("version") != DEPLOY_VERSION:
        raise ValueError(f"Artifact version {scorer.metadata.get('version')} != {DEPLOY_VERSION}")
    assert_no_leakage(scorer.feature_names)
    return scorer


def save_scorer(scorer: LeadScorer, path) -> None:
    joblib.dump(scorer, path)


def load_scorer(path) -> LeadScorer:
    scorer = joblib.load(path)
    if not isinstance(scorer, LeadScorer):
        raise TypeError(f"{path} does not contain a LeadScorer")
    if scorer.metadata.get("version") != ARTIFACT_VERSION:
        raise ValueError(f"Artifact version {scorer.metadata.get('version')} != {ARTIFACT_VERSION}")
    assert_no_leakage(scorer.feature_names)
    return scorer
