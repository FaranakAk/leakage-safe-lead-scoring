"""Model pipelines, the tuning search spaces, and post-hoc calibration.

Every model is a single scikit-learn ``Pipeline`` that takes *raw* rows:

    build_features (stateless, whitelists inputs)
      -> ColumnTransformer (imputation / scaling / one-hot, fitted on training rows only)
      -> estimator

so a forbidden column such as ``duration`` cannot reach the estimator even if
the caller's data contains it, and nothing is fitted outside ``Pipeline.fit``.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.evaluation import probability_metrics
from src.features import PRIMARY, build_features, feature_names

SEED = 42

NAIVE = "naive"
LOGREG = "logreg"
XGBOOST = "xgboost"

# Numeric features that are missing by construction and mean "zero / not applicable"
# when missing; their companion indicator carries the distinction.
ZERO_IMPUTED = ("days_since_prior_contact",)

# --- search spaces and predeclared defaults (kept deliberately small) -----------

LOGREG_CANDIDATES: tuple[dict, ...] = tuple({"C": c} for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0))

# Conservative complexity: shallow trees, sizeable leaves, mild subsampling.
XGB_SEARCH_SPACE = {
    "max_depth": [2, 3],
    "min_child_weight": [5, 20, 50],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [100, 200, 400],
    "reg_lambda": [1.0, 10.0],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}
N_XGB_CANDIDATES = 20

# Used when the inner folds cannot distinguish candidates (see ``select_candidate``).
# Declared before the redesigned search was run.
LOGREG_DEFAULT = {"C": 1.0}  # scikit-learn's default regularisation
XGB_DEFAULT = {
    "colsample_bytree": 0.8,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_child_weight": 20,
    "n_estimators": 100,
    "reg_lambda": 1.0,
    "subsample": 0.8,
}

# A tuned candidate replaces the default only if its per-fold PR-AUC gain over the
# default is, on average, at least this large AND at least two standard errors.
MIN_TUNING_GAIN = 0.005


def xgb_candidates(n: int = N_XGB_CANDIDATES, seed: int = SEED) -> list[dict]:
    """A fixed random sample of the XGBoost search space, plus the predeclared default."""
    sampled = [dict(sorted(p.items())) for p in ParameterSampler(XGB_SEARCH_SPACE, n_iter=n, random_state=seed)]
    default = dict(sorted(XGB_DEFAULT.items()))
    return sampled if default in sampled else [*sampled, default]


def logreg_candidates() -> list[dict]:
    return list(LOGREG_CANDIDATES) if LOGREG_DEFAULT in LOGREG_CANDIDATES else [*LOGREG_CANDIDATES, LOGREG_DEFAULT]


# --- pipelines ---------------------------------------------------------------


def make_preprocessor(feature_set: str = PRIMARY) -> ColumnTransformer:
    numeric = feature_names(feature_set, "numeric")
    zero_imputed = [c for c in numeric if c in ZERO_IMPUTED]
    median_imputed = [c for c in numeric if c not in ZERO_IMPUTED]
    transformers = [
        (
            "numeric",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            median_imputed,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            feature_names(feature_set, "categorical"),
        ),
    ]
    if zero_imputed:
        transformers.append(
            (
                "zero_imputed",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                zero_imputed,
            )
        )
    binary = feature_names(feature_set, "binary")
    if binary:
        transformers.append(("binary", "passthrough", binary))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_estimator(family: str, params: dict | None = None):
    """Unfitted estimator. No class weighting: the Phase 2 experiment showed it hurt ranking
    and distorted probabilities (reports/history/v1_60_20_20/draft_spec.json)."""
    params = dict(params or {})
    if family == NAIVE:
        return DummyClassifier(strategy="prior")
    if family == LOGREG:
        return LogisticRegression(max_iter=5000, random_state=SEED, **params)
    if family == XGBOOST:
        from xgboost import XGBClassifier  # imported lazily so inference does not need xgboost

        return XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            n_jobs=4,
            random_state=SEED,
            **params,
        )
    raise ValueError(f"Unknown model family {family!r}")


def make_pipeline(estimator, feature_set: str = PRIMARY) -> Pipeline:
    return Pipeline(
        [
            ("features", FunctionTransformer(build_features, kw_args={"feature_set": feature_set})),
            ("preprocess", make_preprocessor(feature_set)),
            ("model", estimator),
        ]
    )


def fit_pipeline(raw: pd.DataFrame, y, family: str, params=None, feature_set=PRIMARY) -> Pipeline:
    estimator = make_estimator(family, params)
    return make_pipeline(estimator, feature_set).fit(raw, np.asarray(y))


def positive_scores(model, raw: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(raw)[:, 1]


# --- order-preserving cross-validation -----------------------------------------


def cross_validate(raw, y, family, candidates, folds, feature_set=PRIMARY) -> pd.DataFrame:
    """Score each candidate on expanding-window folds; one row per candidate x fold."""
    y = np.asarray(y)
    rows = []
    for candidate_id, params in enumerate(candidates):
        for fold, (fit_index, eval_index) in enumerate(folds):
            model = fit_pipeline(raw.iloc[fit_index], y[fit_index], family, params, feature_set)
            metrics = probability_metrics(y[eval_index], positive_scores(model, raw.iloc[eval_index]))
            rows.append(
                {
                    "family": family,
                    "feature_set": feature_set,
                    "candidate_id": candidate_id,
                    "params": repr(params),
                    "fold": fold,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarise_cv(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over folds, best mean PR-AUC first."""
    keys = ["family", "feature_set", "candidate_id", "params"]
    summary = cv.groupby(keys).agg(
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        brier_mean=("brier", "mean"),
        mean_predicted_mean=("mean_predicted", "mean"),
        base_rate_mean=("base_rate", "mean"),
    )
    return summary.reset_index().sort_values("pr_auc_mean", ascending=False, ignore_index=True)


def select_candidate(cv: pd.DataFrame, default: dict, min_gain: float = MIN_TUNING_GAIN) -> dict:
    """Pick the best mean-PR-AUC candidate only if it beats the default meaningfully; else keep the default.

    ``cv`` holds one family's per-fold results. The comparison is paired by fold.
    """
    default_key = repr(dict(sorted(default.items())))
    summary = summarise_cv(cv)
    best = summary.iloc[0]
    per_fold = cv.pivot_table(index="fold", columns="params", values="pr_auc")
    gain = per_fold[best["params"]] - per_fold[default_key]
    mean_gain = float(gain.mean())
    standard_error = float(gain.std(ddof=1) / np.sqrt(len(gain))) if len(gain) > 1 else float("inf")
    informative = bool(mean_gain >= min_gain and mean_gain >= 2 * standard_error)
    return {
        "informative": informative,
        "chosen_params": ast.literal_eval(best["params"]) if informative else dict(sorted(default.items())),
        "best_params": ast.literal_eval(best["params"]),
        "best_mean_pr_auc": float(best["pr_auc_mean"]),
        "default_mean_pr_auc": float(summary.loc[summary["params"] == default_key, "pr_auc_mean"].iloc[0]),
        "mean_gain_over_default": mean_gain,
        "gain_standard_error": standard_error,
        "pr_auc_spread_across_candidates": float(summary["pr_auc_mean"].max() - summary["pr_auc_mean"].min()),
    }


# --- primary-model rule (declared before the redesigned validation results) ---------
# Logistic regression (simpler) is primary unless XGBoost is better on all three counts.

XGB_MIN_PR_AUC_GAIN = 0.01
PRIMARY_RULE = (
    "Eligible: validation ROC-AUC > 0.5 and PR-AUC above the naive baseline. "
    f"XGBoost is primary only if its validation PR-AUC exceeds logistic regression's by >= {XGB_MIN_PR_AUC_GAIN}, "
    "its top-20% lift is at least as high, and its PR-AUC is at least as high in every validation month "
    "(temporal stability). Otherwise logistic regression, if eligible."
)


def choose_primary(validation: dict) -> tuple[str | None, dict]:
    """Apply PRIMARY_RULE to the primary feature set's validation results.

    ``validation`` maps family -> metrics dict with ``pr_auc``, ``roc_auc``, ``top_k``
    (records with ``top_fraction`` and ``lift``) and ``pr_auc_by_period``.
    """
    naive_pr = validation[NAIVE]["pr_auc"]
    eligible = {
        family: validation[family]["roc_auc"] > 0.5 and validation[family]["pr_auc"] > naive_pr
        for family in (LOGREG, XGBOOST)
    }
    lr, xgb = validation[LOGREG], validation[XGBOOST]
    lift20 = {f: next(r["lift"] for r in validation[f]["top_k"] if r["top_fraction"] == 0.2) for f in eligible}
    periods = lr["pr_auc_by_period"].keys() & xgb["pr_auc_by_period"].keys()
    checks = {
        "pr_auc_gain_at_least_min": xgb["pr_auc"] - lr["pr_auc"] >= XGB_MIN_PR_AUC_GAIN,
        "top20_lift_at_least_logreg": lift20[XGBOOST] >= lift20[LOGREG],
        "pr_auc_at_least_logreg_every_month": all(
            xgb["pr_auc_by_period"][m] >= lr["pr_auc_by_period"][m] for m in periods
        ),
    }
    if eligible[XGBOOST] and all(checks.values()):
        primary = XGBOOST
    elif eligible[LOGREG]:
        primary = LOGREG
    elif eligible[XGBOOST]:
        primary = XGBOOST
    else:
        primary = None
    return primary, {"eligible": eligible, "xgboost_checks": checks, "top20_lift": lift20}


# --- post-hoc calibration ------------------------------------------------------

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibratedModel:
    """A fitted pipeline plus a logistic (Platt) recalibration of its log-odds.

    With a positive slope the mapping is strictly increasing, so ranking metrics
    are unchanged and only the probability scale moves. Fitted on a later,
    out-of-time segment.
    """

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.calibrator = LogisticRegression(C=np.inf, max_iter=1000)

    def fit(self, raw: pd.DataFrame, y) -> "PlattCalibratedModel":
        self.calibrator.fit(_logit(positive_scores(self.pipeline, raw)), np.asarray(y))
        # A non-positive slope means the score ranking is inverted on this segment;
        # it is reported by callers rather than silently "fixed".
        return self

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict_proba(_logit(positive_scores(self.pipeline, raw)))

    @property
    def slope_intercept(self) -> tuple[float, float]:
        return float(self.calibrator.coef_[0, 0]), float(self.calibrator.intercept_[0])
