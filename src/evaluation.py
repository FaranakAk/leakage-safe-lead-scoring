"""Ranking, probability-quality and business (top-k) metrics.

Ranking (PR-AUC, ROC-AUC, top-k) and calibration (Brier, reliability) are kept
separate on purpose: the base conversion rate changes strongly over time, so a
model can rank leads well while its probabilities are badly off-scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

TOP_FRACTIONS = (0.1, 0.2, 0.3)


def probability_metrics(y, p) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        # Brier score of always predicting this segment's own base rate (not
        # achievable in practice, since the rate is unknown in advance).
        "brier_at_segment_base_rate": float(y.mean() * (1 - y.mean())),
    }


def expected_hits_in_top(y, p, k: int) -> float:
    """Expected conversions among the k highest scores, ties at the cut-off broken at random."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if not 0 < k <= len(y):
        raise ValueError("k must be between 1 and the number of rows")
    cutoff = np.sort(p)[::-1][k - 1]
    above = p > cutoff
    tied = p == cutoff
    n_from_ties = k - above.sum()
    return float(y[above].sum() + n_from_ties * y[tied].mean())


def top_k_table(y, p, fractions=TOP_FRACTIONS) -> pd.DataFrame:
    """Precision, conversions captured, lift and contacts per conversion in the top-scored groups.

    Lift is relative to this segment's own base rate (random selection within the segment).
    """
    y = np.asarray(y)
    base_rate = y.mean()
    rows = []
    for fraction in fractions:
        k = int(round(fraction * len(y)))
        hits = expected_hits_in_top(y, p, k)
        precision = hits / k
        rows.append(
            {
                "top_fraction": fraction,
                "n_contacted": k,
                "expected_conversions": hits,
                "precision": precision,
                "capture_rate": hits / y.sum(),
                "lift": precision / base_rate,
                "contacts_per_conversion": k / hits if hits > 0 else np.inf,
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table on equal-count bins of predicted probability."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False)
    table = frame.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    return table.reset_index()


# Frozen before the test segment was scored.
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20261006
BOOTSTRAP_LEVEL = 0.95
BOOTSTRAP_CAVEAT = (
    "Nonparametric row bootstrap (rows resampled with replacement, same resamples for every model). "
    "Intervals describe row-level sampling uncertainty under an independence approximation. They do not "
    "capture temporal dependence, distribution drift, or repeated clients (no client ID is available)."
)


def paired_bootstrap(
    y,
    scores: dict[str, np.ndarray],
    metrics: dict,
    differences: tuple[tuple[str, str], ...] = (),
    n_boot: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    level: float = BOOTSTRAP_LEVEL,
) -> dict:
    """Percentile intervals for each metric and model, using one shared set of row resamples.

    ``differences`` lists (model_a, model_b) pairs whose metric difference a - b is also
    interval-estimated on the same resamples.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    draws = {(m, name): [] for m in scores for name in metrics}
    diff_draws = {(a, b, name): [] for a, b in differences for name in metrics}
    for _ in range(n_boot):
        index = rng.integers(0, len(y), len(y))
        if y[index].min() == y[index].max():
            continue
        values = {(m, name): fn(y[index], np.asarray(s)[index]) for m, s in scores.items() for name, fn in metrics.items()}
        for key, value in values.items():
            draws[key].append(value)
        for a, b, name in diff_draws:
            diff_draws[(a, b, name)].append(values[(a, name)] - values[(b, name)])
    tail = (1 - level) / 2

    def interval(values):
        return [float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))]

    result = {m: {name: interval(draws[(m, name)]) for name in metrics} for m in scores}
    result["differences"] = {f"{a} - {b}": {name: interval(diff_draws[(a, b, name)]) for name in metrics} for a, b in differences}
    result["settings"] = {"repetitions": n_boot, "seed": seed, "level": level, "caveat": BOOTSTRAP_CAVEAT}
    return result


def band_table(y, bands, order=("High", "Medium", "Low")) -> pd.DataFrame:
    """Leads, conversion rate, share of conversions, lift and contacts per conversion for each priority band."""
    frame = pd.DataFrame({"y": np.asarray(y), "band": np.asarray(bands)})
    base_rate = frame["y"].mean()
    rows = []
    for band in order:
        members = frame.loc[frame["band"] == band, "y"]
        conversions = int(members.sum())
        rate = float(members.mean()) if len(members) else float("nan")
        rows.append(
            {
                "band": band,
                "n_leads": int(len(members)),
                "share_of_leads": len(members) / len(frame),
                "conversions": conversions,
                "conversion_rate": rate,
                "share_of_conversions": conversions / frame["y"].sum(),
                "lift": rate / base_rate,
                "contacts_per_conversion": len(members) / conversions if conversions else float("inf"),
            }
        )
    return pd.DataFrame(rows)


def operating_point_confusion(bands: pd.DataFrame) -> pd.DataFrame:
    """Confusion matrices for 'contact High only' and 'contact High + Medium', from a band_table.

    Pure arithmetic on band counts: positives are conversions; a lead is selected if its band is contacted.
    """
    table = bands.set_index("band")
    total_leads = int(table["n_leads"].sum())
    total_conversions = int(table["conversions"].sum())
    rows = []
    for name, selected in (("contact High only", ["High"]), ("contact High + Medium", ["High", "Medium"])):
        tp = int(table.loc[selected, "conversions"].sum())
        contacted = int(table.loc[selected, "n_leads"].sum())
        fp = contacted - tp
        fn = total_conversions - tp
        tn = total_leads - contacted - fn
        rows.append(
            {
                "operating_point": name,
                "contacted": contacted,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "precision": tp / contacted,
                "recall": tp / total_conversions,
                "false_positive_rate": fp / (fp + tn),
            }
        )
    return pd.DataFrame(rows)


def subgroup_table(groups: pd.Series, y, p) -> pd.DataFrame:
    """Observed vs mean predicted conversion within each level of a grouping column."""
    frame = pd.DataFrame({"group": np.asarray(groups), "y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    table = frame.groupby("group").agg(n=("y", "size"), observed_rate=("y", "mean"), mean_predicted=("p", "mean"))
    table["share_of_conversions"] = frame.groupby("group")["y"].sum() / frame["y"].sum()
    return table.reset_index().rename(columns={"group": groups.name or "group"})


def unseen_levels(train: pd.DataFrame, other: pd.DataFrame, columns) -> dict:
    """Category levels present in ``other`` but absent from ``train``, with row counts."""
    result = {}
    for column in columns:
        unseen = sorted(set(other[column]) - set(train[column]))
        if unseen:
            counts = other[column].value_counts()
            result[column] = {level: int(counts[level]) for level in unseen}
    return result
