"""Static report figures (matplotlib, light theme).

Palette: validated two-slot categorical (blue, orange) on a near-white surface;
gray is reserved for reference lines (random selection, perfect calibration).
Series are always direct-labelled and legended, so identity never rests on colour alone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")  # slot 1 blue, slot 2 orange

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def _caption(fig, text: str) -> None:
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=8.5, color=INK_SECONDARY, wrap=True)


def gains_curve(y, scores: dict[str, np.ndarray], title: str, caption: str):
    """Cumulative share of conversions captured vs share of leads contacted (highest score first)."""
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--", label="Random selection")
    for (name, s), color in zip(scores.items(), SERIES):
        order = np.argsort(-np.asarray(s), kind="stable")
        captured = np.concatenate([[0], np.cumsum(y[order]) / y.sum()])
        share = np.linspace(0, 1, len(captured))
        ax.plot(share, captured, color=color, linewidth=2, label=name)
        for fraction in (0.1, 0.2, 0.3):
            k = int(round(fraction * len(y)))
            ax.plot(share[k], captured[k], "o", markersize=6, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(name, (share[len(share) // 3], captured[len(share) // 3]), xytext=(6, -14 if color == SERIES[1] else 6),
                    textcoords="offset points", color=INK_SECONDARY, fontsize=9)
    for fraction in (0.1, 0.2, 0.3):
        ax.axvline(fraction, color=GRID, linewidth=1, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of leads contacted (highest score first)")
    ax.set_ylabel("Share of conversions captured")
    ax.set_title(title)
    ax.legend(loc="lower right")
    _caption(fig, caption)
    return fig


def reliability_plot(tables: dict[str, pd.DataFrame], actual_rate: float, title: str, caption: str):
    """Observed conversion rate vs mean predicted probability, per equal-count bin."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    upper = max(max(t["observed_rate"].max(), t["mean_predicted"].max()) for t in tables.values()) * 1.1
    ax.plot([0, upper], [0, upper], color=MUTED, linewidth=1.2, linestyle="--", label="Perfect calibration")
    ax.axhline(actual_rate, color=BASELINE, linewidth=1.2, zorder=0)
    ax.annotate(f"Actual rate {actual_rate:.1%}", (upper, actual_rate), xytext=(-4, 4), textcoords="offset points",
                ha="right", color=INK_SECONDARY, fontsize=9)
    for (name, table), color in zip(tables.items(), SERIES):
        ax.plot(table["mean_predicted"], table["observed_rate"], color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        last = table.iloc[-1]
        ax.annotate(name, (last["mean_predicted"], last["observed_rate"]), xytext=(6, -4), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Mean predicted probability (decile of scores)")
    ax.set_ylabel("Observed conversion rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    _caption(fig, caption)
    return fig


def lift_comparison(lifts: dict[str, list[float]], fractions=(0.1, 0.2, 0.3), title: str = "", caption: str = ""):
    """Grouped bars: lift in the top-scored groups under two evaluation designs."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(fractions))
    width = 0.36
    for i, ((name, values), color) in enumerate(zip(lifts.items(), SERIES)):
        positions = x + (i - 0.5) * (width + 0.02)
        bars = ax.bar(positions, values, width=width, color=color, label=name, edgecolor=SURFACE, linewidth=2)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.2f}×", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                        textcoords="offset points", ha="center", color=INK_SECONDARY, fontsize=9)
    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle="--", label="Random selection (1.0×)")
    ax.set_xticks(x, [f"Top {int(f * 100)}%" for f in fractions])
    ax.set_ylabel("Lift over the segment's own base rate")
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.legend(loc="upper right")
    _caption(fig, caption)
    return fig


def drift_by_period(summary: pd.DataFrame, segment_first_rows: dict[str, int], title: str, caption: str):
    """Monthly conversion rate in row order, with development/validation/test bands.

    ``summary`` has one row per contiguous month run (``first_row``, ``n_rows``); segment
    boundaries that fall inside a month are drawn at the matching fraction of that bar.
    """
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(summary))
    ax.bar(x, summary["conversion_rate"], width=0.7, color=SERIES[0], edgecolor=SURFACE, linewidth=2)

    def position(row: int) -> float:
        run = int(np.searchsorted(summary["first_row"].to_numpy(), row, side="right") - 1)
        fraction = (row - summary["first_row"].iloc[run]) / summary["n_rows"].iloc[run]
        return run - 0.5 + fraction

    ordered = sorted(segment_first_rows.items(), key=lambda kv: kv[1])
    bounds = [position(row) for _, row in ordered] + [len(summary) - 0.5]
    for i, (name, _) in enumerate(ordered):
        start, end = bounds[i], bounds[i + 1]
        if i % 2 == 1:
            ax.axvspan(start, end, color=GRID, alpha=0.6, zorder=0, linewidth=0)
        if i > 0:
            ax.axvline(start, color=MUTED, linewidth=1, linestyle=":", zorder=0)
        ax.annotate(name, ((start + end) / 2 if end - start > 3 else start + 0.1, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -2), textcoords="offset points", va="top", ha="center" if end - start > 3 else "left",
                    rotation=0 if end - start > 3 else 90, color=INK, fontsize=9, fontweight="bold")
    periods = summary["period"].tolist()
    ax.set_xticks(x, [p if p.endswith(("-05", "-11")) else "" for p in periods])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Conversion rate")
    ax.set_xlabel("Month (one bar per month, in file order)")
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    _caption(fig, caption)
    return fig


DIVERGING_POSITIVE = "#2a78d6"  # blue pole: pushed the score higher
DIVERGING_NEGATIVE = "#e34948"  # red pole: pushed the score lower


def band_bars(tables: dict[str, pd.DataFrame], title: str, caption: str):
    """Grouped bars: conversion rate per priority band, one series per segment, with segment averages."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bands = ["High", "Medium", "Low"]
    x = np.arange(len(bands))
    width = 0.36
    top = 0
    for i, ((name, table), color) in enumerate(zip(tables.items(), SERIES)):
        table = table.set_index("band").loc[bands]
        average = table["conversions"].sum() / table["n_leads"].sum()
        positions = x + (i - 0.5) * (width + 0.02)
        bars = ax.bar(positions, table["conversion_rate"], width=width, color=color, edgecolor=SURFACE, linewidth=2,
                      label=f"{name} - average {average:.1%}")
        for bar, rate, share in zip(bars, table["conversion_rate"], table["share_of_leads"]):
            ax.annotate(f"{rate:.1%}\n{share:.0%} of leads", (bar.get_x() + bar.get_width() / 2, rate), xytext=(0, 3),
                        textcoords="offset points", ha="center", color=INK_SECONDARY, fontsize=8.5)
        top = max(top, table["conversion_rate"].max())
    ax.set_xticks(x, [f"{b} priority" for b in bands])
    ax.set_xlim(-0.6, len(bands) - 0.4)
    ax.set_ylim(0, top * 1.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Conversion rate")
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.legend(loc="upper right")
    _caption(fig, caption)
    return fig


def importance_bars(labels: list[str], values: np.ndarray, title: str, caption: str):
    """Horizontal bars, largest at the top, single series."""
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(labels) + 1.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for yi, value in zip(y, values):
        ax.annotate(f"{value:.2f}", (value, yi), xytext=(4, 0), textcoords="offset points", va="center",
                    color=INK_SECONDARY, fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, values.max() * 1.15)
    ax.set_xlabel("Mean |contribution| to ranking score (log-odds)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    _caption(fig, caption)
    return fig


def local_explanation_bars(labels: list[str], values: np.ndarray, title: str, caption: str):
    """Diverging horizontal bars: positive contributions right (blue), negative left (red)."""
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = np.asarray(values)[order]
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(labels) + 1.6))
    y = np.arange(len(labels))
    colors = [DIVERGING_POSITIVE if v > 0 else DIVERGING_NEGATIVE for v in values]
    ax.barh(y, values, height=0.6, color=colors, edgecolor=SURFACE, linewidth=2)
    span = np.abs(values).max()
    for yi, value in zip(y, values):
        ax.annotate(f"{value:+.2f}", (value, yi), xytext=(4 if value >= 0 else -4, 0), textcoords="offset points",
                    va="center", ha="left" if value >= 0 else "right", color=INK_SECONDARY, fontsize=9)
    ax.axvline(0, color=BASELINE, linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_xlabel("Contribution to ranking score (log-odds)")
    ax.grid(axis="y", visible=False)
    handles = [matplotlib.patches.Patch(color=DIVERGING_POSITIVE, label="Pushed score higher"),
               matplotlib.patches.Patch(color=DIVERGING_NEGATIVE, label="Pushed score lower")]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title(title)
    _caption(fig, caption)
    return fig


def feature_audit_graphic(table: pd.DataFrame, labels: dict[str, str], title: str, caption: str):
    """Three cards: features used by the deployed model, kept out on purpose, excluded as leakage."""
    used = table[table["in_primary_model"]]
    held_out = table[table["sensitivity_only"]]
    excluded = table[table["availability_at_scoring"].eq("unavailable")]

    def used_line(row):
        if row["raw_column"] == "campaign":
            return "Earlier calls in this campaign (campaign - 1)"
        return labels.get(row["raw_column"], row["raw_column"])

    reasons = {
        "month": "Call month: period effect, does not generalise",
        "pdays": "Days since earlier contact: only 6 training rows",
    }
    economic = [r for r in held_out["raw_column"] if r not in reasons]
    held_lines = [reasons[r] for r in held_out["raw_column"] if r in reasons]
    if economic:
        held_lines.append(f"{len(economic)} economic indicators: timing undocumented")
    excluded_lines = {
        "duration": "Call duration: known only after the call",
        "y": "Subscribed (the outcome itself)",
    }
    cards = [
        ("Used by the deployed model", f"{len(used)} pre-call fields", [used_line(r) for _, r in used.iterrows()], SERIES[0]),
        ("Available, kept out on purpose", "evaluated only as sensitivity checks", held_lines, MUTED),
        ("Excluded: leakage", "never reach any model", [excluded_lines[r] for r in excluded["raw_column"]], DIVERGING_NEGATIVE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.9))
    for ax, (heading, sub, lines, color) in zip(axes, cards):
        ax.set_axis_off()
        ax.add_patch(
            matplotlib.patches.FancyBboxPatch(
                (0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03", transform=ax.transAxes,
                facecolor=SURFACE, edgecolor=GRID, linewidth=1.2,
            )
        )
        ax.add_patch(matplotlib.patches.Rectangle((0, 0.965), 1, 0.035, transform=ax.transAxes, color=color, linewidth=0))
        ax.text(0.06, 0.9, heading, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=INK, va="top")
        ax.text(0.06, 0.825, sub, transform=ax.transAxes, fontsize=9, color=INK_SECONDARY, va="top")
        for i, line in enumerate(lines):
            ax.text(0.06, 0.73 - i * 0.058, f"•  {line}", transform=ax.transAxes, fontsize=9, color=INK, va="top")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.05, top=0.86)
    _caption(fig, caption)
    return fig
