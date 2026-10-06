# Phase 2–3 results: validation design, baseline, boosted model, one-time test

All figures below are read from saved outputs: `reports/frozen_spec.json`, `reports/validation_metrics.json`,
`reports/metrics.json` (one-time test) and `reports/random_split_diagnostic.json`. Notebook:
`notebooks/02_modeling_and_evaluation.ipynb`. Figures: `reports/figures/`.

## Design (frozen before the test segment was scored)

- **Decision point:** a lead is scored immediately before the recorded outbound call is dialled.
- **Segments (row order; rows are in date order):**

  | Segment | Rows | Period | Conversion rate |
  |---|---|---|---|
  | Development | 0–27,971 (27,972) | May 2008 – Mar 2009 | 5.24% |
  | Validation | 27,972–32,948 (4,977) | Apr – May 2009 | 12.76% |
  | Test (untouched until the single scoring run) | 32,949–41,187 (8,239) | May 2009 – Nov 2010 | 30.83% |

  The test boundary has been fixed since Phase 1. Validation starts at the month boundary nearest 70% of rows.
  **May 2009 is split between validation and test**: the segments are chronological and non-overlapping but not
  fully month-disjoint.
- **Primary features (12):** age, job, marital, education, default, housing, loan, contact, day_of_week,
  prior_contacts_current_campaign (= campaign − 1), previous, poutcome. `duration` is excluded everywhere.
  `month`, the pdays-derived features and the economic indicators are kept for validation-only sensitivity
  analyses.
- **Models:** naive constant-rate baseline; logistic regression (C = 1.0); XGBoost (depth 3, min child weight 20,
  learning rate 0.1, 100 trees, subsample and column fraction 0.8). No class weighting, no resampling.
- **Tuning** used expanding-window folds inside the development period. It was **not informative**: every candidate
  scored at noise level, within about 0.005 PR-AUC of the naive baseline. So the predeclared defaults were used, not
  the noisy best candidates.
- **Primary model:** logistic regression. Under the predeclared rule, XGBoost would have needed a PR-AUC gain of at
  least 0.01, at least equal top-20% lift and at least equal PR-AUC in every validation month. It failed the
  monthly-stability check (May 2009: 0.111 vs 0.121).
- **Calibration:** Platt scaling on the model's log-odds, fitted on validation (slope 1.258, intercept 1.456),
  applied unchanged to test.
- **Priority bands:** not defined (Phase 4).

## Validation (Apr – May 2009, out-of-time, before calibration)

| Model | PR-AUC | ROC-AUC | Brier | Top 10% lift | Top 20% lift | Top 30% lift |
|---|---|---|---|---|---|---|
| Naive | 0.128 | 0.500 | 0.117 | 1.00 | 1.00 | 1.00 |
| Logistic regression | 0.200 | 0.627 | 0.114 | 2.11 | 1.65 | 1.51 |
| XGBoost | 0.213 | 0.616 | 0.114 | 2.19 | 1.66 | 1.48 |

## Headline: one-time chronological test (May 2009 – Nov 2010; 8,239 leads; 30.83% converted)

### Ranking

| Model | PR-AUC [95% interval] | ROC-AUC [95% interval] |
|---|---|---|
| Naive | 0.308 | 0.500 |
| **Logistic regression (primary)** | **0.459** [0.440, 0.480] | **0.645** [0.632, 0.658] |
| XGBoost (comparison) | 0.391 [0.374, 0.410] | 0.588 [0.575, 0.601] |
| Logistic minus XGBoost (same resamples) | +0.051 to +0.085 | +0.045 to +0.068 |

### Business metrics in the top-scored groups

| Model | Group | Leads contacted | Conversions | Precision | Share of conversions captured | Lift | Contacts per conversion |
|---|---|---|---|---|---|---|---|
| Naive (random order) | top 10 / 20 / 30% | 824 / 1,648 / 2,472 | 254 / 508 / 762 (expected) | 30.8% | 10 / 20 / 30% | 1.00 | 3.24 |
| **Logistic regression** | top 10% | 824 | 456 | 55.3% | 18.0% | 1.80 | 1.81 |
| | top 20% | 1,648 | 896 | 54.4% | 35.3% | 1.76 | 1.84 |
| | top 30% | 2,472 | 1,166 | 47.2% | 45.9% | 1.53 | 2.12 |
| XGBoost | top 10% | 824 | 398 | 48.3% | 15.7% | 1.57 | 2.07 |
| | top 20% | 1,648 | 695 | 42.2% | 27.4% | 1.37 | 2.37 |
| | top 30% | 2,472 | 953 | 38.6% | 37.5% | 1.25 | 2.59 |

### Calibration (logistic regression)

| | Before calibration | After Platt calibration |
|---|---|---|
| Brier score | 0.262 | 0.228 |
| Mean predicted probability | 7.3% | 14.7% |
| Actual test conversion rate | 30.8% | 30.8% |

For reference, the Brier score of always predicting the test period's own rate is 0.213. That benchmark is not
achievable in practice, because the rate is unknown in advance. Calibration lowers the Brier score but leaves
probabilities far too low: in the top decile it predicts 34% against 55% observed, and in the bottom decile 6%
against 22%. This is **temporal calibration drift**. The mapping was fitted on a period converting at 12.8% and
applied to one converting at 30.8%, and it was deliberately not refitted on test. Calibration does not change the
ranking.

XGBoost has no calibration mapping (it was not the primary model). Its uncalibrated Brier score is 0.267, with a
mean prediction of 7.0%.

### Uncertainty

The intervals come from a nonparametric row bootstrap: 2,000 resamples, seed 20261006, 95% percentile intervals,
with the same resamples used for both models. They describe row-level sampling uncertainty under an independence
approximation. They do **not** capture temporal dependence, distribution drift, or repeated clients; no client ID is
available.

## Optimistic diagnostic only: stratified random split (not a headline, not used for any decision)

The same frozen recipe was run on a stratified random 70/10/20 split of all rows, so training and evaluation share
months. The evaluation set has 8,238 leads with an 11.26% conversion rate.

| Model | PR-AUC [95%] | ROC-AUC [95%] | Brier | Top 10% lift | Top 20% lift | Top 30% lift |
|---|---|---|---|---|---|---|
| Naive | 0.113 | 0.500 | 0.100 | 1.00 | 1.00 | 1.00 |
| Logistic regression | 0.370 [0.337, 0.403] | 0.742 [0.723, 0.758] | 0.085 before / 0.085 after | 3.50 | 2.40 | 2.00 |
| XGBoost | 0.391 [0.358, 0.425] | 0.746 [0.727, 0.763] | 0.084 | 3.78 | 2.48 | 2.04 |

Absolute PR-AUC is not comparable across the two designs because the base rates differ; lift, ROC-AUC and
calibration are comparable.

- **Lift:** under a random split, top-10% lift roughly doubles (3.50 vs 1.80) and ROC-AUC rises from 0.645 to 0.742.
- **Calibration:** probabilities look well calibrated (mean prediction 11.5% vs 11.3% actual).
- **Model ranking flips:** XGBoost edges out logistic regression on PR-AUC (difference interval −0.031 to −0.010).

A random split would therefore have overstated performance and favoured the model that generalised worse over time.

## What this means for a sales team (no dollar ROI claimed)

- **Contacting the top 20%** of scored leads in the test period would have reached 35% of conversions, at 1.84
  contacts per conversion. Random selection over the same number of leads gives 20% of conversions at 3.24 contacts
  per conversion. This is historical backtesting, not a guarantee.
- **Use the scores for ordering, not as probabilities.** The scores ranked leads usefully a year after the training
  data ended. As probabilities they were far too low, because the overall conversion rate shifted. Any use of them
  as absolute probabilities needs recent data for recalibration.
- **The gain over random is moderate** (lift about 1.5–1.8). Signal in the early training period was weak, and
  relationships changed over time (e.g. prior-campaign outcome, age extremes, students and retired clients).

## Limitations

- One Portuguese bank, 2008–2010, during the financial crisis. A backtest, not a forecast for another business.
- Strong drift in base rate, contact channel and prior-campaign history across segments. The development period
  contains only 38 prior successes (`poutcome = success`).
- May 2009 is split between validation and test.
- Rows are final contacts per client and campaign; there is no client ID, so a client may appear in several periods.
- The models were never refitted on later data. A production system would retrain and recalibrate on recent outcomes.
- Bootstrap intervals assume independent rows.
- Priority bands are not yet defined (Phase 4).

## Redesign history (before any test scoring)

The first design (60/20/20, `month` in the primary set, tuning by picking the best CV score, a class-weighting
experiment) is archived in `reports/history/v1_60_20_20/`.

- **Uninformative tuning:** inner CV on 2008-only data could not tell candidates apart, and the selected XGBoost
  ranked validation leads in reverse.
- **Class weighting** hurt ranking and distorted probabilities.
- **`month` artefact:** logistic regression's validation advantage came mainly from `month` levels that never
  appeared in training.
- **Shallow-XGBoost observation:** a shallow XGBoost looked better on the first validation period. That observation
  motivated the conservative revised search space. It was not used to pick a configuration.
