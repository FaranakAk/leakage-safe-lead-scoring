# Phase 4 results: priority bands, explanations, inference artifact

The frozen Phase 3 logistic regression is unchanged. No model was fitted, refitted or recalibrated in this phase.
Sources: `reports/priority_rule.json`, `reports/bands_validation.csv`, `reports/bands_test_secondary.csv`,
`reports/shap_*.{csv,json}`. Notebook: `notebooks/03_priority_bands_and_explainability.ipynb`.

## Priority-band rule (frozen before any band-level test results were computed)

- **Rule:** High = top 20% of ranking scores, Medium = next 30%, Low = remaining 50%. The bands are rank-based, not
  probability thresholds, because the chronological test showed calibration drift while ranking held up.
- **Ranking score:** the log-odds (decision function) of the uncalibrated logistic regression. The Platt layer is
  monotonic, so ordering is unaffected.
- **Reference distribution:** the validation period (April–May 2009, 4,977 leads).
  - Cutoffs: High if score ≥ −2.5050; Medium if score ≥ −2.7216; otherwise Low. Ties go to the higher band.
  - The full sorted reference scores are stored in the artifact, so a single new lead gets a historical percentile.
    5-point percentiles are also saved in `priority_rule.json`.
- **Batch mode:** the same 20/30/50 rule applied to a batch's own score distribution.

## Validation band results (reference period; 12.8% converted)

| Band | Leads | Share of leads | Conversion rate | Share of conversions | Lift | Contacts per conversion |
|---|---|---|---|---|---|---|
| High | 996 | 20% | 21.0% | 32.9% | 1.64 | 4.8 |
| Medium | 1,493 | 30% | 14.1% | 33.2% | 1.11 | 7.1 |
| Low | 2,488 | 50% | 8.6% | 33.9% | 0.68 | 11.6 |

## Secondary operational characterisation on test (May 2009 – Nov 2010; 30.8% converted)

These figures come from the frozen cutoffs applied once to the saved Phase 3 test scores, with no re-scoring.
They were not used to change anything.

| View | Band | Leads | Share of leads | Conversion rate | Share of conversions | Lift | Contacts per conversion |
|---|---|---|---|---|---|---|---|
| Historical cutoffs | High | 2,594 | 31.5% | 46.6% | 47.6% | 1.51 | 2.1 |
| | Medium | 1,986 | 24.1% | 26.5% | 20.7% | 0.86 | 3.8 |
| | Low | 3,659 | 44.4% | 22.0% | 31.7% | 0.71 | 4.5 |
| Batch-relative 20/30/50 | High | 1,648 | 20.0% | 54.4% | 35.3% | 1.76 | 1.8 |
| | Medium | 2,472 | 30.0% | 29.4% | 28.6% | 0.95 | 3.4 |
| | Low | 4,119 | 50.0% | 22.3% | 36.1% | 0.72 | 4.5 |

- **Band order holds.** Conversion falls from High to Medium to Low in both views.
- **Historical cutoffs let the High band grow.** The score distribution shifted upward, so 31.5% of test leads
  crossed the historical High cutoff instead of 20%.
- **Medium and Low separate weakly on test.** Medium converts at 26.5% and Low at 22.0%. Most of the usable signal
  sits in the High band.

## SHAP findings (ranking model; validation leads explained against a development-period background)

Contributions are measured on the log-odds ranking-score scale and summed back to business fields. They are exactly
additive: the largest reconstruction error was 9e-16. They show model associations ("pushed the score higher or
lower"), not causes of conversion, and they do not describe the calibrated probability.

- **Largest average influence:** contact channel. Cellular pushed scores up by about +0.26 and telephone down by
  about −0.26. It ranks first partly because the channel mix changed: development leads were 51% cellular,
  validation leads 94%.
- **Earlier-campaign history:**
  - A previous success pushed scores up (+0.76 on average), but there were only 130 such validation leads.
  - A previous failure pushed scores down (−0.22).
  - Each additional earlier-campaign contact pushed further down: −0.16 for one, −0.48 for three to five.
- **Job type:** students (+0.94) and retired clients (+0.48) were pushed up; housemaids (−0.37) and unemployed
  clients (−0.30) down.
- **Calls already made in this campaign:** more of them pushed scores down (about −0.19 after more than five).
- **Where the model's associations fall short.**
  - Friday and Wednesday calls were pushed up and Thursday down, a pattern learned from 2008 – early 2009. Yet
    Thursday had the highest validation conversion (18.3%).
  - Clients over 60 were pushed down by the single linear age term, although that small group converted at high
    rates in both development (44% of 39 leads) and validation (39% of 104). This is a limitation of the linear
    form rather than drift.
  - Both are concrete reasons the lift is moderate, and both belong on the list of retraining and modelling
    priorities.
- **Local examples** (chosen by rule: the validation lead closest to each band's median score):
  - **High lead, 90th historical percentile:** pushed up by cellular contact (+0.26), management job (+0.14) and no
    earlier calls this campaign (+0.06); pushed down slightly by a Tuesday call (−0.04).
  - **Low lead, 25th percentile:** pushed down by two earlier-campaign contacts (−0.32) and a previous failure
    (−0.22). Cellular contact (+0.26) and a Wednesday call (+0.10) only partly offset them.

## Deployment artifact

`artifacts/models/lead-scorer-1.0.0.joblib` (about 150 KB), with human-readable metadata in
`artifacts/models/lead-scorer-1.0.0.json`. It contains:

- the frozen preprocessing and logistic-regression pipeline, and the already-fitted Platt calibrator wrapping it;
- the band rule and cutoffs, and the sorted validation reference scores used for percentiles;
- the 12 model feature names (leakage-checked on load);
- a 1,000-row development-period SHAP background, with `duration` and `y` removed;
- metadata: version, development/validation/test periods, calibration parameters, the test-period calibration drift
  (14.7% predicted vs 30.8% actual), the primary/secondary output contract, the calibration warning, data and spec
  hashes, and library versions.

`src/inference.py` provides `load_scorer`, `LeadScorer.score`, `score_batch` and `explain`:

- **Primary outputs:** priority, then ranking score and historical percentile.
- **Secondary output:** `calibrated_probability_secondary`, which carries the drift warning.
- **Batch mode:** adds `batch_percentile` and `batch_priority`.

## Limitations

- Bands describe historical ranking associations, not guaranteed conversion.
- Historical-cutoff band sizes drift with the score distribution. Batch-relative mode keeps the 20/30/50 split but
  depends on the batch's composition.
- The probability is secondary and was badly under-predicted on the test period; it needs recent outcomes before it
  can be trusted.
- SHAP explains a model trained on 2008 – early 2009. Some of its associations no longer held later. Importance is
  relative to the development-period background.
- This full artifact (about 150 KB, including 1,000 development-period rows for SHAP) stays git-ignored. The app
  uses a 51 KB deployment artifact derived from it in Phase 5 (`artifacts/models/lead-scorer-1.0.0-deploy.joblib`),
  which reproduces its scores, percentiles, priorities and explanations exactly and is committed.
- Confusion matrices at the two band operating points (contact High only; contact High + Medium) are in
  `reports/threshold_analysis.csv`, computed from the saved band tables.
