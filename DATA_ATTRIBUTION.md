# Data attribution and licence

## Source

**Bank Marketing** dataset, UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/222/bank+marketing>

- Creators: Sérgio Moro, Paulo Rita, Paulo Cortez. Donated 13 February 2012.
- DOI: <https://doi.org/10.24432/C5K306>
- Licence: **Creative Commons Attribution 4.0 International (CC BY 4.0)**,
  <https://creativecommons.org/licenses/by/4.0/>, as stated on the official UCI dataset page.
- File used: `bank-additional-full.csv`, from `bank-additional.zip` inside the official archive. It contains 41,188
  rows, 20 inputs plus the outcome, ordered by date from May 2008 to November 2010.

## Required citation

> Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5K306

Introductory paper, as requested in the dataset's codebook (`bank-additional-names.txt`):

> S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision
> Support Systems (2014). https://doi.org/10.1016/j.dss.2014.03.001

## What this repository contains from the dataset

The raw dataset is **not** committed. `scripts/download_data.py` downloads it from the official archive into
`data/raw/`, which is git-ignored.

Derived material that is committed, with the changes made:

| File | Content | Changes |
|---|---|---|
| `app_assets/example_leads.csv` | 200 leads from the April–May 2009 rows | 12 inputs only; outcome, call duration, month, `pdays` and economic indicators removed; `campaign` converted to earlier calls (`campaign − 1`); columns renamed |
| `reports/test_scores.csv` | outcome and model scores for the 8,239 test-period rows | no input fields; model scores added |
| `reports/*.csv`, `reports/*.json`, `reports/figures/` | aggregate statistics, metrics and charts | derived summaries |
| `artifacts/models/lead-scorer-1.0.0-deploy.*` | a model trained on the data | contains no raw rows; reference score distribution and summary statistics only |

Anyone reusing these files must credit the dataset creators as above, as CC BY 4.0 requires. The original data
was created and published by the authors above. They do not endorse this project.

## Scope of this licence

CC BY 4.0 applies to the dataset and to the data-derived files listed above. The repository's source code is
licensed separately under the MIT License (see `LICENSE`). The MIT License does **not** apply to the dataset or to
these data-derived files.
