import numpy as np
import pandas as pd

from src import evaluation as ev
from src import plots


def test_figures_build_from_small_inputs():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 200)
    a, b = y + rng.normal(0, 1, 200), rng.normal(size=200)
    assert plots.gains_curve(y, {"A": a, "B": b}, "t", "c") is not None
    table = ev.calibration_table(y, 1 / (1 + np.exp(-a)))
    assert plots.reliability_plot({"A": table}, y.mean(), "t", "c") is not None
    assert plots.lift_comparison({"A": [2, 1.5, 1.2], "B": [3, 2, 1.5]}, title="t", caption="c") is not None
    summary = pd.DataFrame(
        {"period": ["2008-05", "2008-06", "2008-07"], "first_row": [0, 10, 20], "n_rows": [10, 10, 10],
         "conversion_rate": [0.1, 0.2, 0.3]}
    )
    assert plots.drift_by_period(summary, {"development": 0, "validation": 15, "test": 25}, "t", "c") is not None
