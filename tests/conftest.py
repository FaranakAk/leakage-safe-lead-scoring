import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import RAW_CSV, load_raw  # noqa: E402


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """Small synthetic frame with the raw UCI schema, including forbidden columns."""
    return pd.DataFrame(
        {
            "age": [30, 45, 58],
            "job": ["admin.", "blue-collar", "unknown"],
            "marital": ["single", "married", "divorced"],
            "education": ["university.degree", "basic.9y", "unknown"],
            "default": ["no", "unknown", "no"],
            "housing": ["yes", "no", "unknown"],
            "loan": ["no", "no", "unknown"],
            "contact": ["cellular", "telephone", "cellular"],
            "month": ["may", "jun", "nov"],
            "day_of_week": ["mon", "tue", "fri"],
            "duration": [120, 0, 900],
            "campaign": [1, 4, 2],
            "pdays": [999, 999, 0],
            "previous": [0, 1, 2],
            "poutcome": ["nonexistent", "failure", "success"],
            "emp.var.rate": [1.1, 1.4, -1.1],
            "cons.price.idx": [93.994, 94.465, 94.767],
            "cons.conf.idx": [-36.4, -41.8, -50.8],
            "euribor3m": [4.857, 4.961, 1.028],
            "nr.employed": [5191.0, 5228.1, 4963.6],
            "y": ["no", "no", "yes"],
        }
    )


@pytest.fixture
def raw_train(raw_sample) -> pd.DataFrame:
    """Repeat the synthetic rows with varied age and a mixed target so models can fit."""
    raw = pd.concat([raw_sample] * 20, ignore_index=True)
    raw["age"] = np.arange(len(raw)) % 50 + 20
    raw["y"] = np.where(np.arange(len(raw)) % 4 == 0, "yes", "no")
    return raw


@pytest.fixture(scope="session")
def raw_full() -> pd.DataFrame:
    if not RAW_CSV.exists():
        pytest.skip("Raw data not downloaded; run scripts/download_data.py")
    return load_raw()
