"""Acquisition and loading of the UCI Bank Marketing data.

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
License: CC BY 4.0 (Moro, Rita & Cortez, 2014).

Nothing here transforms features; see ``src/features.py`` for that.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
INNER_ZIP_NAME = "bank-additional.zip"
CSV_MEMBER_SUFFIX = "bank-additional-full.csv"
NAMES_MEMBER_SUFFIX = "bank-additional-names.txt"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CSV = RAW_DIR / "bank-additional-full.csv"
RAW_NAMES = RAW_DIR / "bank-additional-names.txt"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _member_ending_with(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [
        name
        for name in archive.namelist()
        if name.endswith(suffix) and not name.startswith("__MACOSX")
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one member ending in {suffix!r}, found {matches}")
    return matches[0]


def download_raw(raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Download the official UCI archive and extract the full CSV and its codebook."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / RAW_CSV.name
    names_path = raw_dir / RAW_NAMES.name
    if csv_path.exists() and names_path.exists() and not force:
        return csv_path

    with urllib.request.urlopen(UCI_ZIP_URL, timeout=120) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(_member_ending_with(outer, INNER_ZIP_NAME))))
    csv_path.write_bytes(inner.read(_member_ending_with(inner, CSV_MEMBER_SUFFIX)))
    names_path.write_bytes(inner.read(_member_ending_with(inner, NAMES_MEMBER_SUFFIX)))
    return csv_path


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV in its original row order.

    The file has no timestamp column; row order is the only temporal signal,
    so the index is kept as the original row position and rows are never shuffled.
    """
    return pd.read_csv(path, sep=";")
