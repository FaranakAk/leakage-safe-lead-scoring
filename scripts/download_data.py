"""Download the official UCI Bank Marketing data into data/raw/.

Usage: python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import RAW_CSV, UCI_ZIP_URL, download_raw, sha256_of  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()

    path = download_raw(force=args.force)
    print(f"Source : {UCI_ZIP_URL}")
    print(f"Saved  : {path}")
    print(f"Bytes  : {path.stat().st_size}")
    print(f"SHA256 : {sha256_of(path)}")
    assert path == RAW_CSV


if __name__ == "__main__":
    main()
