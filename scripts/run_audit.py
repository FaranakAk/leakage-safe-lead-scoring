"""Run the data and feature-availability audit and write the results to reports/.

Usage: python scripts/run_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit import contiguous_segment_summary, period_summary, run_audit  # noqa: E402
from src.data import PROJECT_ROOT, RAW_CSV, UCI_ZIP_URL, load_raw, sha256_of  # noqa: E402
from src.features import availability_table  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    raw = load_raw()

    audit = {"source_url": UCI_ZIP_URL, "file": RAW_CSV.name, "sha256": sha256_of(RAW_CSV), **run_audit(raw)}
    (REPORTS_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    availability_table().to_csv(REPORTS_DIR / "feature_availability.csv", index=False)
    period_summary(raw).to_csv(REPORTS_DIR / "period_summary.csv", index=False)
    contiguous_segment_summary(raw).to_csv(REPORTS_DIR / "contiguous_segment_summary.csv", index=False)

    print(json.dumps(audit, indent=2))
    print(contiguous_segment_summary(raw).to_string(index=False))


if __name__ == "__main__":
    main()
