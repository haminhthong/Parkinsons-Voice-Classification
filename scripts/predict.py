"""CLI kiểm tra nhanh một CSV feature table bằng artifact v1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Cho phép chạy trực tiếp ``python scripts/predict.py`` từ thư mục repo.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.settings import ARTIFACT_PATH  # noqa: E402
from parkinson_voice.predict import load_bundle, predict_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV có name và 20 hoặc 22 feature")
    parser.add_argument("--artifact", default=str(ARTIFACT_PATH))
    parser.add_argument(
        "--record-output",
        help="Tùy chọn: đường dẫn CSV để lưu score từng recording.",
    )
    args = parser.parse_args()
    records, subjects = predict_records(
        pd.read_csv(args.csv),
        load_bundle(args.artifact),
    )
    print(subjects.to_json(orient="records", force_ascii=False, indent=2))
    if args.record_output:
        records.to_csv(args.record_output, index=False)


if __name__ == "__main__":
    main()
