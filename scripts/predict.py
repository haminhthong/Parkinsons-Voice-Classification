"""CLI kiểm tra nhanh một CSV feature table bằng artifact v1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Cho phép chạy trực tiếp ``python scripts/predict.py`` từ thư mục repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.settings import ARTIFACT_PATH
from src.predict import load_bundle, predict_records


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
