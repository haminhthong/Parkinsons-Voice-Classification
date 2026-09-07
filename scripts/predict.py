"""CLI kiểm tra nhanh một CSV feature table bằng artifact v1."""

from __future__ import annotations

import argparse

import pandas as pd

from app.settings import ARTIFACT_PATH
from src.predict import load_bundle, predict_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV có name và 20 hoặc 22 feature")
    parser.add_argument("--artifact", default=str(ARTIFACT_PATH))
    args = parser.parse_args()
    records, subjects = predict_records(
        pd.read_csv(args.csv),
        load_bundle(args.artifact),
    )
    print(subjects.to_json(orient="records", force_ascii=False, indent=2))
    records.to_csv("prediction_record_scores.csv", index=False)


if __name__ == "__main__":
    main()
