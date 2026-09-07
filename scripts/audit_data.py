"""Chạy dataset integrity audit và ghi data manifest."""

from __future__ import annotations

import argparse

from src.audit import run_dataset_integrity_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/parkinsons.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    print(run_dataset_integrity_audit(args.data, args.artifacts))


if __name__ == "__main__":
    main()
