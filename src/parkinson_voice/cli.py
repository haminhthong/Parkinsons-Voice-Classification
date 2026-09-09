"""CLI mỏng cho các thao tác reproducible của pipeline."""

from __future__ import annotations

import argparse

from parkinson_voice.audit import run_dataset_integrity_audit
from parkinson_voice.train import train


def audit() -> None:
    """Ghi manifest audit cho dataset mặc định."""
    parser = argparse.ArgumentParser(description="Audit dataset Parkinsons.")
    parser.add_argument("--data", default="data/parkinsons.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    print(run_dataset_integrity_audit(args.data, args.artifacts))


def evaluate() -> None:
    """Chạy nested CV và fit artifact deployment."""
    parser = argparse.ArgumentParser(description="Train/evaluate Parkinson screening model.")
    parser.add_argument("--data", default="data/parkinsons.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    print(train(args.data, args.artifacts).to_string(index=False))
