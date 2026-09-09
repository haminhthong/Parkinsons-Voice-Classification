"""Chạy nested subject-level evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phép chạy trực tiếp ``python scripts/evaluate.py`` từ thư mục repo.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parkinson_voice.train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/parkinsons.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    train(args.data, args.artifacts)


if __name__ == "__main__":
    main()
