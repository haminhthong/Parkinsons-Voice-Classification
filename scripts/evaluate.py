"""Chạy nested subject-level evaluation."""

from __future__ import annotations

import argparse

from src.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/parkinsons.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    train(args.data, args.artifacts)


if __name__ == "__main__":
    main()
