"""Audit dữ liệu và thí nghiệm minh họa record-level leakage.

Dataset audit là bước bắt buộc trước khi huấn luyện. Naive split chỉ được giữ
như một experiment để minh họa vì nó không phải giao thức đánh giá hợp lệ.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from parkinson_voice.data import (
    ID_COLUMN,
    ORIGINAL_FEATURES,
    SUBJECT_COLUMN,
    TARGET_COLUMN,
    load_data,
)
from parkinson_voice.features import MODEL_FEATURES

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "parkinsons.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def _subject_class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    """Đếm số subject theo từng nhãn, không đếm số recording."""
    return {
        str(label): int(count)
        for label, count in (
            frame.groupby(SUBJECT_COLUMN)[TARGET_COLUMN].first().value_counts().sort_index().items()
        )
    }


def build_data_manifest(frame: pd.DataFrame, data_path: str | Path) -> dict:
    """Tạo manifest bất biến mô tả dataset và feature contract."""
    recordings_per_subject = frame.groupby(SUBJECT_COLUMN).size()
    duplicate_names = int(frame[ID_COLUMN].duplicated().sum())
    duplicate_vectors = int(frame.duplicated(subset=ORIGINAL_FEATURES).sum())

    return {
        "dataset": "UCI Parkinsons",
        "n_recordings": int(len(frame)),
        "n_subjects": int(frame[SUBJECT_COLUMN].nunique()),
        "subject_distribution": _subject_class_distribution(frame),
        "source_features": len(ORIGINAL_FEATURES),
        "model_features": len(MODEL_FEATURES),
        "subject_id_rule": "drop_final_recording_suffix",
        "dropped_features": ["Jitter:DDP", "Shimmer:DDA"],
        "duplicate_recording_names": duplicate_names,
        "duplicate_full_feature_vectors": duplicate_vectors,
        "recordings_per_subject": {
            "min": int(recordings_per_subject.min()),
            "median": float(recordings_per_subject.median()),
            "max": int(recordings_per_subject.max()),
        },
        "evaluation_protocol": "nested-stratified-subject-cv-4x3",
    }


def run_dataset_integrity_audit(
    data_path: str | Path = DATA_PATH,
    artifact_dir: str | Path = ARTIFACT_DIR,
) -> dict:
    """Kiểm tra schema và ghi ``data_manifest.json`` cho lần chạy hiện tại."""
    frame = load_data(data_path)
    manifest = build_data_manifest(frame, data_path)
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def run_naive_split_audit(
    data_path: str | Path = DATA_PATH,
    artifact_dir: str | Path = ARTIFACT_DIR,
) -> dict:
    """Thực hiện naive record-level split audit và xuất kết quả vào thư mục artifact."""
    frame = load_data(data_path)

    X = frame[MODEL_FEATURES]
    y = frame[TARGET_COLUMN]

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        frame.index,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_subjects = set(frame.loc[train_idx, "subject_id"])
    test_subjects = set(frame.loc[test_idx, "subject_id"])
    overlapping = train_subjects.intersection(test_subjects)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = float(model.score(X_test, y_test))

    predictions = frame.loc[test_idx, ["name", "subject_id", TARGET_COLUMN]].copy()
    predictions["is_overlapping_subject"] = predictions["subject_id"].isin(overlapping)

    audit_metrics = {
        "split_unit": "record",
        "test_size": 0.2,
        "random_state": 42,
        "model": "RandomForestClassifier",
        "test_subjects": len(test_subjects),
        "overlapping_test_subjects": len(overlapping),
        "accuracy": accuracy,
    }

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "naive_split_audit.json").write_text(
        json.dumps(audit_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions.to_csv(output_dir / "naive_split_predictions.csv", index=False)

    return audit_metrics


def main() -> None:
    """Điểm vào CLI chạy dataset integrity audit và naive split audit."""
    import argparse

    parser = argparse.ArgumentParser(description="Audit dataset Parkinsons và đối chứng rò rỉ.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Đường dẫn tệp CSV.")
    parser.add_argument("--artifacts", default=str(ARTIFACT_DIR), help="Thư mục xuất artifact.")
    args = parser.parse_args()

    manifest = run_dataset_integrity_audit(args.data, args.artifacts)
    naive = run_naive_split_audit(args.data, args.artifacts)
    print("=== DATA MANIFEST ===")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\n=== NAIVE SPLIT AUDIT ===")
    print(json.dumps(naive, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
