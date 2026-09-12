"""Suy luận subject-level từ acoustic feature table.

Recording chỉ nhận một screening score. Decision threshold chỉ được áp dụng
sau khi lấy median các recording score của cùng subject.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from parkinson_voice.data import ID_COLUMN, SUBJECT_COLUMN, validate_dataframe
from parkinson_voice.features import MODEL_FEATURES, ORIGINAL_FEATURES
from parkinson_voice.utils import positive_class_probability

DANGEROUS_CSV_PREFIXES = ("=", "+", "-", "@")


def sanitize_csv_value(value: str) -> str:
    """Bảo vệ giá trị text khi xuất CSV khỏi formula injection."""
    value = str(value)
    return "'" + value if value.startswith(DANGEROUS_CSV_PREFIXES) else value


def load_bundle(path: str | Path) -> dict:
    """Nạp artifact mô hình và kiểm tra các trường cơ bản."""
    filepath = Path(path)
    if not filepath.is_file():
        raise RuntimeError(f"Không tìm thấy model artifact tại: {filepath}")

    bundle = joblib.load(filepath)
    required = {
        "model",
        "feature_columns",
        "decision_threshold",
        "aggregation",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Artifact thiếu trường bắt buộc: {sorted(missing)}")
    if list(bundle["feature_columns"]) != MODEL_FEATURES:
        raise ValueError("Danh sách đặc trưng trong artifact không khớp MODEL_FEATURES.")
    if bundle["aggregation"] != "median":
        raise ValueError("Quy tắc gộp phải là 'median'.")
    threshold = float(bundle["decision_threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Artifact có decision threshold không hợp lệ.")
    return bundle


def _prepare_inference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa input 20/22 feature và tạo feature dẫn xuất nếu cần."""
    if frame.empty:
        raise ValueError("Dữ liệu suy luận không có bản ghi.")
    if "status" in frame.columns:
        raise ValueError("Input suy luận không được chứa nhãn huấn luyện 'status'.")
    if ID_COLUMN not in frame.columns:
        raise ValueError("Input suy luận bắt buộc có cột 'name'.")

    allowed = set(ORIGINAL_FEATURES) | {ID_COLUMN, SUBJECT_COLUMN}
    unknown = sorted(set(frame.columns).difference(allowed))
    if unknown:
        raise ValueError(f"Input chứa cột không thuộc feature contract: {unknown}")
    missing_model = sorted(set(MODEL_FEATURES).difference(frame.columns))
    if missing_model:
        raise ValueError(f"Thiếu các feature mô hình bắt buộc: {missing_model}")

    prepared = frame.copy()
    if "Jitter:DDP" not in prepared:
        prepared["Jitter:DDP"] = prepared["MDVP:RAP"] * 3.0
    if "Shimmer:DDA" not in prepared:
        prepared["Shimmer:DDA"] = prepared["Shimmer:APQ3"] * 3.0

    validated = validate_dataframe(prepared, require_target=False, require_name=True)
    redundant_relations = {
        "Jitter:DDP": ("MDVP:RAP", 3.0),
        "Shimmer:DDA": ("Shimmer:APQ3", 3.0),
    }
    for derived, (base, multiplier) in redundant_relations.items():
        if derived in frame.columns and not np.allclose(
            validated[derived].to_numpy(dtype=float),
            validated[base].to_numpy(dtype=float) * multiplier,
            # Dataset nguồn lưu các feature dẫn xuất đã làm tròn đến 5 chữ số.
            rtol=1e-4,
            atol=2e-5,
        ):
            raise ValueError(
                f"Feature dẫn xuất {derived!r} không khớp quan hệ đại số với {base!r}."
            )
    supplied_subjects = frame.get(SUBJECT_COLUMN)
    if supplied_subjects is not None:
        expected = validated[SUBJECT_COLUMN].astype(str).reset_index(drop=True)
        supplied = supplied_subjects.astype(str).reset_index(drop=True)
        if not supplied.equals(expected):
            raise ValueError("subject_id trong input không khớp quy tắc suy ra từ name.")
    return validated


def _training_ranges(bundle: dict) -> dict[str, tuple[float, float]]:
    """Đọc dải giá trị huấn luyện (P1-P99) để cảnh báo plausibility."""
    ranges = bundle.get("feature_ranges") or bundle.get("training_feature_ranges", {})
    return {str(column): (float(values[0]), float(values[1])) for column, values in ranges.items()}


def check_recording_training_range(
    row: pd.Series,
    feature_ranges: dict[str, tuple[float, float]] | None,
) -> list[str]:
    """Phát cảnh báo khi feature nằm ngoài dải P1-P99 của training."""
    if not feature_ranges:
        return []
    warnings = []
    for column, (lower, upper) in feature_ranges.items():
        value = float(row[column])
        if value < lower or value > upper:
            warnings.append(
                f"FEATURE_OUTSIDE_TRAINING_RANGE: {column}={value:.4f} "
                f"ngoài [{lower:.4f}, {upper:.4f}]"
            )
    return warnings


def predict_records(frame: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tính score recording và median subject score, không gán nhãn chẩn đoán."""
    validated = _prepare_inference_frame(frame)
    feature_columns = list(bundle["feature_columns"])
    scores = positive_class_probability(bundle["model"], validated[feature_columns])
    warning_lists = [
        check_recording_training_range(row, _training_ranges(bundle))
        for _, row in validated[feature_columns].iterrows()
    ]
    records = pd.DataFrame(
        {
            "recording_id": validated[ID_COLUMN].map(sanitize_csv_value),
            SUBJECT_COLUMN: validated[SUBJECT_COLUMN].map(sanitize_csv_value),
            "screening_score": scores,
            "feature_warnings": warning_lists,
        }
    )

    threshold = float(bundle["decision_threshold"])
    subject_rows = []
    for subject_id, group in records.groupby(SUBJECT_COLUMN, sort=True):
        subject_score = float(group["screening_score"].median())
        warnings = [
            warning for warning_list in group["feature_warnings"] for warning in warning_list
        ]
        if len(group) < 3:
            warnings.insert(
                0,
                f"FEWER_RECORDINGS_WARNING: Đối tượng có {len(group)} bản ghi; "
                "mô hình được huấn luyện trên nhiều bản ghi lặp lại mỗi đối tượng.",
            )
        warnings = list(dict.fromkeys(warnings))
        subject_rows.append(
            {
                "subject_id": subject_id,
                "n_recordings": int(len(group)),
                "subject_screening_score": subject_score,
                "decision_threshold": threshold,
                "screening_result": (
                    "above_internal_threshold"
                    if subject_score >= threshold
                    else "below_internal_threshold"
                ),
                "warnings": warnings,
            }
        )
    return records, pd.DataFrame(subject_rows)


def predict_subject_records(
    subject_id: str,
    recordings: list[dict[str, float]],
    bundle: dict,
) -> dict:
    """Sàng lọc một subject từ 1..N recording, không nhận training label."""
    subject_id = str(subject_id).strip()
    if not subject_id:
        raise ValueError("subject_id không được để trống.")
    if not recordings:
        raise ValueError("Danh sách recordings không được để trống.")
    if any("status" in recording for recording in recordings):
        raise ValueError("Input suy luận không được chứa nhãn huấn luyện 'status'.")

    frame = pd.DataFrame(recordings)
    frame[ID_COLUMN] = [f"{subject_id}_{index + 1}" for index in range(len(frame))]
    frame[SUBJECT_COLUMN] = subject_id
    records, subjects = predict_records(frame, bundle)
    subject = subjects.iloc[0]
    return {
        "subject_id": subject_id,
        "n_recordings": int(subject["n_recordings"]),
        "subject_screening_score": float(subject["subject_screening_score"]),
        "decision_threshold": float(subject["decision_threshold"]),
        "screening_result": str(subject["screening_result"]),
        "warnings": list(subject["warnings"]),
        "aggregation": "median",
        "recording_scores": records["screening_score"].astype(float).tolist(),
    }
