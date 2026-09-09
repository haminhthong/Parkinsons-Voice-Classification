"""Module tiện ích hệ thống và kiểm tra khả năng tái lập.

Cung cấp hàm tính mã băm SHA-256 của tệp để định danh chính xác phiên bản dữ liệu đầu vào.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SUPPORTED_AGGREGATIONS = ("median",)


def sha256_file(path: str | Path) -> str:
    """Tính toán mã kiểm tra băm SHA-256 của một tệp tin.

    Args:
        path: Đường dẫn tệp tin cần tính hash.

    Returns:
        str: Chuỗi 64 ký tự hex đại diện cho mã SHA-256 của tệp.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_class_probability(model, features) -> np.ndarray:
    """Lấy xác suất của lớp dương ``status=1`` từ mô hình phân loại.

    Hàm dùng chung này giúp bước đánh giá và bước suy luận diễn giải thứ tự
    ``classes_`` theo cùng một cách, thay vì mặc định lớp dương luôn ở cột thứ hai.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError("Mô hình bắt buộc phải hỗ trợ phương thức predict_proba.")

    positive_indices = np.flatnonzero(np.asarray(model.classes_) == 1)
    if positive_indices.size != 1:
        raise ValueError("Mô hình phải chứa đúng một lớp dương có nhãn 1.")

    probabilities = np.asarray(
        model.predict_proba(features)[:, int(positive_indices[0])],
        dtype=float,
    )
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Mô hình trả về xác suất không hợp lệ hoặc nằm ngoài [0, 1].")
    return probabilities


def normalize_aggregation(aggregation: str) -> str:
    """Chuẩn hóa và kiểm tra tên quy tắc gộp xác suất theo bệnh nhân."""
    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"Cách gộp {aggregation!r} không hợp lệ; chọn một trong {list(SUPPORTED_AGGREGATIONS)}."
        )
    return aggregation
