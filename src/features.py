"""Feature contract và pipeline production của dự án.

Hai cột dẫn xuất được loại bỏ vì quan hệ đại số tất định. Đây là xử lý dư thừa
đặc trưng, không phải biện pháp chống leakage. Production v1 chỉ chuẩn hóa 20
đặc trưng rồi huấn luyện Logistic Regression có regularization L2.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import ORIGINAL_FEATURES

REDUNDANT_FEATURES = ["Jitter:DDP", "Shimmer:DDA"]


# Danh sách 20 đặc trưng số độc lập đưa vào huấn luyện mô hình
MODEL_FEATURES = [column for column in ORIGINAL_FEATURES if column not in REDUNDANT_FEATURES]


def compute_feature_percentiles(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    lower: float = 0.01,
    upper: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """Tính dải phân vị dùng làm cảnh báo plausibility theo training range."""
    selected_features = features or MODEL_FEATURES
    missing = sorted(set(selected_features).difference(frame.columns))
    if missing:
        raise ValueError(f"Thiếu đặc trưng khi tính training range: {missing}")
    if not 0 <= lower < upper <= 1:
        raise ValueError("lower và upper phải thỏa mãn 0 <= lower < upper <= 1.")

    return {
        column: (
            float(frame[column].quantile(lower)),
            float(frame[column].quantile(upper)),
        )
        for column in selected_features
    }


def make_pipeline(model: BaseEstimator) -> Pipeline:
    """Tạo pipeline chuẩn hóa và estimator, không có bước chọn feature."""
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def make_logistic_pipeline(
    *,
    C: float = 1.0,
    class_weight: str | None = None,
    random_state: int = 42,
    max_iter: int = 3000,
) -> Pipeline:
    """Tạo pipeline Logistic Regression canonical cho production v1."""
    if C <= 0:
        raise ValueError("C phải lớn hơn 0.")
    return make_pipeline(
        LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            solver="liblinear",
            random_state=random_state,
        )
    )
