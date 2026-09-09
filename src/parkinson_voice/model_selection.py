"""Chọn cấu hình và đánh giá nested CV ở cấp độ subject.

Production v1 chỉ dùng StandardScaler → L2 Logistic Regression. Các quyết định
C, class_weight và threshold đều được chọn từ dữ liệu OOF của phần train tương
ứng; outer-test chỉ được dùng để đo tổng quát hóa.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from parkinson_voice.data import SUBJECT_COLUMN, TARGET_COLUMN
from parkinson_voice.evaluate import (
    aggregate_subject_predictions,
    calculate_metrics,
    make_subject_folds,
    select_decision_threshold,
)
from parkinson_voice.features import MODEL_FEATURES, make_logistic_pipeline
from parkinson_voice.utils import positive_class_probability


def _oof_recording_scores(
    frame: pd.DataFrame,
    estimator,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_columns: list[str],
) -> np.ndarray:
    """Sinh score OOF; mỗi recording chỉ được dự đoán ở fold validation của nó."""
    scores = np.full(len(frame), np.nan, dtype=float)
    for fit_index, valid_index in folds:
        model = clone(estimator)
        fit_frame = frame.iloc[fit_index]
        model.fit(fit_frame[feature_columns], fit_frame[TARGET_COLUMN])
        scores[valid_index] = positive_class_probability(
            model,
            frame.iloc[valid_index][feature_columns],
        )
    if np.isnan(scores).any():
        raise AssertionError("OOF chưa tạo score cho toàn bộ recording.")
    return scores


def search_logistic_configuration(
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    feature_columns: list[str] | None = None,
    C_values: list[float] | None = None,
    class_weights: list[str | None] | None = None,
    max_iter: int = 3000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Chọn C và class_weight chỉ bằng OOF subject-level của inner CV."""
    feature_columns = feature_columns or MODEL_FEATURES
    C_values = C_values or [0.01, 0.1, 1.0, 10.0, 100.0]
    class_weights = class_weights or [None, "balanced"]
    candidates: list[dict[str, Any]] = []

    for C in C_values:
        for class_weight in class_weights:
            estimator = make_logistic_pipeline(
                C=float(C),
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=random_state,
            )
            scores = _oof_recording_scores(frame, estimator, folds, feature_columns)
            subjects = aggregate_subject_predictions(frame, scores, aggregation="median")
            threshold, threshold_table = select_decision_threshold(subjects)
            subjects["prediction"] = (subjects["probability"] >= threshold).astype(int)
            metrics = calculate_metrics(
                subjects["status"],
                subjects["prediction"],
                subjects["probability"],
            )
            candidates.append(
                {
                    "C": float(C),
                    "class_weight": class_weight,
                    "threshold": float(threshold),
                    "oof_probabilities": scores,
                    "subjects": subjects,
                    "threshold_table": threshold_table,
                    **metrics,
                }
            )

    comparison = pd.DataFrame(
        [
            {
                "_candidate_index": index,
                **{
                    key: value
                    for key, value in candidate.items()
                    if key not in {"oof_probabilities", "subjects", "threshold_table"}
                },
            }
            for index, candidate in enumerate(candidates)
        ]
    ).sort_values(
        ["Balanced Accuracy", "F1-macro", "ROC-AUC", "Brier score", "C"],
        ascending=[False, False, False, True, True],
        na_position="last",
    )
    best_index = int(comparison.iloc[0]["_candidate_index"])
    comparison = comparison.drop(columns=["_candidate_index"]).reset_index(drop=True)
    return {**candidates[best_index], "comparison": comparison}


def nested_subject_cross_fitted(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    outer_splits: int = 4,
    inner_splits: int = 3,
    random_state: int = 42,
    C_values: list[float] | None = None,
    class_weights: list[str | None] | None = None,
    max_iter: int = 3000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đánh giá nested subject CV và trả về cross-fitted prediction của mọi subject."""
    feature_columns = feature_columns or MODEL_FEATURES
    outer_folds = make_subject_folds(frame, n_splits=outer_splits, random_state=random_state)
    fold_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    selection_tables: list[pd.DataFrame] = []

    for outer_fold, (fit_index, test_index) in enumerate(outer_folds, start=1):
        outer_train = frame.iloc[fit_index].reset_index(drop=True)
        outer_test = frame.iloc[test_index].reset_index(drop=True)
        inner_folds = make_subject_folds(
            outer_train,
            n_splits=inner_splits,
            random_state=random_state + outer_fold,
        )
        selected = search_logistic_configuration(
            outer_train,
            inner_folds,
            feature_columns=feature_columns,
            C_values=C_values,
            class_weights=class_weights,
            max_iter=max_iter,
            random_state=random_state + outer_fold,
        )
        model = make_logistic_pipeline(
            C=selected["C"],
            class_weight=selected["class_weight"],
            max_iter=max_iter,
            random_state=random_state,
        )
        model.fit(outer_train[feature_columns], outer_train[TARGET_COLUMN])
        test_scores = positive_class_probability(model, outer_test[feature_columns])
        subjects = aggregate_subject_predictions(
            outer_test,
            test_scores,
            aggregation="median",
            threshold=selected["threshold"],
        )
        fold_rows.append(
            {
                "Outer fold": outer_fold,
                "Model": "Logistic Regression",
                "Aggregation": "median",
                "C": selected["C"],
                "class_weight": selected["class_weight"],
                "Threshold": selected["threshold"],
                **calculate_metrics(
                    subjects["status"],
                    subjects["prediction"],
                    subjects["probability"],
                ),
            }
        )
        prediction_tables.append(
            subjects.rename(
                columns={
                    "recordings": "n_recordings",
                    "probability": "subject_score",
                }
            )[
                [
                    SUBJECT_COLUMN,
                    TARGET_COLUMN,
                    "n_recordings",
                    "subject_score",
                    "prediction",
                ]
            ].assign(
                outer_fold=outer_fold,
                decision_threshold=float(selected["threshold"]),
            )
        )
        selection_tables.append(selected["comparison"].assign(outer_fold=outer_fold))

    predictions = pd.concat(prediction_tables, ignore_index=True)[
        [
            SUBJECT_COLUMN,
            TARGET_COLUMN,
            "outer_fold",
            "n_recordings",
            "subject_score",
            "decision_threshold",
            "prediction",
        ]
    ]
    if predictions[SUBJECT_COLUMN].duplicated().any():
        raise AssertionError(
            "Mỗi subject phải xuất hiện đúng một lần trong cross-fitted predictions."
        )
    return (
        pd.DataFrame(fold_rows),
        predictions.sort_values(SUBJECT_COLUMN).reset_index(drop=True),
        pd.concat(selection_tables, ignore_index=True),
    )
