"""Chọn cấu hình và đánh giá nested CV ở cấp độ subject.

Production v1 chỉ dùng StandardScaler → L2 Logistic Regression. Các quyết định
C, class_weight và threshold đều được chọn từ dữ liệu OOF của phần train tương
ứng; outer-test chỉ được dùng để đo tổng quát hóa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from src.data import SUBJECT_COLUMN, TARGET_COLUMN
from src.evaluate import (
    aggregate_subject_predictions,
    calculate_metrics,
    make_subject_folds,
    positive_score,
    select_decision_threshold,
)
from src.features import MODEL_FEATURES, make_logistic_pipeline


@dataclass
class ChampionResult:
    """Kết quả lựa chọn model của API tương thích; production luôn là Logistic Regression."""

    name: str
    estimator: Any
    parameters: dict[str, Any]
    metrics: dict[str, float]


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
        scores[valid_index] = positive_score(model, frame.iloc[valid_index][feature_columns])
    if np.isnan(scores).any():
        raise AssertionError("OOF chưa tạo score cho toàn bộ recording.")
    return scores


def evaluate_parameter_set(
    estimator,
    parameters: dict[str, Any],
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_columns: list[str],
    *,
    aggregation: str = "median",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Đánh giá một estimator generic; dùng chủ yếu cho experiment tương thích."""
    rows = []
    for fold_number, (fit_index, valid_index) in enumerate(folds, start=1):
        fit_frame = frame.iloc[fit_index]
        valid_frame = frame.iloc[valid_index]
        model = clone(estimator).set_params(**parameters)
        model.fit(fit_frame[feature_columns], fit_frame[TARGET_COLUMN])
        scores = positive_score(model, valid_frame[feature_columns])
        subjects = aggregate_subject_predictions(
            valid_frame,
            scores,
            aggregation=aggregation,
            threshold=threshold,
        )
        rows.append(
            {
                "Fold": fold_number,
                **calculate_metrics(
                    subjects["status"],
                    subjects["prediction"],
                    subjects["probability"],
                ),
            }
        )
    return pd.DataFrame(rows)


def search_subject_level(
    estimator,
    parameter_grid: dict[str, list[Any]],
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_columns: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    """Tìm tham số generic theo subject; benchmark ngoài production."""
    candidates = []
    for parameters in list(ParameterGrid(parameter_grid)) if parameter_grid else [{}]:
        fold_table = evaluate_parameter_set(
            estimator,
            parameters,
            frame,
            folds,
            feature_columns,
        )
        candidates.append(
            {
                "Parameters": parameters,
                "Subject F1-macro mean": float(fold_table["F1-macro"].mean()),
                "Subject F1-macro std": float(fold_table["F1-macro"].std(ddof=0)),
                "Subject Balanced Accuracy mean": float(fold_table["Balanced Accuracy"].mean()),
                "Subject ROC-AUC mean": float(fold_table["ROC-AUC"].mean()),
            }
        )
    result = pd.DataFrame(candidates).sort_values(
        [
            "Subject Balanced Accuracy mean",
            "Subject F1-macro mean",
            "Subject ROC-AUC mean",
        ],
        ascending=False,
    )
    return result.iloc[0], result


def _model_complexity_rank(name: str) -> int:
    """Xếp hạng legacy để experiment cũ không bị lỗi import."""
    return {
        "Dummy": 0,
        "Logistic Regression": 1,
        "KNN": 2,
        "Random Forest": 3,
        "HistGradientBoosting": 4,
    }.get(name, 10)


def select_champion(
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_specs: dict[str, tuple[Any, dict[str, list[Any]]]],
    *,
    feature_columns: list[str] | None = None,
    f1_tolerance: float = 0.005,
) -> ChampionResult:
    """API tương thích cho experiment; không được production dùng để chọn model."""
    del f1_tolerance
    feature_columns = feature_columns or MODEL_FEATURES
    candidates = []
    for name, (estimator, grid) in model_specs.items():
        best, _ = search_subject_level(
            estimator,
            grid,
            frame,
            folds,
            feature_columns,
        )
        candidates.append(
            ChampionResult(
                name=name,
                estimator=clone(estimator).set_params(**best["Parameters"]),
                parameters=best["Parameters"],
                metrics={
                    "Subject F1-macro mean": float(best["Subject F1-macro mean"]),
                    "Subject F1-macro std": float(best["Subject F1-macro std"]),
                    "Subject Balanced Accuracy mean": float(best["Subject Balanced Accuracy mean"]),
                    "Subject ROC-AUC mean": float(best["Subject ROC-AUC mean"]),
                },
            )
        )
    if not candidates:
        raise ValueError("model_specs không được rỗng.")
    return max(
        candidates,
        key=lambda candidate: (
            candidate.metrics["Subject Balanced Accuracy mean"],
            candidate.metrics["Subject F1-macro mean"],
            -candidate.metrics["Subject F1-macro std"],
            -_model_complexity_rank(candidate.name),
        ),
    )


def search_logistic_configuration(
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    feature_columns: list[str] | None = None,
    C_values: list[float] | None = None,
    class_weights: list[str | None] | None = None,
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
            random_state=random_state + outer_fold,
        )
        model = make_logistic_pipeline(
            C=selected["C"],
            class_weight=selected["class_weight"],
            random_state=random_state,
        )
        model.fit(outer_train[feature_columns], outer_train[TARGET_COLUMN])
        test_scores = positive_score(model, outer_test[feature_columns])
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


def fit_selection_rule(
    train_frame: pd.DataFrame,
    oof_probabilities: np.ndarray,
    *,
    aggregation_candidates: tuple[str, ...] = ("median",),
    minimum_specificity: float | None = None,
) -> tuple[str, float]:
    """Khóa median và threshold thống kê từ subject-level OOF."""
    if aggregation_candidates != ("median",):
        raise ValueError("Production chỉ cho phép aggregation='median'.")
    subjects = aggregate_subject_predictions(train_frame, oof_probabilities, aggregation="median")
    threshold, _ = select_decision_threshold(
        subjects,
        minimum_specificity=minimum_specificity,
    )
    return "median", threshold


def fit_complete_pipeline(
    champion: ChampionResult,
    train_frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    n_splits: int = 3,
    random_state: int = 42,
) -> tuple[Any, str, float]:
    """Fit model không calibration và khóa median/threshold từ group OOF."""
    feature_columns = feature_columns or MODEL_FEATURES
    folds = make_subject_folds(train_frame, n_splits=n_splits, random_state=random_state)
    scores = _oof_recording_scores(train_frame, champion.estimator, folds, feature_columns)
    subjects = aggregate_subject_predictions(train_frame, scores, aggregation="median")
    threshold, _ = select_decision_threshold(subjects)
    model = clone(champion.estimator)
    model.fit(train_frame[feature_columns], train_frame[TARGET_COLUMN])
    return model, "median", threshold


def nested_subject_evaluation(
    frame: pd.DataFrame,
    model_specs: dict[str, tuple[Any, dict[str, list[Any]]]] | None = None,
    *,
    feature_columns: list[str] | None = None,
    outer_splits: int = 4,
    inner_splits: int = 3,
    random_state: int = 42,
    return_predictions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chạy nested 4x3 canonical; model_specs chỉ giữ tương thích test cũ."""
    C_values = None
    class_weights = None
    if model_specs:
        logistic_spec = next(
            (
                (estimator, grid)
                for name, (estimator, grid) in model_specs.items()
                if "logistic" in name.lower()
            ),
            None,
        )
        if logistic_spec is not None:
            estimator, grid = logistic_spec
            C_values = [
                float(value)
                for value in grid.get(
                    "model__C",
                    [estimator.get_params().get("model__C", 1.0)],
                )
            ]
            class_weights = grid.get(
                "model__class_weight",
                [estimator.get_params().get("model__class_weight")],
            )
    fold_metrics, predictions, selections = nested_subject_cross_fitted(
        frame,
        feature_columns=feature_columns,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        random_state=random_state,
        C_values=C_values,
        class_weights=class_weights,
    )
    result = fold_metrics.rename(columns={"Model": "Champion"})
    if return_predictions:
        return result, predictions, selections
    return result
