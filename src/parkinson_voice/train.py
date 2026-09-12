"""Huấn luyện và đánh giá mô hình phân loại giọng nói Parkinson ở cấp độ subject.

Quy trình:
1. Đọc và kiểm tra dữ liệu, trích xuất mã đối tượng (subject_id).
2. Nested Subject CV (4 outer × 3 inner) đánh giá khả năng tổng quát hóa không rò rỉ.
3. Chọn siêu tham số (C, class_weight) và ngưỡng quyết định từ OOF inner folds.
4. Ước lượng khoảng tin cậy 95% CI bằng subject-level bootstrap.
5. Huấn luyện mô hình cuối cùng trên toàn bộ dữ liệu và lưu vào artifacts/model.joblib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from parkinson_voice.audit import _subject_class_distribution
from parkinson_voice.data import TARGET_COLUMN, load_data
from parkinson_voice.evaluate import (
    bootstrap_subject_confidence_intervals,
    calculate_metrics,
    make_subject_folds,
)
from parkinson_voice.features import (
    MODEL_FEATURES,
    compute_feature_percentiles,
    make_logistic_pipeline,
)
from parkinson_voice.model_selection import (
    nested_subject_cross_fitted,
    search_logistic_configuration,
)

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"
DEFAULT_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RANDOM_STATE = int(DEFAULT_CONFIG["random_state"])
AGGREGATION = str(DEFAULT_CONFIG["aggregation"])
MAX_ITER = int(DEFAULT_CONFIG["model"]["max_iter"])
if AGGREGATION != "median":
    raise ValueError("configs/default.json phải khóa aggregation='median'.")


def train(data_path: str | Path, artifact_dir: str | Path = "artifacts") -> pd.DataFrame:
    """Huấn luyện mô hình và lưu artifact."""
    data_path = Path(data_path)
    output = Path(artifact_dir)
    evaluation_dir = output / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    frame = load_data(data_path)

    C_values = [float(value) for value in DEFAULT_CONFIG["search"]["C"]]
    class_weights = list(DEFAULT_CONFIG["search"]["class_weight"])
    outer_folds = int(DEFAULT_CONFIG["outer_folds"])
    inner_folds = int(DEFAULT_CONFIG["inner_folds"])

    # Nested subject cross-validation (outer folds chỉ để đánh giá)
    fold_metrics, cross_fitted, selection_table = nested_subject_cross_fitted(
        frame,
        feature_columns=MODEL_FEATURES,
        outer_splits=outer_folds,
        inner_splits=inner_folds,
        random_state=RANDOM_STATE,
        C_values=C_values,
        class_weights=class_weights,
        max_iter=MAX_ITER,
    )
    cross_fitted_metrics = calculate_metrics(
        cross_fitted[TARGET_COLUMN],
        cross_fitted["prediction"],
        cross_fitted["subject_score"],
    )
    confidence_intervals = bootstrap_subject_confidence_intervals(
        cross_fitted.rename(columns={"subject_score": "probability"}),
        n_bootstrap=int(DEFAULT_CONFIG.get("bootstrap_replicates", 1000)),
        random_state=RANDOM_STATE,
    )

    # Full-data subject OOF để chọn hyperparameter và threshold cho mô hình cuối
    full_inner_folds = make_subject_folds(
        frame,
        n_splits=inner_folds,
        random_state=RANDOM_STATE + 101,
    )
    full_selection = search_logistic_configuration(
        frame,
        full_inner_folds,
        feature_columns=MODEL_FEATURES,
        C_values=C_values,
        class_weights=class_weights,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    final_model = make_logistic_pipeline(
        C=full_selection["C"],
        class_weight=full_selection["class_weight"],
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER,
    )
    final_model.fit(frame[MODEL_FEATURES], frame[TARGET_COLUMN])

    feature_ranges = compute_feature_percentiles(frame, MODEL_FEATURES)
    final_threshold = round(float(full_selection["threshold"]), 4)

    # Đóng gói gọn artifact triển khai
    bundle = {
        "model": final_model,
        "feature_columns": MODEL_FEATURES,
        "decision_threshold": final_threshold,
        "aggregation": AGGREGATION,
        "feature_ranges": feature_ranges,
    }
    joblib.dump(bundle, output / "model.joblib")

    # Lưu kết quả đánh giá
    cross_fitted.to_csv(evaluation_dir / "cross_fitted_subject_predictions.csv", index=False)
    fold_metrics.to_csv(evaluation_dir / "fold_metrics.csv", index=False)
    selection_table.to_csv(evaluation_dir / "inner_selection.csv", index=False)
    confidence_intervals.to_csv(evaluation_dir / "bootstrap_ci.csv", index=False)
    full_selection["threshold_table"].to_csv(evaluation_dir / "threshold_search.csv", index=False)

    final_oof_subjects = full_selection["subjects"].copy()
    final_oof_metrics = calculate_metrics(
        final_oof_subjects["status"],
        final_oof_subjects["prediction"],
        final_oof_subjects["probability"],
    )

    metrics_payload = {
        "dataset": {
            "n_recordings": int(len(frame)),
            "n_subjects": int(frame["subject_id"].nunique()),
            "subject_distribution": _subject_class_distribution(frame),
        },
        "selection": {
            "C": full_selection["C"],
            "class_weight": full_selection["class_weight"],
        },
        "nested_cv_subject": {
            **cross_fitted_metrics,
            "fold_mean": {
                column: float(fold_metrics[column].mean())
                for column in ["Balanced Accuracy", "F1-macro", "ROC-AUC"]
            },
            "fold_std": {
                column: float(fold_metrics[column].std(ddof=0))
                for column in ["Balanced Accuracy", "F1-macro", "ROC-AUC"]
            },
        },
        "deployment_oof": {
            **final_oof_metrics,
            "decision_threshold": final_threshold,
            "aggregation": AGGREGATION,
        },
        "evaluation_protocol": {
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "unit": "subject",
            "primary_metric": "Balanced Accuracy",
        },
        "artifact": "artifacts/model.joblib",
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return full_selection["comparison"]


def main() -> None:
    """Điểm vào CLI cho training/evaluation."""
    parser = argparse.ArgumentParser(description="Train/evaluate Parkinson screening model.")
    parser.add_argument("--data", default="data/parkinsons.csv", help="Đường dẫn CSV.")
    parser.add_argument("--artifacts", default="artifacts", help="Thư mục artifact.")
    args = parser.parse_args()
    print(train(args.data, args.artifacts).to_string(index=False))


if __name__ == "__main__":
    main()
