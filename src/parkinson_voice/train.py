"""Huấn luyện và đánh giá pipeline canonical của dự án.

Luồng chính:
1. Audit dataset và ghi manifest.
2. Nested subject CV 4 outer × 3 inner trên toàn bộ subject.
3. Chọn C/class_weight và threshold chỉ từ OOF của phần train.
4. Bootstrap trên 32 cross-fitted subject predictions.
5. Fit final Logistic Regression trên toàn bộ dataset để triển khai.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from parkinson_voice.audit import build_data_manifest
from parkinson_voice.data import ORIGINAL_FEATURES, TARGET_COLUMN, load_data
from parkinson_voice.evaluate import (
    bootstrap_subject_confidence_intervals,
    calculate_metrics,
    expected_calibration_error,
    make_subject_folds,
)
from parkinson_voice.features import (
    MODEL_FEATURES,
    REDUNDANT_FEATURES,
    compute_feature_percentiles,
    make_logistic_pipeline,
)
from parkinson_voice.model_selection import (
    nested_subject_cross_fitted,
    search_logistic_configuration,
)
from parkinson_voice.utils import sha256_file

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"
DEFAULT_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RANDOM_STATE = int(DEFAULT_CONFIG["random_state"])
AGGREGATION = str(DEFAULT_CONFIG["aggregation"])
MAX_ITER = int(DEFAULT_CONFIG["model"]["max_iter"])
if AGGREGATION != "median":
    raise ValueError("configs/default.json phải khóa aggregation='median'.")


def _git_commit(project_root: Path) -> str:
    """Lấy commit hiện tại để truy nguyên artifact."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _json_records(frame: pd.DataFrame) -> list[dict]:
    """Đổi DataFrame sang object JSON mà không giữ kiểu NumPy."""
    return json.loads(frame.to_json(orient="records"))


def train(data_path: str | Path, artifact_dir: str | Path = "artifacts") -> pd.DataFrame:
    """Chạy audit, nested evaluation và fit release v1.0.0."""
    data_path = Path(data_path)
    output = Path(artifact_dir)
    evaluation_dir = output / "evaluation"
    release_dir = output / "releases" / "v1.0.0"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)

    frame = load_data(data_path)
    manifest = build_data_manifest(frame, data_path)
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    C_values = [float(value) for value in DEFAULT_CONFIG["search"]["C"]]
    class_weights = list(DEFAULT_CONFIG["search"]["class_weight"])
    outer_folds = int(DEFAULT_CONFIG["outer_folds"])
    inner_folds = int(DEFAULT_CONFIG["inner_folds"])

    # Outer-test chỉ dùng để đo khả năng tổng quát hóa.
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
        n_bootstrap=int(DEFAULT_CONFIG["bootstrap_replicates"]),
        random_state=RANDOM_STATE,
    )

    # Full-data group OOF dùng để khóa threshold cho model deployment.
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

    recordings_per_subject = frame.groupby("subject_id").size()
    training_recordings = {
        "min": int(recordings_per_subject.min()),
        "median": float(recordings_per_subject.median()),
        "max": int(recordings_per_subject.max()),
    }
    data_sha256 = sha256_file(data_path)
    git_commit = _git_commit(PROJECT_ROOT)
    final_oof_subjects = full_selection["subjects"].copy()
    final_oof_metrics = calculate_metrics(
        final_oof_subjects["status"],
        final_oof_subjects["prediction"],
        final_oof_subjects["probability"],
    )
    final_oof_metrics["ECE (5 bins)"] = expected_calibration_error(
        final_oof_subjects["status"],
        final_oof_subjects["probability"],
    )
    feature_ranges = compute_feature_percentiles(frame, MODEL_FEATURES)
    model_version = "1.0.0"

    metadata = {
        "schema_version": 2,
        "model_version": model_version,
        "model_type": "logistic_regression",
        "feature_count": len(MODEL_FEATURES),
        "aggregation": AGGREGATION,
        "decision_threshold": float(full_selection["threshold"]),
        "dataset_sha256": data_sha256,
        "evaluation_protocol": {
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "unit": "subject",
            "primary_metric": "Balanced Accuracy",
        },
        "training_subjects": int(frame["subject_id"].nunique()),
        "training_recordings": int(len(frame)),
        "selection": {
            "C": full_selection["C"],
            "class_weight": full_selection["class_weight"],
        },
        "sklearn_version": sklearn.__version__,
        "git_commit": git_commit,
        "scope": "research-only",
    }
    feature_schema = {
        "schema_version": 2,
        "source_features": ORIGINAL_FEATURES,
        "model_features": MODEL_FEATURES,
        "dropped_features": REDUNDANT_FEATURES,
        "subject_id_rule": "drop_final_recording_suffix",
        "measurement_protocol": (
            "UCI Parkinsons precomputed acoustic measurements; runtime input must use a compatible protocol."
        ),
    }
    bundle = {
        "model": final_model,
        "model_version": model_version,
        "model_type": "logistic_regression",
        "champion_name": "Logistic Regression",
        "feature_columns": MODEL_FEATURES,
        "original_feature_columns": ORIGINAL_FEATURES,
        "dropped_redundant_features": REDUNDANT_FEATURES,
        "decision_threshold": float(full_selection["threshold"]),
        "aggregation": AGGREGATION,
        "probability_aggregation": AGGREGATION,
        "training_feature_ranges": feature_ranges,
        "feature_p1_p99": feature_ranges,
        "training_recordings_per_subject": training_recordings,
        "training_min_recordings": training_recordings["min"],
        "training_subjects": int(frame["subject_id"].nunique()),
        "training_recordings": int(len(frame)),
        "class_distribution": manifest["subject_distribution"],
        "dataset_sha256": data_sha256,
        "data_sha256": data_sha256,
        "evaluation_protocol": metadata["evaluation_protocol"],
        "nested_cv_metrics": cross_fitted_metrics,
        "deployment_oof_metrics": final_oof_metrics,
        "calibration": "none; Brier and ECE are descriptive evaluation metrics",
        "schema_version": 2,
        "artifact_version": model_version,
        "feature_contract_version": model_version,
        "git_commit_sha": git_commit,
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
    }

    joblib.dump(bundle, release_dir / "model.joblib")
    cross_fitted.to_csv(evaluation_dir / "cross_fitted_subject_predictions.csv", index=False)
    fold_metrics.to_csv(evaluation_dir / "fold_metrics.csv", index=False)
    selection_table.to_csv(evaluation_dir / "inner_selection.csv", index=False)
    confidence_intervals.to_csv(evaluation_dir / "bootstrap_ci.csv", index=False)
    full_selection["threshold_table"].to_csv(evaluation_dir / "threshold_search.csv", index=False)

    (release_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (release_dir / "feature_schema.json").write_text(
        json.dumps(feature_schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    evaluation_json = {
        "protocol": metadata["evaluation_protocol"],
        "cross_fitted_subject_metrics": cross_fitted_metrics,
        "fold_metrics": _json_records(fold_metrics),
        "bootstrap_ci": _json_records(confidence_intervals),
        "deployment_oof_metrics": final_oof_metrics,
        "deployment_selection": {
            "C": full_selection["C"],
            "class_weight": full_selection["class_weight"],
            "threshold": full_selection["threshold"],
            "aggregation": AGGREGATION,
        },
    }
    (release_dir / "evaluation.json").write_text(
        json.dumps(evaluation_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    release_model_card = f"""# Parkinson Voice Feature Screening v{model_version}

## Phạm vi

Research prototype, không dùng để chẩn đoán. Mô hình nhận 20 acoustic features
từ measurement protocol tương thích với UCI; không nhận WAV/MP3 và không tự trích
xuất đặc trưng âm thanh.

## Pipeline và quyết định

- `StandardScaler` → L2 `LogisticRegression` (`C={full_selection["C"]}`, `class_weight={full_selection["class_weight"]}`).
- Recording chỉ có `screening_score`; subject score là median của các recording.
- Threshold OOF của deployment là `{full_selection["threshold"]:.12f}`.
- Ít recording hơn ngưỡng training sẽ tạo cảnh báo `INSUFFICIENT_RECORDINGS`.

## Đánh giá

- Nested stratified subject CV: {outer_folds} outer × {inner_folds} inner trên
  {int(frame["subject_id"].nunique())} subject; mỗi subject làm outer-test đúng một lần.
- Primary metric: Balanced Accuracy = {cross_fitted_metrics["Balanced Accuracy"]:.4f}.
- Macro-F1 = {cross_fitted_metrics["F1-macro"]:.4f}; ROC-AUC = {cross_fitted_metrics["ROC-AUC"]:.4f}.
- Bootstrap 95% CI dùng 5.000 mẫu ở cấp subject; đây là ước lượng nghiên cứu nội bộ.
- Chưa có external patient cohort hoặc clinical validation.
"""
    (release_dir / "model_card.md").write_text(release_model_card, encoding="utf-8")

    root_metrics = {
        "dataset": manifest,
        "selection": metadata["selection"],
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
            "decision_threshold": float(full_selection["threshold"]),
            "aggregation": AGGREGATION,
        },
        "evaluation_protocol": metadata["evaluation_protocol"],
        "artifact": "releases/v1.0.0/model.joblib",
    }
    (output / "metrics.json").write_text(
        json.dumps(root_metrics, ensure_ascii=False, indent=2),
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
