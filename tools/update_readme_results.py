"""Tool tự động cập nhật kết quả benchmark và metrics từ artifacts vào README.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[1]
README_PATH = PROJECT_ROOT / "README.md"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

START_MARKER = "<!-- GENERATED_RESULTS_START -->"
END_MARKER = "<!-- GENERATED_RESULTS_END -->"


def generate_results_markdown() -> str:
    """Đọc artifacts và render bảng kết quả dưới dạng chuỗi Markdown."""
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    benchmark_path = ARTIFACTS_DIR / "model_benchmark.csv"

    if not metrics_path.exists() or not benchmark_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy artifacts/metrics.json hoặc artifacts/model_benchmark.csv."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    benchmark = pd.read_csv(benchmark_path)

    lines = [
        "### Bảng So Sánh Benchmark Mô Hình (Subject-Level Cross-Validation)",
        "",
        "| Model | Subject F1-macro mean | Subject F1-macro std | Subject Balanced Accuracy mean | Subject ROC-AUC mean |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for _, row in benchmark.iterrows():
        model = str(row["Model"])
        f1_mean = f"{float(row['Subject F1-macro mean']):.4f}"
        f1_std = f"{float(row['Subject F1-macro std']):.4f}"
        bal_acc = f"{float(row['Subject Balanced Accuracy mean']):.4f}"
        roc_auc = f"{float(row['Subject ROC-AUC mean']):.4f}"
        lines.append(f"| {model} | {f1_mean} | {f1_std} | {bal_acc} | {roc_auc} |")

    nested = metrics.get("nested_cv_subject", {})
    holdout = metrics.get("holdout_subject", {})

    lines.extend(
        [
            "",
            "### Kết Quả Đánh Giá Tổng Thể Pipeline",
            "",
            f"- **Mô hình Champion**: `{metrics.get('selection', {}).get('champion', 'N/A')}`",
            f"- **Quy tắc gộp xác suất**: `{metrics.get('calibration', {}).get('aggregation', 'N/A')}`",
            f"- **Ngưỡng quyết định**: `{metrics.get('calibration', {}).get('threshold', 0.5):.4f}`",
            "",
            "#### Nested Subject-Level Cross-Validation:",
            f"- **Subject F1-macro mean**: `{nested.get('F1-macro mean', 0.0):.4f}` (std: `{nested.get('F1-macro std', 0.0):.4f}`)",
            f"- **Subject Balanced Accuracy mean**: `{nested.get('Balanced Accuracy mean', 0.0):.4f}`",
            f"- **Subject ROC-AUC mean**: `{nested.get('ROC-AUC mean', 0.0):.4f}`",
            "",
            "#### Holdout Patient Test Set:",
            f"- **F1-macro**: `{holdout.get('F1-macro', 0.0):.4f}`",
            f"- **Balanced Accuracy**: `{holdout.get('Balanced Accuracy', 0.0):.4f}`",
            f"- **Recall/Sensitivity**: `{holdout.get('Recall/Sensitivity', 0.0):.4f}`",
            f"- **Specificity**: `{holdout.get('Specificity', 0.0):.4f}`",
            f"- **ROC-AUC**: `{holdout.get('ROC-AUC', 0.0):.4f}`",
            f"- **ECE (5 bins)**: `{holdout.get('ECE (5 bins)', 0.0):.4f}`",
        ]
    )

    return "\n".join(lines)


def update_readme(check_only: bool = False) -> int:
    """Cập nhật hoặc kiểm tra vùng kết quả tự động trong README.md."""
    if not README_PATH.exists():
        print(f"Error: Non-existent file {README_PATH}", file=sys.stderr)
        return 1

    content = README_PATH.read_text(encoding="utf-8")

    if START_MARKER not in content or END_MARKER not in content:
        print(f"Error: README.md missing markers {START_MARKER} and {END_MARKER}", file=sys.stderr)
        return 1

    start_idx = content.find(START_MARKER) + len(START_MARKER)
    end_idx = content.find(END_MARKER)

    existing_generated = content[start_idx:end_idx].strip()
    new_generated = generate_results_markdown().strip()

    if check_only:
        if existing_generated != new_generated:
            print("README.md results section is out of date with artifacts!", file=sys.stderr)
            return 1
        print("README.md results section is up to date.")
        return 0

    updated_content = content[:start_idx] + "\n\n" + new_generated + "\n\n" + content[end_idx:]
    README_PATH.write_text(updated_content, encoding="utf-8")
    print("Successfully updated README.md results section.")
    return 0


def generate_results_markdown() -> str:
    """Render kết quả canonical từ metrics.json và evaluation artifacts."""
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    nested = metrics.get("nested_cv_subject", {})
    deployment = metrics.get("deployment_oof", {})
    protocol = metrics.get("evaluation_protocol", {})
    release_metadata_path = ARTIFACTS_DIR / "releases" / "v1.0.0" / "metadata.json"
    release_metadata = (
        json.loads(release_metadata_path.read_text(encoding="utf-8"))
        if release_metadata_path.exists()
        else {}
    )
    threshold = deployment.get(
        "decision_threshold",
        release_metadata.get("decision_threshold", "N/A"),
    )
    lines = [
        "### Kết quả nested subject-level evaluation",
        "",
        f"- **Protocol:** `{protocol.get('outer_folds', 4)} outer × {protocol.get('inner_folds', 3)} inner`, unit=`{protocol.get('unit', 'subject')}`",
        "- **Primary metric:** `Balanced Accuracy`",
        f"- **Cross-fitted subjects:** `{metrics.get('dataset', {}).get('n_subjects', 'N/A')}`; mỗi subject xuất hiện đúng một lần.",
        f"- **Pooled Balanced Accuracy:** `{nested.get('Balanced Accuracy', 0.0):.4f}`",
        f"- **Pooled Macro-F1:** `{nested.get('F1-macro', 0.0):.4f}`",
        f"- **Pooled ROC-AUC:** `{nested.get('ROC-AUC', 0.0):.4f}`",
        "",
        "### Deployment contract",
        "",
        "- `StandardScaler → L2 Logistic Regression` với `aggregation=median`.",
        f"- `C={metrics.get('selection', {}).get('C', 'N/A')}`, `class_weight={metrics.get('selection', {}).get('class_weight', 'N/A')}`.",
        f"- Full-data group-OOF threshold: `{threshold}`.",
        "- Artifact được fit trên toàn bộ 32 subject sau khi protocol khóa; chưa có external cohort.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update README.md results section from artifacts.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if README results section is up to date without modifying.",
    )
    args = parser.parse_args()
    sys.exit(update_readme(check_only=args.check))
